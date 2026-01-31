import gym
import time
import numpy as np
import os
from os.path import join
import subprocess
from gym.spaces import Box
import copy

try:  # make sure to create a fake environment without ros installed
    import rospy
    import rospkg
except ModuleNotFoundError:
    pass

from envs.utils import GazeboSimulation, mppi_MoveBase, JackalRos

GAZEBO_PORT_BASE = 15000

class JackalBase(gym.Env):
    def __init__(
        self,
        world_name="jackal_world.world",
        planner='MPPI',
        max_velocity = 1.5,
        gui=False,
        rviz_gui=False,
        init_position=None,
        goal_position=None,
        max_step=100,
        time_step=1,
        slack_reward=-1,
        failure_reward=-50,
        success_reward=0,
        collision_reward=0,
        smoothness_reward=0,
        verbose=True,
        img_dir=None,
        pid=0,
        WORLD_PATH=None,
        use_vlm = False,
        data_mode = 'auto',
        ros_port = 11311,
        gazebo_port = None,
        save_image = True,
        algorithm_name = 'Unknown'
    ):
        """Base RL env that initialize jackal simulation in Gazebo
        """
        super().__init__()
        # config
        if init_position is None:
            init_position = [-2.25, 3, 1.57]
        if goal_position is None:
            goal_position = [0, 15, 0]

        self.use_vlm = use_vlm
        self.data_mode = data_mode
        self.save_image = save_image
        self.algorithm_name = algorithm_name

        # 根据算法类型确定观测和行为模式
        self.obs_type = self._get_obs_type()

        self.rviz_gui = rviz_gui
        self.world_name = world_name
        self.gui = gui
        self.init_position = init_position
        self.goal_position = goal_position
        self.verbose = verbose
        self.time_step = time_step
        self.max_step = max_step
        self.slack_reward = slack_reward
        self.failure_reward = failure_reward
        self.success_reward = success_reward
        self.collision_reward = collision_reward
        self.smoothness_reward = smoothness_reward

        self.img_dir = img_dir
        self.p_id = pid
        self.ros_port = ros_port  # 用于 close() 时只杀自己的进程
        self.gazebo_port = gazebo_port if gazebo_port else (GAZEBO_PORT_BASE + pid)  # MPPI: 15000-15299
        self.WORLD_PATH = WORLD_PATH

        self.planner = planner
        self.max_velocity = max_velocity

        self.launch_gazebo(world_name=self.world_name, gui=self.gui, verbose=self.verbose)
        self.launch_move_base(goal_position=goal_position,
                              base_local_planner="base_local_planner/TrajectoryPlannerROS")
        self.set_start_goal_BARN(init_position, goal_position)

        # place holders
        self.action_space = None
        self.observation_space = None

        self.step_count = 0
        self.collision_count = 0
        self.collided = 0
        self.start_time = self.current_time = None

    def launch_move_base(self, goal_position, base_local_planner):
        rospack = rospkg.RosPack()
        self.BASE_PATH = rospack.get_path('jackal_helper')
        launch_file = join(self.BASE_PATH, 'launch', 'ddp_move_base_launch.launch')
        self.move_base_process = subprocess.Popen(
            ['roslaunch', launch_file, 'base_local_planner:=' + base_local_planner])

        time.sleep(5)  # Additional buffer time for costmap initialization

        self.move_base = mppi_MoveBase(goal_position=goal_position, base_local_planner=base_local_planner)

    def launch_gazebo(self, world_name, gui, verbose):
        # launch gazebo
        rospy.logwarn(">>>>>>>>>>>>>>>>>> Load world: %s <<<<<<<<<<<<<<<<<<" % (world_name))

        # ros_package_path = os.environ.get('ROS_PACKAGE_PATH', 'Not set')
        # print(f"ROS_PACKAGE_PATH: {ros_package_path}")

        rospack = rospkg.RosPack()
        self.BASE_PATH = rospack.get_path('jackal_helper')
        self.WORLD_PATH = join(self.BASE_PATH, self.WORLD_PATH)
        world_name = join(self.WORLD_PATH, world_name)

        if self.rviz_gui is False:
            launch_file = join(self.BASE_PATH, 'launch', 'gazebo_ddp.launch')
        else:
            launch_file = join(self.BASE_PATH, 'launch', 'gazebo_ddp_rviz.launch')

        planner = str(self.planner).upper().replace('-', '_')
        valid = { 'DDP', 'DWA', 'DWA_DDP', 'MPPI', 'MPPI_DDP' }
        if planner not in valid:
            raise ValueError(f"Unsupported planner: {self.planner}. Choose from {valid}")

        self.gazebo_process = subprocess.Popen([
            'roslaunch',
            launch_file,
            'world_name:=' + world_name,
            'planner:=' + planner,
            'gui:=' + ("true" if gui else "false"),
            'verbose:=' + ("true" if verbose else "false"),
        ])

        time.sleep(5)  # sleep to wait until the gazebo being created

        rospy.logwarn(">>>>>>>>>>>>>>>>>> !!Load world2: %s <<<<<<<<<<<<<<<<<<" % (world_name))

        # Initialize ROS node only if not already initialized
        if not rospy.core.is_initialized():
            rospy.init_node('gym_mppi', anonymous=True, log_level=rospy.FATAL)
            rospy.logwarn(">>>>>>>>>>>>>>>>>> ROS node initialized <<<<<<<<<<<<<<<<<<")
        else:
            rospy.logwarn(">>>>>>>>>>>>>>>>>> ROS node already initialized, skipping <<<<<<<<<<<<<<<<<<")

        rospy.set_param('/use_sim_time', True)
        rospy.logwarn(">>>>>>>>>>>>>>>>>> !!Load world3: %s <<<<<<<<<<<<<<<<<<" % (world_name))

    def set_start_goal_BARN(self, init_position, goal_position):
        """Use predefined start and goal position for BARN dataset
        """

        self.gazebo_sim = GazeboSimulation(init_position)
        self.jackal_ros = JackalRos(init_position = init_position, goal_position = goal_position, use_move_base = True, img_dir = self.img_dir, world_path = self.WORLD_PATH, id = self.p_id, use_vlm = self.use_vlm, data_mode = self.data_mode, save_image = self.save_image, algorithm_name = self.algorithm_name)
        self.start_position = init_position
        self.global_goal = goal_position
        self.local_goal = [0, 0, 0]

    def seed(self, seed):
        np.random.seed(seed)

    def reset(self):
        """reset the environment
        """
        self.step_count = 0
        self.gazebo_sim.reset()
        self.jackal_ros.reset(self.param_init)
        self.gazebo_sim.unpause()
        self._reset_move_base()
        self.jackal_ros.set_params(self.param_init)
        obs = self._get_observation()
        self.gazebo_sim.pause()

        self._reset_reward()
        self.jackal_ros.reference_state = copy.deepcopy(self.jackal_ros.state)
        self.jackal_ros.save_info(self.param_init, True, False, None)

        self.collision_count = 0
        self.traj_pos = []
        self.smoothness = 0

        # 统一的返回逻辑
        return self._format_observation(obs)

    def step(self, action):
        """
        take an action and step the environment
        """
        # 保存状态（所有算法都需要）
        self.jackal_ros.last_state = copy.deepcopy(self.jackal_ros.state)

        self.jackal_ros.save_frame()

        # 根据算法类型处理动作
        processed_action = self._process_action(action)
        self._take_action(processed_action)
        self.step_count += 1

        # 获取观测和奖励
        self.gazebo_sim.unpause()
        obs = self._get_observation()
        self.jackal_ros.set_params(action)
        self.gazebo_sim.pause()

        rew = self._get_reward()
        done, status = self._get_done()
        info = self._get_info(status)

        # 保存信息
        if done == True:
            self.jackal_ros.save_info(action, False, True, info)
        else:
            self.jackal_ros.save_info(action, False, False, info)

        pos = self.gazebo_sim.get_model_state().pose.position
        self.traj_pos.append((pos.x, pos.y))

        # 统一的返回逻辑
        return self._format_observation(obs), rew, done, info

    def _get_reward(self):
        if self.jackal_ros.get_collision():
            return self.failure_reward

        if self._get_flip_status():
            return self.failure_reward

        if self.step_count >= self.max_step:
            return self.failure_reward

        if self._get_success():
            return self.success_reward
        else:
            rew = self.slack_reward

        laser_scan = np.array(self.jackal_ros.scan.ranges)
        d = np.mean(sorted(laser_scan)[:10])
        if d < 0.15:
            penalty_ratio = (1 - d / 0.15) ** 2
            rew += self.collision_reward * penalty_ratio

        robot_pos = self.jackal_ros.state.get_robot_state()
        current_distance = self._compute_distance(
            [robot_pos[0], robot_pos[1]],
            self.global_goal
        )

        distance_progress = self.last_distance - current_distance
        rew += distance_progress * 5
        self.last_distance = current_distance

        smoothness = self._compute_angle(len(self.traj_pos) - 1)
        self.smoothness += smoothness

        # Reward clipping: 最低不低于 failure_reward
        rew = max(rew, self.failure_reward)
        return rew

    def _get_done(self):
        success = self._get_success()
        flip = self._get_flip_status()
        timeout = self.step_count >= self.max_step
        abort = self.jackal_ros.should_abort

        if abort == True:
            return True, "collision"

        if success:
            return True, "success"
        elif flip:
            return True, "flip"
        elif timeout:
            return True, "timeout"
        else:
            return False, "running"

    def _reset_reward(self):
        self.traj_pos = []
        self.collision_count = 0
        self.smoothness = 0

        # self.Y = self.jackal_ros.get_robot_state()[1]
        robot_pos = self.jackal_ros.state.get_robot_state()
        self.last_distance = self._compute_distance(
            [robot_pos[0], robot_pos[1]],
            self.global_goal
        )

    def _get_success(self):
        robot_position = np.array([self.move_base.robot_config.X,
                                   self.move_base.robot_config.Y])

        if robot_position[1] > self.goal_position[1]:
            return True

        if self.global_goal[1] == 10:
            d = 1
        else:
            d = 4

        if self._compute_distance(robot_position, self.global_goal) <= d:
            return True

        return False

    def _get_info(self, status):
        bn, nn = self.jackal_ros.get_bad_vel()

        self.collision_count += self.jackal_ros.get_collision()

        return dict(
            world=self.world_name,
            time=rospy.get_time() - self.jackal_ros.start_time,
            collision=self.collision_count,
            status=status,
            recovery= 1.0 * (bn + 0.0001) / (nn + 0.0001),
            smoothness=self.smoothness,
            img_label=self.jackal_ros.drawer.img_name,
            img_PIL=self.jackal_ros.drawer.img_PIL,
            last_state=[self.jackal_ros.last_state.v, self.jackal_ros.last_state.w]
        )

    def _compute_distance(self, p1, p2):
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    def _compute_angle(self, idx):
        def dis(x1, y1, x2, y2):
            return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        assert self.traj_pos is not None
        if len(self.traj_pos) > 2:
            x1, y1 = self.traj_pos[idx - 2]
            x2, y2 = self.traj_pos[idx - 1]
            x3, y3 = self.traj_pos[idx]
            a = - np.arccos(((x2 - x1) * (x3 - x2) + (y2 - y1) * (y3 - y2)) / dis(x1, y1, x2, y2) / dis(x2, y2, x3, y3))
        else:
            a = 0
        return a

    def _get_flip_status(self):
        robot_position = self.gazebo_sim.get_model_state().pose.position
        return robot_position.z > 0.1

    def _get_obs_type(self):
        """根据算法类型确定观测空间类型"""
        if self.algorithm_name in ['APPLR', 'Heurstic_based']:
            return 'generate'  # 返回完整观测 (laser + goal)
        elif self.algorithm_name in ['ChatGPT', 'IL', 'Qwen']:
            return 'predict'  # 只返回速度 [v, w]
        else:
            return 'generate'  # 默认

    def _format_observation(self, obs):
        """根据算法类型格式化观测"""
        if self.obs_type == 'predict':
            return [self.jackal_ros.state.v, self.jackal_ros.state.w]
        else:
            return obs

    def _process_action(self, action):
        """根据算法类型处理动作"""
        if self.algorithm_name == 'Heurstic_based':
            return self._get_heuristic_action(action)
        elif self.algorithm_name == 'APPLR':
            # MPPI的RL模式需要固定action[-2]
            action[-2] = 0.022
            return action
        else:
            return action

    def _get_heuristic_action(self, fallback_action):
        """获取启发式参数（MPPI专用）"""
        if self.jackal_ros.row is not None:
            return [
                self.jackal_ros.row["max_vel_x"],
                self.jackal_ros.row["max_vel_theta"],
                self.jackal_ros.row["nr_pairs_"],
                self.jackal_ros.row["nr_steps_"],
                self.jackal_ros.row["linear_stddev"],
                self.jackal_ros.row["angular_stddev"],
                self.jackal_ros.row["lambda"],
                self.jackal_ros.row["final_inflation"],
            ]
        else:
            return fallback_action

    def _should_save_frame(self):
        """判断是否需要保存帧数据"""
        return self.save_image

    def _take_action(self, action):
        raise NotImplementedError()

    def _get_observation(self):
        raise NotImplementedError()

    def _get_robot_state(self):
        raise NotImplementedError()

    def _reset_move_base(self):
        self.move_base.reset_robot_in_odom()
        self._clear_costmap()
        self.move_base.set_global_goal()

    def _clear_costmap(self):
        self.move_base.clear_costmap()
        rospy.sleep(0.1)
        self.move_base.clear_costmap()
        rospy.sleep(0.1)
        self.move_base.clear_costmap()

    def soft_close(self):
        self.gazebo_process.terminate()
        self.gazebo_process.wait()

        self.move_base_process.terminate()
        self.move_base_process.wait()

    def close(self):
        os.system(f"fuser -k {self.ros_port}/tcp 2>/dev/null || true")
        os.system(f"fuser -k {self.gazebo_port}/tcp 2>/dev/null || true")

        import rospy.impl.registration
        rospy.core._shutdown_flag = False
        rospy.core._in_shutdown = False
        rospy.core.is_shutdown_requested = lambda: False
        rospy.core.is_initialized = lambda: False
        rospy.impl.registration._init_node_args = None

        print("Soft close completed")

    def _get_pos_psi(self):
        pose = self.gazebo_sim.get_model_state().pose
        pos = pose.position

        q1 = pose.orientation.x
        q2 = pose.orientation.y
        q3 = pose.orientation.z
        q0 = pose.orientation.w
        psi = np.arctan2(2 * (q0 * q3 + q1 * q2), (1 - 2 * (q2 ** 2 + q3 ** 2)))
        assert -np.pi <= psi <= np.pi, psi

        return pos, psi

    def _path_coord_to_gazebo_coord(self, x, y):
        RADIUS = 0.075
        r_shift = -RADIUS - (30 * RADIUS * 2)
        c_shift = RADIUS + 5

        gazebo_x = x * (RADIUS * 2) + r_shift
        gazebo_y = y * (RADIUS * 2) + c_shift

        return (gazebo_x, gazebo_y)
