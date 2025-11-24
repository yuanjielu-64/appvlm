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
    ):
        """Base RL env that initialize jackal simulation in Gazebo
        """
        super().__init__()
        # config
        if init_position is None:
            init_position = [-2.25, 3, 1.57]
        if goal_position is None:
            goal_position = [0, 15, 0]

        self.use_RL = True
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

        # 等待move_base启动和costmap初始化
        time.sleep(3)

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

        if self.rviz_gui == False:
            launch_file = join(self.BASE_PATH, 'launch', 'ddp_launch.launch')
        else:
            launch_file = join(self.BASE_PATH, 'launch', 'ddp_launch_rviz.launch')

        if self.planner == 'ddp':
            p1 = "RunMP"
            p2 = "DDP"
        elif self.planner == 'dwa':
            p1 = "RunDWA"
            p2 = "DDPDWAPlanner"
        elif self.planner == 'MPPI':
            p1 = "RunMPPI"
            p2 = "MPPIPlanner"
        else:
            raise FileNotFoundError

        self.gazebo_process = subprocess.Popen(['roslaunch',
                                                launch_file,
                                                'world_name:=' + world_name,
                                                'ddp_param1:=' + p1,
                                                'ddp_param2:=' + p2,
                                                'gui:=' + ("true" if gui else "false"),
                                                'verbose:=' + ("true" if verbose else "false"),
                                                ])

        time.sleep(5)  # sleep to wait until the gazebo being created

        # initialize the node for gym env
        rospy.init_node('gym', anonymous=True, log_level=rospy.FATAL)
        rospy.set_param('/use_sim_time', True)

    def set_start_goal_BARN(self, init_position, goal_position):
        """Use predefined start and goal position for BARN dataset
        """

        self.gazebo_sim = GazeboSimulation(init_position)
        self.jackal_ros = JackalRos(init_position = init_position, goal_position = goal_position, use_move_base = True, img_dir = self.img_dir, world_path = self.WORLD_PATH, id = self.p_id)
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
        self.jackal_ros.set_params(self.param_init)
        self.gazebo_sim.pause()

        self._reset_reward()
        self.jackal_ros.reference_state = copy.deepcopy(self.jackal_ros.state)
        self.jackal_ros.save_info(self.param_init, True, False, None)

        self.collision_count = 0
        self.traj_pos = []
        self.smoothness = 0
        return obs

    def step(self, action):
        """
        take an action and step the environment
        """

        alg = "RL" if (self.jackal_ros.iteration % 2 == 0) else "HB"

        result = self.jackal_ros.save_frame()

        self.jackal_ros.last_state = copy.deepcopy(self.jackal_ros.state)

        if alg == "RL":
            action_0 = action
            self._take_action(action_0)
        else:
            if self.jackal_ros.row != None:
                action_0 = [self.jackal_ros.row["max_vel_x"],
                          self.jackal_ros.row["max_vel_theta"],
                          self.jackal_ros.row["nr_pairs_"],
                          self.jackal_ros.row["nr_steps_"],
                          self.jackal_ros.row["linear_stddev"],
                          self.jackal_ros.row["angular_stddev"],
                          self.jackal_ros.row["lambda"],
                          self.jackal_ros.row["final_inflation"],
                          ]

                self._take_action(action_0)

            else:
                action_0 = action
                self._take_action(action_0)

        self.step_count += 1

        self.gazebo_sim.unpause()
        obs = self._get_observation()
        self.jackal_ros.set_params(action)
        self.gazebo_sim.pause()

        rew = self._get_reward()
        done, status = self._get_done()
        info = self._get_info(status)

        if done == True:
            self.jackal_ros.save_info(action, False, True, info)
        else:
            self.jackal_ros.save_info(action, False, False, info)

        pos = self.gazebo_sim.get_model_state().pose.position
        self.traj_pos.append((pos.x, pos.y))
        return obs, rew, done, info

    def _get_reward(self):

        if self.jackal_ros.get_collision():
            return self.failure_reward
        if self.step_count >= self.max_step:  # or self._get_flip_status():
            return self.failure_reward

        if self._get_success():
            return self.success_reward
        else:
            rew = self.slack_reward

        laser_scan = np.array(self.jackal_ros.scan.ranges)
        d = np.mean(sorted(laser_scan)[:10])
        if d < 0.05:
            penalty_ratio = (1 - d / 0.05) ** 2
            rew += self.collision_reward * penalty_ratio

        robot_pos = self.jackal_ros.state.get_robot_state()
        current_distance = self._compute_distance(
            [robot_pos[0], robot_pos[1]],
            self.global_goal
        )

        distance_progress = self.last_distance - current_distance
        rew += distance_progress * 10
        self.last_distance = current_distance

        smoothness = self._compute_angle(len(self.traj_pos) - 1)
        self.smoothness += smoothness

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
            smoothness=self.smoothness
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

    def close(self):
        # These will make sure all the ros processes being killed
        os.system("killall -9 rosmaster")
        os.system("killall -9 gzclient")
        os.system("killall -9 gzserver")
        os.system("killall -9 roscore")

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
