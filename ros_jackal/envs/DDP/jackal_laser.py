# jackal_laser.py
import gym
import numpy as np
from gym.spaces import Box
import math

#
from envs.DDP import JackalBase

class JackalLaser(JackalBase):
    def __init__(self, laser_clip=4, **kwargs):
        super().__init__(**kwargs)
        self.laser_clip = laser_clip

        # laser scan + goal
        obs_dim = 720 + 1
        self.observation_space = Box(
            low=0,
            high=laser_clip,
            shape=(obs_dim,),
            dtype=np.float32
        )

    def _get_observation(self):
        # observation is the 720 dim laser scan + one local goal in angle
        laser_scan = self._get_laser_scan()
        laser_scan = (laser_scan - self.laser_clip/2.) / self.laser_clip

        local_goal = self._get_local_goal()
        local_goal = local_goal / (2.0 * np.pi)

        obs = np.concatenate([laser_scan, local_goal])

        return obs

    def _get_laser_scan(self):
        """Get 720 dim laser scan
        Returns:
            np.ndarray: (720,) array of laser scan
        """
        laser_scan = self.move_base.get_laser_scan()
        laser_scan = np.array(laser_scan.ranges)
        laser_scan[laser_scan > self.laser_clip] = self.laser_clip
        return laser_scan

    def _get_local_goal(self):
        local_goal = self.jackal_ros.local_goal
        local_goal = np.array([np.arctan2(local_goal[1], local_goal[0])])

        return local_goal

    def _get_global_goal(self):
        return self.jackal_ros.get_global_goal()

    def _get_robot_state(self):
        return self.jackal_ros.get_robot_state()

    def transform_goal(self, goal_pos, pos, psi):
        """ transform goal in the robot frame
        params:
            pos_1
        """
        R_r2i = np.matrix([[np.cos(psi), -np.sin(psi), pos.x], [np.sin(psi), np.cos(psi), pos.y], [0, 0, 1]])
        R_i2r = np.linalg.inv(R_r2i)
        pi = np.matrix([[goal_pos[0]], [goal_pos[1]], [1]])
        pr = np.matmul(R_i2r, pi)
        lg = np.array([pr[0,0], pr[1, 0]])
        return lg

    def calculate_goal_direction_and_distance(self, robot_status, local_goal):
        robot_x, robot_y, robot_yaw = robot_status[0], robot_status[1], robot_status[2]  # 需要机器人朝向
        goal_x, goal_y = local_goal[0], local_goal[1]

        dx = goal_x - robot_x
        dy = goal_y - robot_y

        distance = math.sqrt(dx ** 2 + dy ** 2)

        goal_angle_world = math.atan2(dy, dx)

        relative_direction = goal_angle_world - robot_yaw

        relative_direction = math.atan2(math.sin(relative_direction), math.cos(relative_direction))

        return relative_direction, distance