# jackal_laser.py
import gym
import numpy as np
from gym.spaces import Box

try:  # make sure to create a fake environment without ros installed
    import rospy
    from geometry_msgs.msg import Twist
except ModuleNotFoundError:
    pass

from envs.DDP import JackalBase

RANGE_DICT = {
    'max_vel_x': [0.0, 2],
    "max_vel_theta": [0.314, 3.14],
    "nr_pairs_": [400, 800],
    "dist_local_goal": [1, 6],
    "distance": [0.01, 0.4],
    "robot_radius": [0.01, 0.15],
    "inflation_radius": [0.1, 0.6],
}

class Parameters(JackalBase):
    def __init__(
            self,
            param_init=[1.5, 3, 600, 4, 0.1, 0.02, 0.25],
            param_list=["max_vel_x",
                 "max_vel_theta",
                 "nr_pairs_",
                 "dist_local_goal",
                 "distance",
                 "robot_radius",
                 "inflation_radius"],
            **kwargs
    ):

        super().__init__(**kwargs)

        self._cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        self.param_list = param_list
        self.param_init = param_init

        self.action_space = Box(
            low=np.array([RANGE_DICT[k][0] for k in self.param_list]),
            high=np.array([RANGE_DICT[k][1] for k in self.param_list]),
            dtype=np.float32
        )

    def _take_action(self, action):
        assert len(action) == len(self.param_list), "length of the params should match the length of the action"
        self.params = action

        self.gazebo_sim.unpause()

        self.jackal_ros.set_params(action)

        for param_value, param_name in zip(action, self.param_list):
            if (param_name == 'inflation_radius'):
                high_limit = RANGE_DICT[param_name][1]
                low_limit = RANGE_DICT[param_name][0]
                param_value = float(np.clip(param_value, low_limit, high_limit))
                self.move_base.set_navi_param(param_name, param_value)
        # Wait for robot to navigate for one time step
        rospy.sleep(self.time_step)
        self.gazebo_sim.pause()

