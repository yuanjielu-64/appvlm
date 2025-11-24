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
    "distance": [0.01, 0.4],
    "robot_radius": [0.01, 0.15],
    "inflation_radius": [0.1, 0.6],
}

class Parameters(JackalBase):
    def __init__(
            self,
            param_init=[1.5, 3, 600, 0.1, 0.02, 0.25],
            param_list=["max_vel_x",
                 "max_vel_theta",
                 "nr_pairs_",
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

        # Clip all actions to RANGE_DICT limits
        clipped_action = []
        for param_value, param_name in zip(action, self.param_list):
            low, high = RANGE_DICT[param_name]
            clipped_value = np.clip(param_value, low, high)
            clipped_action.append(clipped_value)

        self.params = clipped_action
        self.gazebo_sim.unpause()

        # Set parameters
        self.jackal_ros.set_params(clipped_action)

        # Special handling for inflation_radius
        for param_value, param_name in zip(clipped_action, self.param_list):
            if param_name == 'inflation_radius':
                self.move_base.set_navi_param(param_name, float(param_value))

        rospy.sleep(self.time_step)
        self.gazebo_sim.pause()
        self.jackal_ros.last_action = clipped_action

