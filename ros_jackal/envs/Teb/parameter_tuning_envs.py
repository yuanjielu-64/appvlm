from typing import Any, NamedTuple
from gym.spaces import Box
import numpy as np
import rospy

from envs.Teb.Teb_base_envs import TebBase, TebBaseLaser

# A contant dict that define the ranges of parameters
RANGE_DICT = {
        "TebLocalPlannerROS/max_vel_x": [0.2, 2],
        "TebLocalPlannerROS/max_vel_x_backwards": [0.1, 0.7],
        "TebLocalPlannerROS/max_vel_theta": [0.314, 3.14],
        "TebLocalPlannerROS/dt_ref": [0.1, 0.35],
        "TebLocalPlannerROS/min_obstacle_dist": [0.05, 0.2],
        "TebLocalPlannerROS/inflation_dist": [0.01, 0.2],
        "inflation_radius": [0.1, 0.6]
}

class TebParamContinuous(TebBase):
    def __init__(
        self, 
        param_init=[2, 0.5, 3, 0.25, 0.15, 0.25, 0.2],
        param_list=["TebLocalPlannerROS/max_vel_x",
                    "TebLocalPlannerROS/max_vel_x_backwards",
                    "TebLocalPlannerROS/max_vel_theta",
                    "TebLocalPlannerROS/dt_ref",
                    "TebLocalPlannerROS/min_obstacle_dist",
                    "TebLocalPlannerROS/inflation_dist",
                    "inflation_radius"
                ],
        **kwargs
    ):
        super().__init__(**kwargs)
        self.param_list = param_list
        self.param_init = param_init

        # same as the parameters to tune
        self.action_space = Box(
            low=np.array([RANGE_DICT[k][0] for k in self.param_list]),
            high=np.array([RANGE_DICT[k][1] for k in self.param_list]),
            dtype=np.float32
        )

    def _get_info(self):
        info = dict(success=self._get_success(), params=self.params)
        info.update(super()._get_info())
        return info

    def _take_action(self, action):
        assert len(action) == len(self.param_list), "length of the params should match the length of the action"
        self.params = action
        # Set the parameters

        self.gazebo_sim.unpause()
        for param_value, param_name in zip(action, self.param_list):
            high_limit = RANGE_DICT[param_name][1]
            low_limit = RANGE_DICT[param_name][0]
            param_value = float(np.clip(param_value, low_limit, high_limit))
            self.move_base.set_navi_param(param_name, param_value, False)
            # Wait for robot to navigate for one time step
        rospy.sleep(self.time_step)
        self.gazebo_sim.pause()

class TebParamContinuousLaser(TebParamContinuous, TebBaseLaser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


