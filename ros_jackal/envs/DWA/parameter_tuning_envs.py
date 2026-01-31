from typing import Any, NamedTuple
from gym.spaces import Box
import numpy as np
import rospy

from envs.DWA.dwa_base_envs import DWABase, DWABaseLaser

# A contant dict that define the ranges of parameters
RANGE_DICT = {
    'TrajectoryPlannerROS/max_vel_x': [0.2, 2],
    'TrajectoryPlannerROS/max_vel_theta': [0.314, 3.14],
    'TrajectoryPlannerROS/vx_samples': [4, 12],
    'TrajectoryPlannerROS/vtheta_samples': [8, 40],
    'TrajectoryPlannerROS/path_distance_bias': [0.1, 1.5],
    'TrajectoryPlannerROS/goal_distance_bias': [0.1, 2],
    'inflation_radius': [0.1, 0.6],
}

class DWAParamContinuous(DWABase):
    def __init__(
        self, 
        param_init=[0.5, 1.57, 6, 20, 0.75, 1, 0.3],
        param_list=['TrajectoryPlannerROS/max_vel_x', 
                    'TrajectoryPlannerROS/max_vel_theta', 
                    'TrajectoryPlannerROS/vx_samples', 
                    'TrajectoryPlannerROS/vtheta_samples', 
                    'TrajectoryPlannerROS/path_distance_bias', 
                    'TrajectoryPlannerROS/goal_distance_bias', 
                    'inflation_radius'],
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

    def _take_action(self, action):
        assert len(action) == len(self.param_list), "length of the params should match the length of the action"

        clipped_action = []
        for param_value, param_name in zip(action, self.param_list):
            low, high = RANGE_DICT[param_name]
            if (param_name == "TrajectoryPlannerROS/vx_samples"):
                param_value = int(param_value)
            if (param_name == "TrajectoryPlannerROS/vtheta_samples"):
                param_value = int(param_value)
            clipped_value = np.clip(param_value, low, high)
            clipped_action.append(clipped_value)

        self.params = clipped_action
        self.gazebo_sim.unpause()

        # Set parameters
        self.jackal_ros.set_params(clipped_action)

        # Special handling for inflation_radius
        for param_value, param_name in zip(clipped_action, self.param_list):
            self.move_base.set_navi_param(param_name, float(param_value))

        rospy.sleep(self.time_step)
        self.gazebo_sim.pause()
        self.jackal_ros.last_action = clipped_action


class DWAParamContinuousLaser(DWAParamContinuous, DWABaseLaser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

