from gym.spaces import Discrete, Box
import numpy as np

import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))

from envs.ddp.jackal_parameter import RANGE_DICT as ddp_equation

class InfoEnv:
    """ The infomation environment contains observation space and action space infomation only
    """
    def __init__(self, config):
        env_config = config["env_config"]
        env_id = env_config["env_id"]

        if env_id.startswith("adaptive_dynamics_planning_continues_v0"):

            self.param_init = self.param_list = np.array([
                0.0302, 0.0495, 0.0608, 0.0697, 0.0771,
                0.0835, 0.0893, 0.0946, 0.0994, 0.1039,
                0.1082, 0.1122, 0.116, 0.1196, 0.1231,
                0.1264, 0.1296, 0.1327, 0.1357, 0.1386
            ])

            self.action_space = Box(
                low=np.full(20, 0.02),
                high=np.full(20, 0.18),
                dtype=np.float32
            )
        elif env_id.startswith("adaptive_dynamics_planning_functional_v0"):

            self.param_list = ['p', 'total_time', 'time_steps', 'blend']
            self.param_init = np.array([1.4, 2, 20, 0])

            self.action_space = Box(
                low=np.array([ddp_equation[k][0] for k in self.param_list]),
                high=np.array([ddp_equation[k][1] for k in self.param_list]),
                dtype=np.float32
            )
        else:
            raise NotImplementedError

        last_action_dim = len(self.param_list)

        obs_dim = 720 + last_action_dim  + 2 + 2 + 2  # 720 dim laser scan + local goal (in angle)

        self.observation_space = Box(
            low=0,
            high=env_config["kwargs"]["laser_clip"],
            shape=(obs_dim,),
            dtype=np.float32
        )

        self.config = config
