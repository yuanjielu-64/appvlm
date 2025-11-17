# jackal_laser.py
import gym
import numpy as np
from gym.spaces import Box

from envs.MPPI import JackalBase

class Visualization(JackalBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
