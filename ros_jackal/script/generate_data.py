import os

import yaml
import pickle
from os.path import join, dirname, abspath, exists
import sys

sys.path.append(dirname(dirname(abspath(__file__))))
import torch
import gym
import numpy as np
import random
import time
import rospy
import argparse
import logging

from td3.train import initialize_policy
from envs import registration
from envs.wrappers import ShapingRewardWrapper, StackFrame

os.environ["JACKAL_LASER"] = "1"
os.environ["JACKAL_LASER_MODEL"] = "ust10"
os.environ["JACKAL_LASER_OFFSET"] = "-0.065 0 0.01"

class FileSync:
    def __init__(self, actor_id, buffer_path, actor_dir):
        self.actor_id = actor_id
        self.sync_dir = join(buffer_path, 'sync')
        os.makedirs(self.sync_dir, exist_ok=True)

        self.test_sync_dir = join(buffer_path, 'test_sync')
        os.makedirs(self.test_sync_dir, exist_ok=True)

        self.continue_file = join(self.sync_dir, 'continue.signal')
        self.actor_file = join(self.sync_dir, f'actor_{actor_id}.done')

        self.actor_dir = actor_dir

        self.last_file_time = 0
        self.train_limit = 2
        self.test_limit = 1

        self.status = 'stop'
        self.train_episode = 0
        self.test_episode = 0

    def _read_command(self):

        if not os.path.exists(self.continue_file):
            raise FileNotFoundError

        with open(self.continue_file, 'r') as f:
            command = f.readline().strip()

        self.status = command

    def _write_actor_status(self):
        if self.status == 'train':
            status = f"{self.status}:{self.train_episode}"
            with open(self.actor_file, 'w') as f:
                f.write(status)

        elif self.status == 'test':
            status = f"{self.status}:{self.test_episode}"
            with open(self.actor_file, 'w') as f:
                f.write(status)

def initialize_actor(id, BUFFER_PATH):
    rospy.logwarn(">>>>>>>>>>>>>>>>>> actor id: %s <<<<<<<<<<<<<<<<<<" %(str(id)))
    assert os.path.exists(BUFFER_PATH), BUFFER_PATH
    actor_path = join(BUFFER_PATH, 'actor_%s' %(str(id)))

    if not exists(actor_path):
        os.mkdir(actor_path) # path to store all the trajectories

    f = None
    while f is None:
        try:
            f = open(join(BUFFER_PATH, 'config.yaml'), 'r')
        except:
            rospy.logwarn("wait for critor to be initialized")
            time.sleep(2)

    config = yaml.load(f, Loader=yaml.FullLoader)

    return config

def load_policy(policy):
    f = True
    while f:
        try:
            if not os.path.exists(join(BUFFER_PATH, "policy_copy_actor")):
                policy.load(BUFFER_PATH, "policy")
            f = False
        except FileNotFoundError:
            time.sleep(1)
        except:
            logging.exception('')
            time.sleep(1)

    return policy

def get_world_name(config, id):
    world_name = config["condor_config"]["worlds"][id]
    if isinstance(world_name, int):
        world_name = "world_%d.world" %(world_name)
    return world_name

def main(id):
    actor_dir = join(BUFFER_PATH, 'actor_%s' % (str(id)))
    os.makedirs(actor_dir, exist_ok=True)

    file_sync = FileSync(id, BUFFER_PATH, actor_dir)

    config = initialize_actor(id, BUFFER_PATH)
    env_config = config['env_config']
    world_name = get_world_name(config, id)
    env_config["kwargs"]["world_name"] = world_name
    env_config["kwargs"]["WORLD_PATH"] = words

    env_config["kwargs"]["img_dir"] = file_sync.actor_dir
    env_config["kwargs"]["pid"] = id
    env_config["kwargs"]["use_vlm"] = False

    env = gym.make(env_config["env_id"], **env_config["kwargs"])

    if env_config["shaping_reward"]:
        env = ShapingRewardWrapper(env)

    env = StackFrame(env, stack_frame=env_config["stack_frame"])

    policy, _ = initialize_policy(config, env)

    print(">>>>>>>>>>>>>> Running on %s <<<<<<<<<<<<<<<<" %(world_name))

    num_trials = 1000
    ep = 0

    while ep < num_trials:
        obs = env.reset()
        ep += 1
        done = False
        policy = load_policy(policy)

        while not done:
            act = policy.select_action(obs)
            obs_new, rew, done, info = env.step(act)
            obs = obs_new

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'start an actor')
    parser.add_argument('--id', dest='actor_id', type = int, default = 97)
    #parser.add_argument('--policy_name', dest='policy_name', default="move_base")
    parser.add_argument('--policy_name', dest='policy_name', default="dwa_heurstic")
    parser.add_argument('--buffer_path', dest='buffer_path', default="../buffer/")
    parser.add_argument('--world_path', dest='world_path', default="../jackal_helper/worlds/BARN/")

    args = parser.parse_args()
    BUFFER_PATH = args.buffer_path
    WORLD_PATH = args.world_path
    words = os.path.join(*WORLD_PATH.split(os.sep)[-3:])

    if (os.path.exists(BUFFER_PATH + args.policy_name) == False):
        os.makedirs(BUFFER_PATH + args.policy_name, exist_ok=True)

    BUFFER_PATH = BUFFER_PATH + args.policy_name
    id = args.actor_id
    main(id)
