import os

import yaml
import pickle
from os.path import join, dirname, abspath, exists
import sys

# script/applr/generate_data_single.py -> script/ -> ros_jackal/
sys.path.append(dirname(dirname(dirname(abspath(__file__)))))
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

def initialize_actor(world_id, BUFFER_PATH):
    print(f"[DEBUG] initialize_actor called: world_id={world_id}, BUFFER_PATH={BUFFER_PATH}")
    print(">>>>>>>>>>>>>>>>>> actor world_id: %s <<<<<<<<<<<<<<<<<<" %(str(world_id)))
    sys.stdout.flush()
    assert os.path.exists(BUFFER_PATH), BUFFER_PATH
    actor_path = join(BUFFER_PATH, 'actor_%s' %(str(world_id)))

    if not exists(actor_path):
        os.mkdir(actor_path) # path to store all the trajectories

    print(f"[DEBUG] Loading config from: {join(BUFFER_PATH, 'config.yaml')}")
    sys.stdout.flush()

    f = None
    while f is None:
        try:
            f = open(join(BUFFER_PATH, 'config.yaml'), 'r')
        except Exception as e:
            print(f"[DEBUG] Failed to open config.yaml: {e}")
            rospy.logwarn("wait for critor to be initialized")
            time.sleep(2)

    config = yaml.load(f, Loader=yaml.FullLoader)
    print(f"[DEBUG] Config loaded successfully")
    sys.stdout.flush()

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

def get_world_name(config, world_id):
    world_name = config["condor_config"]["worlds"][world_id]
    if isinstance(world_name, int):
        world_name = "world_%d.world" %(world_id)
    return world_name

def main(world_id, planner='DWA', data_mode='auto', ros_port=11311, save_image=True, algorithm_name='Heurstic_based', num_trials=100):
    actor_dir = join(BUFFER_PATH, 'actor_%s' % (str(world_id)))
    os.makedirs(actor_dir, exist_ok=True)

    file_sync = FileSync(world_id, BUFFER_PATH, actor_dir)

    config = initialize_actor(world_id, BUFFER_PATH)
    env_config = config['env_config']
    world_name = get_world_name(config, world_id)
    env_config["kwargs"]["world_name"] = world_name
    env_config["kwargs"]["WORLD_PATH"] = words

    env_config["kwargs"]["img_dir"] = file_sync.actor_dir
    env_config["kwargs"]["pid"] = world_id
    env_config["kwargs"]["use_vlm"] = False
    env_config["kwargs"]["data_mode"] = data_mode
    env_config["kwargs"]["ros_port"] = ros_port
    env_config["kwargs"]["save_image"] = save_image
    env_config["kwargs"]["algorithm_name"] = algorithm_name

    GAZEBO_PORT_MAP = {'DWA': 12000, 'DDP': 13000, 'TEB': 14000, 'MPPI': 15000}
    gazebo_base = GAZEBO_PORT_MAP.get(planner, 12000)
    env_config["kwargs"]["gazebo_port"] = gazebo_base + world_id

    env = gym.make(env_config["env_id"], **env_config["kwargs"])

    if env_config["shaping_reward"]:
        env = ShapingRewardWrapper(env)

    env = StackFrame(env, stack_frame=env_config["stack_frame"])

    policy, _ = initialize_policy(config, env)

    print(">>>>>>>>>>>>>> Running on %s <<<<<<<<<<<<<<<<" %(world_name))

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

    env.unwrapped.soft_close()
    env.close()

if __name__ == '__main__':
    print("=" * 60)
    print("Starting HB evaluation script")
    print("=" * 60)

    parser = argparse.ArgumentParser(description = 'HB evaluation on a single world')
    parser.add_argument('--world_id', dest='world_id', type = int, default = 0,
                        help='World ID to evaluate (0-299 for BARN dataset)')
    parser.add_argument('--policy_name', dest='policy_name', default="dwa_heurstic",
                        help='Policy name (e.g., dwa_heurstic, ddp_heurstic, teb_heurstic, mppi_heurstic)')
    parser.add_argument('--buffer_path', dest='buffer_path', default="../buffer/")
    parser.add_argument('--world_path', dest='world_path', default="../jackal_helper/worlds/BARN/")
    parser.add_argument('--mode', type=str, default='auto', choices=['auto', 'manual'],
                        help='Data generation mode: auto (RL/HB) or manual (fixed parameters)')
    parser.add_argument('--ros_port', type=int, default=11311, help='ROS master port for this instance')
    parser.add_argument('--save_image', type=lambda x: x.lower() == 'true', default=True,
                        help='Whether to save costmap images (default: True)')
    parser.add_argument('--algorithm_name', type=str, default='Heurstic_based',
                        help='Algorithm name for identification (default: Heurstic_based)')
    parser.add_argument('--num_trials', type=int, default=100, help='Number of trials per world (default: 100)')

    args = parser.parse_args()
    print(f"Arguments parsed: world_id={args.world_id}, policy_name={args.policy_name}")

    BUFFER_PATH = args.buffer_path
    WORLD_PATH = args.world_path
    words = os.path.join(*WORLD_PATH.split(os.sep)[-3:])

    policy_name = args.policy_name
    planner = policy_name.split('_')[0].upper()

    if (os.path.exists(BUFFER_PATH + args.policy_name) == False):
        os.makedirs(BUFFER_PATH + args.policy_name, exist_ok=True)

    BUFFER_PATH = BUFFER_PATH + args.policy_name
    print(f"Buffer path: {BUFFER_PATH}")
    print(f"Config file: {os.path.join(BUFFER_PATH, 'config.yaml')}")
    print(f"Config exists: {os.path.exists(os.path.join(BUFFER_PATH, 'config.yaml'))}")
    print("\nInitializing ROS node...")
    sys.stdout.flush()

    world_id = args.world_id
    main(world_id, planner=planner, data_mode=args.mode, ros_port=args.ros_port, save_image=args.save_image, algorithm_name=args.algorithm_name, num_trials=args.num_trials)
