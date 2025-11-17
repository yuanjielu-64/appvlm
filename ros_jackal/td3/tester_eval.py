from train import initialize_policy
import os
from os.path import dirname, abspath, join, exists
import gym
import torch
import argparse
from gitdb.util import mkdir
import numpy as np
import rospy
import time
import yaml
import pickle
import logging
import shutil
import sys
import csv

sys.path.append(dirname(dirname(abspath(__file__))))

from envs.wrappers import ShapingRewardWrapper, StackFrame

def compute_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def path_coord_to_gazebo_coord(x, y):
    RADIUS = 0.075
    r_shift = -RADIUS - (30 * RADIUS * 2)
    c_shift = RADIUS + 5

    gazebo_x = x * (RADIUS * 2) + r_shift
    gazebo_y = y * (RADIUS * 2) + c_shift

    return (gazebo_x, gazebo_y)

def initialize_config(BUFFER_PATH):

    assert os.path.exists(BUFFER_PATH), BUFFER_PATH

    f = None
    while f is None:
        try:
            f = open(join(BUFFER_PATH, 'config.yaml'), 'r')
        except:
            rospy.logwarn("wait for critor to be initialized")
            time.sleep(2)

    config = yaml.load(f, Loader=yaml.FullLoader)

    if TEST_PATH is not None:

        os.makedirs(TEST_PATH, exist_ok=True)

        shutil.copy(join(BUFFER_PATH, 'config.yaml'), join(TEST_PATH, 'config.yaml'))
        shutil.copy(join(BUFFER_PATH, 'policy_actor'), join(TEST_PATH, 'policy_actor'))
        shutil.copy(join(BUFFER_PATH, 'policy_noise'), join(TEST_PATH, 'policy_noise'))

        rospy.loginfo(f"Config file copied to {TEST_PATH}")

    f.close()

    return config

def get_score(INIT_POSITION, GOAL_POSITION, status, time, world):
    success = False

    if status == "success":
        success = True
    else:
        success = False

    world = int(world.split('_')[1].split('.')[0])

    path_file_name = join(WORLD_PATH, "path_files/", "path_%d.npy" % int(world))
    path_array = np.load(path_file_name)
    path_array = [path_coord_to_gazebo_coord(*p) for p in path_array]
    path_array = np.insert(path_array, 0, (INIT_POSITION[0], INIT_POSITION[1]), axis=0)
    path_array = np.insert(path_array, len(path_array),
                           (INIT_POSITION[0] + GOAL_POSITION[0], INIT_POSITION[1] + GOAL_POSITION[1]), axis=0)
    path_length = 0
    for p1, p2 in zip(path_array[:-1], path_array[1:]):
        path_length += compute_distance(p1, p2)

    optimal_time = path_length / 2
    actual_time = time
    nav_metric = int(success) * optimal_time / np.clip(actual_time, 2 * optimal_time, 8 * optimal_time)

    return optimal_time, nav_metric

def write_buffer(init_pos, goal_pos, traj, ep, id):
   total_reward = 0

   for i in range(len(traj)):
       rew = traj[i][2]
       total_reward += rew

   if RUN_BASELINE == True:
       file_name = join(TEST_PATH, f"baseline_results_{id}.csv")
       method_type = "baseline"
   else:
       file_name = join(TEST_PATH, f"test_results_{id}.csv")
       method_type = "predict"

   info_dict = traj[-1][-1]

   opt_time, nav_metric = get_score(init_pos, goal_pos, info_dict['status'], info_dict['time'], info_dict['world'])

   if not os.path.exists(file_name):
       with open(file_name, 'w', newline='') as f:
           writer = csv.writer(f)
           writer.writerow(['Method', 'Episode', 'Collision', 'Recovery', 'Smoothness', 'Status', 'Time', 'Reward', 'World','optimal_time', 'nav_metric'])

   with open(file_name, 'a', newline='') as f:
       writer = csv.writer(f)
       writer.writerow([method_type, ep, info_dict['collision'], info_dict['recovery'], info_dict['smoothness'],
                        info_dict['status'], info_dict['time'], total_reward, info_dict['world'], opt_time, nav_metric])

def get_world_name(config, id):
    world_name = config["condor_config"]["test_worlds"][id]
    if isinstance(world_name, int):
        world_name = "world_%d.world" %(world_name)
    return world_name

def load_policy(policy):

    policy.load(TEST_PATH, "policy")
    policy.exploration_noise = 0
    return policy

def main(args):
    config = initialize_config(BUFFER_PATH)
    num_trials = config["condor_config"]["num_trials"]
    env_config = config['env_config']
    world_name = get_world_name(config, args.id)

    time_interval = np.array([
        0.0302, 0.0495, 0.0608, 0.0697, 0.0771,
        0.0835, 0.0893, 0.0946, 0.0994, 0.1039,
        0.1082, 0.1122, 0.116, 0.1196, 0.1231,
        0.1264, 0.1296, 0.1327, 0.1357, 0.1386
    ])

    env_config["kwargs"]["world_name"] = world_name

    env_config["kwargs"]["max_step"] = 200

    init_pos = env_config["kwargs"]["init_position"]
    goal_pos = env_config["kwargs"]["goal_position"]

    env = gym.make(env_config["env_id"], **env_config["kwargs"])
    if env_config["shaping_reward"]:
        env = ShapingRewardWrapper(env)
    env = StackFrame(env, stack_frame=env_config["stack_frame"]) 

    policy, _ = initialize_policy(config, env)
    policy = load_policy(policy)

    print(">>>>>>>>>>>>>> Running on %s <<<<<<<<<<<<<<<<" %(world_name))
    ep = 0
    while ep < num_trials:
        obs = env.reset()
        ep += 1
        traj = []
        done = False

        while not done:
            if RUN_BASELINE == False:
                actions = policy.select_action(obs)
            else:
                actions = time_interval

            obs_new, rew, done, info = env.step(actions)
            info["world"] = world_name
            traj.append([None, None, rew, done, info])  # For testing, we only need rew and ep_length
            obs = obs_new

            # _debug_print_robot_status(env, len(traj), rew)

        write_buffer(init_pos, goal_pos, traj, ep, args.id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'start an tester')
    parser.add_argument('--id', dest='id', type = int, default = 7)
    parser.add_argument('--policy_name', dest='policy_name', default="ddp_function")
    parser.add_argument('--test_id', dest='test_id', default="0")
    parser.add_argument('--buffer_path', dest='buffer_path', default="../buffer/")
    parser.add_argument('--world_path', dest='world_path', default="../jackal_helper/worlds/BARN/")
    parser.add_argument('--baseline', dest='baseline', type=str, default='false')

    args = parser.parse_args()

    RUN_BASELINE = args.baseline.lower() == 'true'
    BUFFER_PATH = args.buffer_path
    WORLD_PATH = args.world_path

    if (os.path.exists(BUFFER_PATH + args.policy_name) == False):
        os.makedirs(BUFFER_PATH + args.policy_name, exist_ok=True)

    BUFFER_PATH = BUFFER_PATH + args.policy_name

    if (os.path.exists(BUFFER_PATH + '/test' + args.test_id) == False):
        mkdir(BUFFER_PATH + '/test'+ args.test_id)

    TEST_PATH = BUFFER_PATH + '/test' + args.test_id

    main(args)
