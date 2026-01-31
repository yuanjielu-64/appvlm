"""
Transformer IL 评估脚本
使用训练好的 IL 模型在 BARN 环境中评估导航性能
"""

import os
import yaml
import pickle
from os.path import join, dirname, abspath, exists
import sys

# script/IL/evaluate_IL_single.py -> script/ -> ros_jackal/
sys.path.append(dirname(dirname(dirname(abspath(__file__)))))
import gym
import numpy as np
import random
import time
import rospy
import argparse
import logging
import tf

from td3.train import initialize_policy
from envs import registration
from envs.wrappers import ShapingRewardWrapper, StackFrame
from script.IL.il_client import ILClient  # 使用 IL 客户端

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

    def wait_for_continue(self, opt_time, nav_metric, traj, id, path):
        self._read_command()

        if self.status == 'train':
            self.test_episode = 0
            self.train_episode += 1

            if self.train_episode == self.train_limit:
                self.write_buffer(opt_time, nav_metric, traj, self.train_episode, id, path, 'train')
                return False
            elif self.train_episode > self.train_limit:
                while True:
                    self._read_command()
                    if self.status == 'test' or self.status == 'pause':
                        return True
                    time.sleep(0.5)
            else:
                self.write_buffer(opt_time, nav_metric, traj, self.train_episode, id, path, 'train')
                return True

        elif self.status == 'pause':
            self.test_episode = 0
            self.train_episode = 0
            while True:
                self._read_command()
                if self.status == 'train' or self.status == 'test':
                    self.train_episode = 0
                    self.test_episode = 0
                    return True
                time.sleep(0.5)

        else:
            self.train_episode = 0
            self.test_episode += 1

            if self.test_episode == self.test_limit:
                self._write_actor_status()
                self.write_buffer(opt_time, nav_metric, traj, self.test_episode, id, self.test_sync_dir, 'test')
                return False
            elif self.test_episode > self.test_limit:
                while True:
                    self._read_command()
                    if self.status == 'train' or self.status == 'pause':
                        return True
                    time.sleep(0.5)
            else:
                self.write_buffer(opt_time, nav_metric, traj, self.test_episode, id, self.test_sync_dir, 'test')
                return True

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

    def write_buffer(self, opt_time, nav_metric, traj, ep, id, path, type):
        if not traj or len(traj[-1]) < 5 or len(traj) <= 1:
            return

        if type == 'train':
            total_reward = sum(traj[i][2] for i in range(len(traj)))
            info_dict = traj[-1][4]

            if (info_dict['recovery'] == 1.0 and info_dict['status'] == 'timeout') or (info_dict['time'] >= 70):
                error_dir = os.path.join(BUFFER_PATH, 'actor_error')
                os.makedirs(error_dir, exist_ok=True)
                error_file = os.path.join(error_dir, f'{id}.txt')
                with open(error_file, 'a') as f:
                    f.write(
                        f"Environment {id} and World_name {info_dict['world']} has KeyError in info_dict, time: {info_dict['time']}, recovery: {info_dict['recovery']}, status: {info_dict['status']}\n")
                return

            with open(join(path, "trajectory_results.txt"), 'a') as f:
                f.write(
                    f"Train: Collision: {info_dict['collision']}, Recovery: {info_dict['recovery']:.6f}, Smoothness: {info_dict['smoothness']:.6f}, Status: {info_dict['status']}, Time: {info_dict['time']:.3f} , Reward: {total_reward:.3f}, Opt_time: {opt_time:.3f} , Nav_Metric: {nav_metric:.3f} , World: {info_dict['world']}\n")

            with open(join(path, 'traj_%d.pickle' % (ep)), 'wb') as f:
                try:
                    pickle.dump(traj, f)
                except OSError as e:
                    logging.exception('Failed to dump the trajectory! %s', e)
        else:
            info_dict = traj[-1][4]

            if (info_dict['recovery'] == 1.0 and info_dict['status'] == 'timeout') or (info_dict['time'] >= 70):
                error_dir = os.path.join(BUFFER_PATH, 'actor_error')
                os.makedirs(error_dir, exist_ok=True)
                error_file = os.path.join(error_dir, f'{id}.txt')
                with open(error_file, 'a') as f:
                    f.write(
                        f"Test environment {id} has KeyError in info_dict, time: {info_dict['time']}, recovery: {info_dict['recovery']}, status: {info_dict['status']}\n")
                return

            total_reward = sum(traj[i][2] for i in range(len(traj)))

            with open(join(self.actor_dir, "trajectory_results.txt"), 'a') as f:
                f.write(
                    f"Test: Collision: {info_dict['collision']}, Recovery: {info_dict['recovery']:.6f}, Smoothness: {info_dict['smoothness']:.6f}, Status: {info_dict['status']}, Time: {info_dict['time']:.3f} , Reward: {total_reward:.3f}, Opt_time: {opt_time:.3f} , Nav_Metric: {nav_metric:.3f} , World: {info_dict['world']}\n")

            with open(join(path, 'test_%d_%d.pickle' % (id, ep)), 'wb') as f:
                try:
                    pickle.dump(traj, f)
                except OSError as e:
                    logging.exception('Failed to dump the trajectory! %s', e)


def initialize_actor(world_id, BUFFER_PATH):
    rospy.logwarn(">>>>>>>>>>>>>>>>>> actor world_id: %s <<<<<<<<<<<<<<<<<<" % (str(world_id)))
    assert os.path.exists(BUFFER_PATH), BUFFER_PATH
    actor_path = join(BUFFER_PATH, 'actor_%s' % (str(world_id)))

    if not exists(actor_path):
        os.mkdir(actor_path)

    f = None
    while f is None:
        try:
            f = open(join(BUFFER_PATH, 'config.yaml'), 'r')
        except:
            rospy.logwarn("wait for critor to be initialized")
            time.sleep(2)

    config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def compute_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def path_coord_to_gazebo_coord(x, y):
    RADIUS = 0.075
    r_shift = -RADIUS - (30 * RADIUS * 2)
    c_shift = RADIUS + 5
    gazebo_x = x * (RADIUS * 2) + r_shift
    gazebo_y = y * (RADIUS * 2) + c_shift
    return (gazebo_x, gazebo_y)


def get_score(INIT_POSITION, GOAL_POSITION, status, time, world):
    success = (status == "success")
    world = int(world.split('_')[1].split('.')[0])

    path_file_name = join(WORLD_PATH, "path_files/", "path_%d.npy" % int(world))
    path_array = np.load(path_file_name)
    path_array = [path_coord_to_gazebo_coord(*p) for p in path_array]
    path_array = np.insert(path_array, 0, (INIT_POSITION[0], INIT_POSITION[1]), axis=0)
    path_array = np.insert(path_array, len(path_array),
                           (INIT_POSITION[0] + GOAL_POSITION[0], INIT_POSITION[1] + GOAL_POSITION[1]), axis=0)

    path_length = sum(compute_distance(p1, p2) for p1, p2 in zip(path_array[:-1], path_array[1:]))

    optimal_time = path_length / 2
    actual_time = time
    nav_metric = int(success) * optimal_time / np.clip(actual_time, 2 * optimal_time, 8 * optimal_time)

    return optimal_time, nav_metric


def get_world_name():
    return "world_%d.world" % (WORLD_ID)


def main(world_id, planner='DWA', total_worlds=300, runs_per_world=2, il_url="http://localhost:6000", ros_port=11311, save_image=True, algorithm_name='IL', num_trials=1):
    actor_dir = join(BUFFER_PATH, 'actor_%s' % (str(world_id)))
    os.makedirs(actor_dir, exist_ok=True)

    file_sync = FileSync(world_id, BUFFER_PATH, actor_dir)

    config = initialize_actor(world_id, BUFFER_PATH)
    env_config = config['env_config']

    env_config["kwargs"]["WORLD_PATH"] = words
    env_config["kwargs"]["img_dir"] = file_sync.actor_dir
    env_config["kwargs"]["pid"] = world_id
    env_config["kwargs"]["use_vlm"] = True
    env_config["kwargs"]["ros_port"] = ros_port
    env_config["kwargs"]["save_image"] = save_image
    env_config["kwargs"]["algorithm_name"] = algorithm_name

    # 根据算法类型选择 Gazebo 端口池
    GAZEBO_PORT_MAP = {'DWA': 12000, 'DDP': 13000, 'TEB': 14000, 'MPPI': 15000}
    gazebo_base = GAZEBO_PORT_MAP.get(planner, 12000)
    env_config["kwargs"]["gazebo_port"] = gazebo_base + world_id

    # 使用 IL 客户端
    print(f">>>>>>>>>> Using Transformer IL service at {il_url} <<<<<<<<<<")
    il_client = ILClient(
        il_url=il_url,
        algorithm=planner,
        timeout=30.0
    )

    try:
        il_client.wait_for_service(timeout=60)
    except TimeoutError:
        rospy.logerr("IL service not available! Please start il_server.py first")
        return

    world_name = get_world_name()
    env_config["kwargs"]["world_name"] = world_name

    init_pos = env_config["kwargs"]["init_position"]
    goal_pos = env_config["kwargs"]["goal_position"]

    env = gym.make(env_config["env_id"], **env_config["kwargs"])

    if env_config["shaping_reward"]:
        env = ShapingRewardWrapper(env)

    env = StackFrame(env, stack_frame=env_config["stack_frame"])

    time.sleep(5)

    ep = 0

    current_img_id = 0

    print(">>>>>>>>>>>>>> Running on %s <<<<<<<<<<<<<<<<" % (world_name))

    while ep < num_trials:
        ep += 1
        state = env.reset()
        done = False

        while not done:
            linear_vel = state[0][0]
            angular_vel = state[0][1]
            print(f"Velocity: linear={linear_vel:.3f}, angular={angular_vel:.3f}")

            # 构建图像路径
            image_name = f"VLM_{il_client.img_id:06d}.png"
            image_path = os.path.join(file_sync.actor_dir, image_name)

            if os.path.exists(image_path):
                result = il_client.infer_from_server(image_path, linear_vel, angular_vel)

                if result and result.get('success'):
                    act = result['parameters_array']
                else:
                    act = None
            else:
                rospy.logwarn(f"Image not found: {image_path}")
                act = None

            if act is None:
                act = env_config["kwargs"]["param_init"]

            act = act[:-2]
            act[-2] = 0.022
            state, rew, done, info = env.step(act)

            if "img_label" in info and info["img_label"] is not None:
                # img_label 格式: "FTRL_000123.png"，提取数字ID
                import re
                match = re.match(r'[A-Z]+_(\d+)\.png', info["img_label"])
                if match:
                    current_img_id = int(match.group(1))
                else:
                    current_img_id += 1
            else:
                current_img_id += 1

            if done:
                current_img_id += 1

    env.unwrapped.soft_close()
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run IL evaluation on BARN worlds')
    parser.add_argument('--id', dest='actor_id', type=int, default=0)
    parser.add_argument('--world_idx', type=int, default=165)
    parser.add_argument('--policy_name', dest='policy_name', default="mppi_il")
    parser.add_argument('--buffer_path', dest='buffer_path', default="../../buffer/")
    parser.add_argument('--world_path', dest='world_path', default="../../jackal_helper/worlds/BARN1/")
    parser.add_argument('--total_worlds', type=int, default=300, help='Total number of worlds to run')

    parser.add_argument('--il_url', type=str, default="http://localhost:6005", help='IL service URL')
    parser.add_argument('--ros_port', type=int, default=11311, help='ROS master port for this instance')
    parser.add_argument('--save_image', type=lambda x: x.lower() == 'true', default=True,
                        help='Whether to save costmap images (default: True)')
    parser.add_argument('--algorithm_name', type=str, default='IL',
                        help='Algorithm name for identification (default: IL)')
    parser.add_argument('--num_trials', type=int, default=1, help='Number of trials per world (default: 1)')

    args = parser.parse_args()

    BUFFER_PATH = args.buffer_path
    WORLD_PATH = args.world_path
    WORLD_ID = args.world_idx

    # 从 policy_name 提取算法类型 (e.g., "ddp_il" -> "DDP")
    policy_name = args.policy_name
    planner = policy_name.split('_')[0].upper()

    words = os.path.join(*WORLD_PATH.split(os.sep)[-3:])

    if not os.path.exists(BUFFER_PATH + args.policy_name):
        os.makedirs(BUFFER_PATH + args.policy_name, exist_ok=True)

    BUFFER_PATH = BUFFER_PATH + args.policy_name
    world_id = args.actor_id

    main(world_id, planner=planner, total_worlds=args.total_worlds, il_url=args.il_url, ros_port=args.ros_port, save_image=args.save_image, algorithm_name=args.algorithm_name, num_trials=args.num_trials)
