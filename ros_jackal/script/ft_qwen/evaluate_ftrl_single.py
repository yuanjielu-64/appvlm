"""
FTRL数据收集脚本 - 使用VLM+DPT Actor收集经验数据

数据收集流程:
1. 启动 qwen_server.py (script/qwen/) - 提供VLM+DPT推理服务
2. 本脚本通过 QwenClient 调用服务获取导航参数
3. 保存轨迹数据 (obs, action, reward, next_obs, done) 到 buffer/
4. 收集的数据用于 rlft/train.py 进行TD3强化学习训练

注意:
- 数据收集阶段使用预训练的VLM+DPT (监督学习checkpoint)
- 通过 qwen_server.py 提供推理服务 (端口默认 5000-5004)
- 不使用 ftrl_server.py (那个是训练后加载新Actor用的)
"""
import os
import yaml
import pickle
from os.path import join, dirname, abspath, exists
import sys

from timm.models.dpn import dpn107

# script/ft_qwen/evaluate_ftrl_single.py -> script/ -> ros_jackal/
sys.path.append(dirname(dirname(dirname(abspath(__file__)))))

import gym
import numpy as np
import time
import rospy
import argparse
import logging

from envs import registration
from envs.wrappers import ShapingRewardWrapper, StackFrame

# FTRL客户端 (HTTP调用VLM+DPT)
from script.ft_qwen.qwen_client import QwenClient

os.environ["JACKAL_LASER"] = "1"
os.environ["JACKAL_LASER_MODEL"] = "ust10"
os.environ["JACKAL_LASER_OFFSET"] = "-0.065 0 0.01"


class FileSync:
    """
    文件同步器 - 管理训练/测试状态和轨迹保存

    与APPLR保持一致的接口，方便复用训练框架
    """
    def __init__(self, actor_id, buffer_path, actor_dir, TRAIN_LIMIT, TEST_LIMIT):
        self.actor_id = actor_id
        self.sync_dir = join(buffer_path, 'sync')
        os.makedirs(self.sync_dir, exist_ok=True)

        self.test_sync_dir = join(buffer_path, 'test_sync')
        os.makedirs(self.test_sync_dir, exist_ok=True)

        self.continue_file = join(self.sync_dir, 'continue.signal')
        self.actor_file = join(self.sync_dir, f'actor_{actor_id}.done')

        self.actor_dir = actor_dir

        self.last_file_time = 0
        self.train_limit = TRAIN_LIMIT
        self.test_limit = TEST_LIMIT

        self.status = 'stop'
        self.train_episode = 0
        self.test_episode = 0

    def wait_for_continue(self, opt_time, nav_metric, traj, world_id, path):
        self._read_command()

        if self.status == 'train':
            self.test_episode = 0
            self.train_episode += 1

            # 保存轨迹
            self.write_buffer(opt_time, nav_metric, traj, self.train_episode, world_id, path, 'train')

            # 达到训练限制后退出
            if self.train_episode >= self.train_limit:
                return False
            else:
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

        else:  # test mode
            self.train_episode = 0
            self.test_episode += 1

            self.write_buffer(opt_time, nav_metric, traj, self.test_episode, world_id, self.test_sync_dir, 'test')
            self._write_actor_status()

            if self.test_episode >= self.test_limit:
                return False
            else:
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
        elif self.status == 'test':
            status = f"{self.status}:{self.test_episode}"
        else:
            return

        with open(self.actor_file, 'w') as f:
            f.write(status)

    def write_buffer(self, opt_time, nav_metric, traj, ep, world_id, path, type):
        """
        保存轨迹数据

        FTRL特殊处理:
        - 保存costmap图像路径或PIL.Image (供VLM训练使用)
        - 保存action (7个导航参数)
        - 保存reward, done等标准RL信息
        """
        if not traj or len(traj) <= 1 or len(traj[-1]) < 5:
            return

        info_dict = traj[-1][4]

        # 过滤异常数据
        if (info_dict['recovery'] == 1.0 and info_dict['status'] == 'timeout') or (info_dict['time'] >= 70):
            error_dir = os.path.join(BUFFER_PATH, 'actor_error')
            os.makedirs(error_dir, exist_ok=True)
            error_file = os.path.join(error_dir, f'{world_id}.txt')
            with open(error_file, 'a') as f:
                if type == 'train':
                    f.write(f"Environment {world_id} and World_name {info_dict['world']} has KeyError, "
                           f"time: {info_dict['time']}, recovery: {info_dict['recovery']}, status: {info_dict['status']}\n")
                else:
                    f.write(f"Test environment {world_id} has KeyError, "
                           f"time: {info_dict['time']}, recovery: {info_dict['recovery']}, status: {info_dict['status']}\n")
            return

        # 计算总奖励
        total_reward = sum(t[2] for t in traj)

        # 写入轨迹结果摘要
        result_file = join(path, "trajectory_results.txt") if type == 'train' else join(self.actor_dir, "trajectory_results.txt")
        with open(result_file, 'a') as f:
            f.write(f"{type.capitalize()}: Collision: {info_dict['collision']}, "
                   f"Recovery: {info_dict['recovery']:.6f}, Smoothness: {info_dict['smoothness']:.6f}, "
                   f"Status: {info_dict['status']}, Time: {info_dict['time']:.3f}, "
                   f"Reward: {total_reward:.3f}, Opt_time: {opt_time:.3f}, "
                   f"Nav_Metric: {nav_metric:.3f}, World: {info_dict['world']}\n")

        # 保存轨迹pickle (供RLFT训练使用)
        if type == 'train':
            pickle_file = join(path, f'traj_{ep}.pickle')
        else:
            pickle_file = join(path, f'test_{world_id}_{ep}.pickle')

        try:
            with open(pickle_file, 'wb') as f:
                pickle.dump(traj, f)
        except OSError as e:
            logging.exception(f'Failed to dump trajectory: {e}')


def initialize_actor(world_id, BUFFER_PATH):
    """初始化actor配置"""
    print(f"[DEBUG] initialize_actor called: world_id={world_id}, BUFFER_PATH={BUFFER_PATH}")
    print(">>>>>>>>>>>>>>>>>> actor world_id: %s <<<<<<<<<<<<<<<<<<" % (str(world_id)))
    sys.stdout.flush()

    assert os.path.exists(BUFFER_PATH), BUFFER_PATH
    actor_path = join(BUFFER_PATH, 'actor_%s' % (str(world_id)))

    if not exists(actor_path):
        os.mkdir(actor_path)

    config_path = join(BUFFER_PATH, 'config.yaml')
    rospy.logwarn(f"Looking for config.yaml at: {abspath(config_path)}")

    f = None
    while f is None:
        try:
            f = open(config_path, 'r')
            rospy.logwarn(f"Successfully opened config.yaml from: {abspath(config_path)}")
        except Exception as e:
            rospy.logwarn(f"Waiting for config to be initialized at {abspath(config_path)}, error: {e}")
            time.sleep(2)

    config = yaml.load(f, Loader=yaml.FullLoader)
    print(f"[DEBUG] Config loaded successfully")

    return config


def initialize_ftrl_client(server_url, planner):
    print(f"    >>>> Connecting to Qwen VLM+DPT service: {server_url}")
    print(f"    >>>> Algorithm: {planner}")

    client = QwenClient(
        qwen_url=server_url,
        algorithm=planner,
        timeout=30.0
    )

    client.wait_for_service(timeout=60)

    algorithms = client.list_algorithms()
    print(f"    >>>> Supported algorithms: {algorithms}")

    return client


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
    """计算导航性能指标"""
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


def get_world_name(config, world_id):
    world_name = config["condor_config"]["worlds"][world_id]
    if isinstance(world_name, int):
        world_name = "world_%d.world" % (world_name)
    return world_name

def _update_reward(traj):

    failure_reward = traj[-1][2]
    failure_steps = min(4, len(traj))

    for i in range(failure_steps):
        step_idx = len(traj) - 1 - i
        penalty_ratio = 0.5 ** i
        adjusted_reward = failure_reward * penalty_ratio
        traj[step_idx][2] = adjusted_reward

    return traj


def main(world_id, server_url="http://localhost:6000", planner='DDP'):
    """
    主函数 - FTRL数据收集

    关键修改:
    1. 不再本地创建policy，改用HTTP调用ftrl_client
    2. 保存轨迹供RLFT训练使用
    3. 支持定期重新连接以加载最新的Actor
    """
    actor_dir = join(BUFFER_PATH, 'actor_%s' % (str(world_id)))
    os.makedirs(actor_dir, exist_ok=True)

    print(f"BUFFER_PATH (relative): {BUFFER_PATH}")
    print(f"BUFFER_PATH (absolute): {os.path.abspath(BUFFER_PATH)}")
    print(f"actor_dir (relative): {actor_dir}")
    print(f"actor_dir (absolute): {os.path.abspath(actor_dir)}")
    print(f"Current working directory: {os.getcwd()}")

    file_sync = FileSync(world_id, BUFFER_PATH, actor_dir, TRAIN_LIMIT, TEST_LIMIT)

    config = initialize_actor(world_id, BUFFER_PATH)
    env_config = config['env_config']

    world_name = get_world_name(config, world_id)
    env_config["kwargs"]["world_name"] = world_name
    env_config["kwargs"]["WORLD_PATH"] = words

    init_pos = env_config["kwargs"]["init_position"]
    goal_pos = env_config["kwargs"]["goal_position"]

    # 环境配置 (VLM模式)
    env_config["kwargs"]["img_dir"] = actor_dir
    env_config["kwargs"]["pid"] = world_id
    env_config["kwargs"]["use_vlm"] = True  # 重要：启用VLM模式
    env_config["kwargs"]["save_image"] = True  # 保存costmap图像
    env_config["kwargs"]["algorithm_name"] = "FTRL"

    # Gazebo端口映射
    GAZEBO_PORT_MAP = {'DWA': 12000, 'DDP': 13000, 'TEB': 14000, 'MPPI': 15000}
    gazebo_base = GAZEBO_PORT_MAP.get(planner, 13000)
    env_config["kwargs"]["gazebo_port"] = gazebo_base + world_id

    # 创建环境
    env = gym.make(env_config["env_id"], **env_config["kwargs"])
    if env_config.get("shaping_reward", False):
        env = ShapingRewardWrapper(env)
    env = StackFrame(env, stack_frame=env_config.get("stack_frame", 1))

    # 初始化FTRL客户端 (连接到VLM+DPT服务)
    # 服务端: script/qwen/qwen_server.py (使用监督学习checkpoint)
    print(f">>>>>>>>>>>>>> Connecting to Qwen VLM+DPT service: {server_url} <<<<<<<<<<<<<<<<")
    ftrl_client = initialize_ftrl_client(server_url, planner)

    flag = True
    print(f">>>>>>>>>>>>>> Running on {world_name} <<<<<<<<<<<<<<<<")

    current_img_id = 0

    while flag:
        obs = env.reset()
        traj = []
        done = False

        while not done and flag:
            linear_vel = obs[0][0]
            angular_vel = obs[0][1]

            image_name = f"FTRL_{current_img_id:06d}.png"
            image_path = os.path.join(actor_dir, image_name)

            if os.path.exists(image_path):

                result = ftrl_client.infer_from_server(
                    image_path=image_path,
                    linear_vel=linear_vel,
                    angular_vel=angular_vel,
                    algorithm=planner
                )

                if result and result.get('success'):
                    act = ftrl_client.get_parameters_array(result)
                else:
                    act = None
            else:
                rospy.logwarn(f"Image not found: {image_path}")
                act = None

            if act is None:
                act = env_config["kwargs"]["param_init"]

            act = act[:-2]

            obs_new, rew, done, info = env.step(act)
            info["world"] = world_name

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

            # 保存轨迹: [obs, action, reward, done, info, opt_time, nav_metric]
            traj.append([obs, act, rew, done, info, 0, 0])
            obs = obs_new

        if flag:
            info_dict = traj[-1][4]
            opt_time, nav_metric = get_score(init_pos, goal_pos,
                                            info_dict['status'],
                                            info_dict['time'],
                                            info_dict['world'])

            traj[-1][5] = opt_time
            traj[-1][6] = nav_metric

            if traj[-1][3] == False or traj[-1][4]['collision'] >= 1:
                traj = _update_reward(traj)
        else:
            opt_time = nav_metric = 0

        flag = file_sync.wait_for_continue(opt_time, nav_metric, traj, world_id, actor_dir)

    env.unwrapped.soft_close()
    env.close()


if __name__ == '__main__':
    print("=" * 60)
    print("Starting FTRL data collection script (VLM+DPT)")
    print("=" * 60)

    parser = argparse.ArgumentParser(description='FTRL data collection using VLM+DPT Actor')
    parser.add_argument('--id', dest='actor_id', type=int, default=0,
                       help='Actor ID (world ID)')
    parser.add_argument('--server_url', dest='server_url',
                       default="http://192.168.1.175:6000",
                       help='FTRL service URL (default: http://localhost:5000)')
    parser.add_argument('--policy_name', dest='policy_name', default="ddp_rlft",
                       help='Policy name (e.g., dwa_ftrl, ddp_ftrl, teb_ftrl, mppi_ftrl)')
    parser.add_argument('--buffer_path', dest='buffer_path', default="../../buffer/")
    parser.add_argument('--world_path', dest='world_path', default="../../jackal_helper/worlds/BARN/")
    parser.add_argument('--train_limit',  type=int, default=3)
    parser.add_argument('--test_limit', type=int, default=1)

    args = parser.parse_args()
    print(f"Arguments parsed: actor_id={args.actor_id}, server_url={args.server_url}")

    BUFFER_PATH = args.buffer_path
    print(f"[DEBUG] BUFFER_PATH (relative): {BUFFER_PATH}")
    print(f"[DEBUG] BUFFER_PATH (absolute): {os.path.abspath(BUFFER_PATH)}")

    WORLD_PATH = args.world_path
    words = os.path.join(*WORLD_PATH.split(os.sep)[-3:])

    TRAIN_LIMIT = args.train_limit
    TEST_LIMIT= args.test_limit

    policy_name = args.policy_name
    planner = policy_name.split('_')[0].upper()  # ddp_ftrl -> DDP

    # 创建buffer目录
    if not os.path.exists(BUFFER_PATH + args.policy_name):
        os.makedirs(BUFFER_PATH + args.policy_name, exist_ok=True)

    BUFFER_PATH = BUFFER_PATH + args.policy_name
    print(f"Buffer path: {BUFFER_PATH}")
    print(f"Config file: {os.path.join(BUFFER_PATH, 'config.yaml')}")
    print(f"Config exists: {os.path.exists(os.path.join(BUFFER_PATH, 'config.yaml'))}")

    print(f"FTRL server URL: {args.server_url}")

    print("\nInitializing ROS node...")
    sys.stdout.flush()

    world_id = args.actor_id
    main(world_id, server_url=args.server_url, planner=planner)
