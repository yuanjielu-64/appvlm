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
import tf

from td3.train import initialize_policy
from envs import registration
from envs.wrappers import ShapingRewardWrapper, StackFrame
from gpt_evaluation import ChatgptEvaluator

os.environ["JACKAL_LASER"] = "1"
os.environ["JACKAL_LASER_MODEL"] = "ust10"
os.environ["JACKAL_LASER_OFFSET"] = "-0.065 0 0.01"


def wait_for_tf_ready(timeout=10.0):

    tf_listener = tf.TransformListener()
    start_time = rospy.Time.now()

    rospy.loginfo("Waiting for TF tree...")

    while (rospy.Time.now() - start_time).to_sec() < timeout:
        try:
            tf_listener.waitForTransform(
                'odom', 'base_link',
                rospy.Time(0),
                rospy.Duration(1.0)
            )
            rospy.loginfo("TF ready!")
            return True
        except:
            rospy.sleep(0.5)

    return False

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


def initialize_actor(id, BUFFER_PATH):
    rospy.logwarn(">>>>>>>>>>>>>>>>>> actor id: %s <<<<<<<<<<<<<<<<<<" % (str(id)))
    assert os.path.exists(BUFFER_PATH), BUFFER_PATH
    actor_path = join(BUFFER_PATH, 'actor_%s' % (str(id)))

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


def get_world_name(world_id):
    """根据 world_id 生成 world 名称"""
    return f"world_{world_id}.world"


def _update_reward(traj):
    failure_reward = traj[-1][2]
    failure_steps = min(4, len(traj))

    for i in range(failure_steps):
        step_idx = len(traj) - 1 - i
        penalty_ratio = 0.5 ** i
        adjusted_reward = failure_reward * penalty_ratio
        traj[step_idx][2] = adjusted_reward

    return traj


def main(id, total_worlds=300, runs_per_world=2):
    """
    主函数 - 循环运行 300 个环境

    Args:
        id: actor id
        total_worlds: 要运行的环境总数（默认 300）
        runs_per_world: 每个环境运行的次数（默认 2）
    """
    actor_dir = join(BUFFER_PATH, 'actor_%s' % (str(id)))
    os.makedirs(actor_dir, exist_ok=True)

    file_sync = FileSync(id, BUFFER_PATH, actor_dir)

    config = initialize_actor(id, BUFFER_PATH)
    env_config = config['env_config']

    # 基础配置
    env_config["kwargs"]["WORLD_PATH"] = words
    env_config["kwargs"]["img_dir"] = file_sync.actor_dir
    env_config["kwargs"]["pid"] = id
    env_config["kwargs"]["use_vlm"] = True

    init_pos = env_config["kwargs"]["init_position"]
    goal_pos = env_config["kwargs"]["goal_position"]

    # 初始化 ChatGPT Evaluator
    chatgpt = ChatgptEvaluator(
        img_dir=file_sync.actor_dir,
        alg=algorithm,
        init_params=env_config["kwargs"]["param_init"]
    )

    print(f">>>>>>>>>> Starting to run {total_worlds} worlds <<<<<<<<<<")

    # 初始化环境变量
    env = None

    # ========== 循环运行 300 个环境 ==========
    for world_idx in range(0, total_worlds):
        world_name = get_world_name(world_idx)

        print(f"\n========== World {world_idx}/{total_worlds}: {world_name} ==========")

        # 更新环境配置
        env_config["kwargs"]["world_name"] = world_name

        # 如果不是第一个world，先关闭之前的环境
        if env is not None:
            rospy.logwarn(f"Closing previous world before loading world {world_idx}...")
            env.unwrapped.soft_close()  # 只关闭 Gazebo，保留 roscore
            time.sleep(5)  # 等待Gazebo和move_base完全关闭
            rospy.logwarn(f"Loading new world: {world_name}")

        # 创建新环境
        env = gym.make(env_config["env_id"], **env_config["kwargs"])
        if env_config["shaping_reward"]:
            env = ShapingRewardWrapper(env)
        env = StackFrame(env, stack_frame=env_config["stack_frame"])

        # 等待TF树建立，增加超时时间和重试机制
        tf_ready = False
        for attempt in range(3):
            rospy.loginfo(f"Waiting for TF tree (attempt {attempt + 1}/3)...")
            if wait_for_tf_ready(timeout=15.0):
                tf_ready = True
                break
            rospy.logwarn(f"TF not ready, retrying...")
            time.sleep(2)

        if not tf_ready:
            rospy.logerr(f"World_name {world_name}: TF timeout after 3 attempts, skipping")
            env.close()
            continue

        # 额外等待确保所有ROS节点完全启动
        time.sleep(3)

        # 重置 ChatGPT evaluator 的图片计数
        chatgpt.reset()

        # ========== 每个环境运行 2 遍 ==========
        for run_idx in range(2):
            print(f"\n--- World {world_idx} - Run {run_idx + 1}/2 ---")

            # 运行一个 episode
            state = env.reset()
            traj = []
            done = False

            while not done:
                # 获取速度
                linear_vel = state[0][0]
                angular_vel = state[0][1]

                # VLM 预测参数
                act = chatgpt.evaluate_single(linear_vel, angular_vel)

                # 如果 VLM 失败，使用默认参数
                if act is None:
                    rospy.logwarn(f"VLM failed for world {world_idx} run {run_idx + 1}, using default params")
                    act = [1.0, 1.0, 8, 20, 1.0, 1.5, 0.3]  # 根据你的算法调整

                print(f"World {world_idx} Run {run_idx + 1} - Linear: {linear_vel:.4f}, Angular: {angular_vel:.4f}")

                # 执行动作
                state, rew, done, info = env.step(act)
                info["world"] = world_name
                traj.append([state, act, rew, done, info, 0, 0])

            # ========== 保存结果 ==========
            if traj:
                info_dict = traj[-1][4]
                opt_time, nav_metric = get_score(
                    init_pos, goal_pos,
                    info_dict['status'],
                    info_dict['time'],
                    info_dict['world']
                )

                traj[-1][5] = opt_time
                traj[-1][6] = nav_metric

                # 如果失败，调整奖励
                if not traj[-1][3] or traj[-1][4]['collision'] >= 1:
                    traj = _update_reward(traj)

                # 写入结果（使用 world_idx * 2 + run_idx 作为唯一ID）
                episode_id = world_idx * 2 + run_idx
                file_sync.write_buffer(
                    opt_time, nav_metric, traj,
                    episode_id, id, file_sync.test_sync_dir, 'test'
                )

                print(
                    f"World {world_idx} Run {run_idx + 1} - Status: {info_dict['status']}, Time: {info_dict['time']:.2f}s, Nav Metric: {nav_metric:.4f}")

            # 短暂延迟，确保环境稳定
            time.sleep(1)

    # ========== 所有环境完成 ==========
    print(f"\n========== All {total_worlds} worlds completed! ==========")

    # 写入完成标记
    file_sync._write_actor_status()

    # 清理
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run VLM evaluation on 300 worlds')
    parser.add_argument('--id', dest='actor_id', type=int, default=0)
    parser.add_argument('--policy_name', dest='policy_name', default="dwa_test")
    parser.add_argument('--buffer_path', dest='buffer_path', default="../buffer/")
    parser.add_argument('--world_path', dest='world_path', default="../jackal_helper/worlds/BARN1/")
    parser.add_argument('--total_worlds', type=int, default=300, help='Total number of worlds to run')

    args = parser.parse_args()

    BUFFER_PATH = args.buffer_path
    WORLD_PATH = args.world_path

    policy_name = args.policy_name
    algorithm = policy_name.split('_')[0].upper()

    words = os.path.join(*WORLD_PATH.split(os.sep)[-3:])

    if not os.path.exists(BUFFER_PATH + args.policy_name):
        os.makedirs(BUFFER_PATH + args.policy_name, exist_ok=True)

    BUFFER_PATH = BUFFER_PATH + args.policy_name
    id = args.actor_id

    main(id, total_worlds=args.total_worlds)