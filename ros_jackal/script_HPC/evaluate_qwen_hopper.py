import os
import yaml
import pickle
from os.path import join, dirname, abspath, exists
import sys

sys.path.append(dirname(dirname(abspath(__file__))))
import gym
import numpy as np
import random
import time
import rospy
import argparse
import logging
import tf

import os
print(os.environ['PATH'])

from td3.train import initialize_policy
from envs import registration
from envs.wrappers import ShapingRewardWrapper, StackFrame
from script.qwen.qwen_client import QwenClient  # Qwen客户端（唯一的VLM接口）

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


def initialize_actor(id, BUFFER_PATH):
    print(f"\n[DEBUG] ========== initialize_actor ==========")
    print(f"[DEBUG] Actor ID: {id}")
    print(f"[DEBUG] BUFFER_PATH: {BUFFER_PATH}")
    rospy.logwarn(">>>>>>>>>>>>>>>>>> actor id: %s <<<<<<<<<<<<<<<<<<" % (str(id)))
    assert os.path.exists(BUFFER_PATH), BUFFER_PATH
    actor_path = join(BUFFER_PATH, 'actor_%s' % (str(id)))

    if not exists(actor_path):
        print(f"[DEBUG] Creating actor directory: {actor_path}")
        os.mkdir(actor_path)
    else:
        print(f"[DEBUG] Actor directory exists: {actor_path}")

    f = None
    config_path = join(BUFFER_PATH, 'config.yaml')
    print(f"[DEBUG] Looking for config file: {config_path}")
    while f is None:
        try:
            f = open(config_path, 'r')
            print(f"[DEBUG] ✓ Config file opened successfully")
        except Exception as e:
            print(f"[DEBUG] ✗ Config file not found, waiting... ({e})")
            rospy.logwarn("wait for critor to be initialized")
            time.sleep(2)

    config = yaml.load(f, Loader=yaml.FullLoader)
    print(f"[DEBUG] ✓ Config loaded")
    print(f"[DEBUG] env_id: {config.get('env_config', {}).get('env_id', 'N/A')}")
    print(f"[DEBUG] ========================================\n")
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
    return "world_%d.world" %(WORLD_ID)

def _update_reward(traj):
    failure_reward = traj[-1][2]
    failure_steps = min(4, len(traj))

    for i in range(failure_steps):
        step_idx = len(traj) - 1 - i
        penalty_ratio = 0.5 ** i
        adjusted_reward = failure_reward * penalty_ratio
        traj[step_idx][2] = adjusted_reward

    return traj


def main(id, total_worlds=300, runs_per_world=2, use_qwen=False, qwen_url="http://localhost:5000"):
    print(f"\n[DEBUG] ========== main() function started ==========")
    print(f"[DEBUG] id={id}, qwen_url={qwen_url}")

    actor_dir = join(BUFFER_PATH, 'actor_%s' % (str(id)))
    print(f"[DEBUG] Creating actor_dir: {actor_dir}")
    os.makedirs(actor_dir, exist_ok=True)

    print(f"[DEBUG] Initializing FileSync...")
    file_sync = FileSync(id, BUFFER_PATH, actor_dir)

    print(f"[DEBUG] Calling initialize_actor...")
    config = initialize_actor(id, BUFFER_PATH)
    env_config = config['env_config']
    print(f"[DEBUG] ✓ Config initialized")

    env_config["kwargs"]["WORLD_PATH"] = words
    env_config["kwargs"]["img_dir"] = file_sync.actor_dir
    env_config["kwargs"]["pid"] = id
    env_config["kwargs"]["use_vlm"] = True

    # 🔧 强制使用 Qwen (Hopper 版本)
    print(f"\n[DEBUG] ========== Connecting to Qwen ==========")
    print(f">>>>>>>>>> Using Qwen2.5-VL service at {qwen_url} <<<<<<<<<<")
    vlm_client = QwenClient(
        qwen_url=qwen_url,
        algorithm=algorithm,
        timeout=30.0
    )

    vlm_client.img_id = 0

    # 等待 Qwen 服务启动
    print(f"[DEBUG] Waiting for Qwen service (timeout=60s)...")
    try:
        vlm_client.wait_for_service(timeout=60)
        print(f"[DEBUG] ✓ Connected to Qwen service at {qwen_url}")
    except TimeoutError:
        print(f"[DEBUG] ✗ Qwen service timeout!")
        rospy.logerr(f"❌ Qwen service not available at {qwen_url}!")
        rospy.logerr("Please check:")
        rospy.logerr("  1. Is the service running? (squeue -u $USER)")
        rospy.logerr("  2. Is QWEN_HOST correct? (export QWEN_HOST=gpu017)")
        rospy.logerr("  3. Can you reach it? (curl http://gpu017:5000/health)")
        return

    world_name = get_world_name()
    print(f"[DEBUG] World name: {world_name}")

    env_config["kwargs"]["world_name"] = world_name

    init_pos = env_config["kwargs"]["init_position"]
    goal_pos = env_config["kwargs"]["goal_position"]
    print(f"[DEBUG] Init position: {init_pos}, Goal position: {goal_pos}")

    print(f"\n[DEBUG] ========== Creating Gym Environment ==========")
    print(f"[DEBUG] env_id: {env_config['env_id']}")
    print(f"[DEBUG] Creating environment... (this may take a while)")
    env = gym.make(env_config["env_id"], **env_config["kwargs"])
    print(f"[DEBUG] ✓ Environment created")

    if env_config["shaping_reward"]:
        print(f"[DEBUG] Wrapping with ShapingRewardWrapper...")
        env = ShapingRewardWrapper(env)

    print(f"[DEBUG] Wrapping with StackFrame (stack_frame={env_config['stack_frame']})...")
    env = StackFrame(env, stack_frame=env_config["stack_frame"])

    print(f"[DEBUG] Sleeping 5 seconds for environment stabilization...")
    time.sleep(5)
    print(f"[DEBUG] ✓ Environment ready")

    num_trials = 1
    ep = 0

    vlm_client.img_id = 0

    print(f"\n[DEBUG] ========== Starting Episode Loop ==========")
    print(">>>>>>>>>>>>>> Running on %s <<<<<<<<<<<<<<<<" % (world_name))

    while ep < num_trials:
        ep += 1
        print(f"\n[DEBUG] Episode {ep}/{num_trials}")
        print(f"[DEBUG] Resetting environment...")
        state = env.reset()
        print(f"[DEBUG] ✓ Environment reset complete")
        done = False

        while not done:
            linear_vel = state[0][0]
            angular_vel = state[0][1]
            print(str(state[0][0]) + "-- " + str(state[0][1]))

            # 🔧 使用 Qwen 推理
            image_name = f"VLM_{vlm_client.img_id:06d}.png"
            image_path = os.path.join(file_sync.actor_dir, image_name)

            if os.path.exists(image_path):
                result = vlm_client.infer_from_path(image_path, linear_vel, angular_vel)
                vlm_client.img_id += 1

                print(f"Image {vlm_client.img_id}: {image_path}")

                if result and result.get('success'):
                    act = result['parameters_array']
                    print(f"✓ Qwen predicted: {act}")
                else:
                    rospy.logwarn(f"Qwen inference failed, using default params")
                    act = None
            else:
                rospy.logwarn(f"Image not found: {image_path}")
                act = None

            if act is None:
                act = env_config["kwargs"]["param_init"]

            state, rew, done, info = env.step(act)

    env.unwrapped.soft_close()
    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run VLM evaluation on Hopper GPU cluster')
    parser.add_argument('--id', dest='actor_id', type=int, default=0)
    parser.add_argument('--world_idx', type=int, default=0)
    parser.add_argument('--policy_name', dest='policy_name', default="ddp_qwen")
    parser.add_argument('--buffer_path', dest='buffer_path', default="../buffer/")
    parser.add_argument('--world_path', dest='world_path', default="../jackal_helper/worlds/BARN1/")
    parser.add_argument('--total_worlds', type=int, default=300, help='Total number of worlds to run')

    # 🔧 Hopper 配置: 默认使用 Qwen，从环境变量读取 GPU 节点
    parser.add_argument('--qwen_host', type=str, default=os.environ.get('QWEN_HOST', 'gpu017'),
                       help='Qwen service GPU node (e.g., gpu017)')
    parser.add_argument('--qwen_port', type=int, default=5000,
                       help='Qwen service port')

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"[DEBUG] ========== Script Start ==========")
    print(f"{'='*60}")
    print(f"[DEBUG] Arguments received:")
    print(f"  --id: {args.actor_id}")
    print(f"  --world_idx: {args.world_idx}")
    print(f"  --policy_name: {args.policy_name}")
    print(f"  --buffer_path: {args.buffer_path}")
    print(f"  --world_path: {args.world_path}")
    print(f"  --qwen_host: {args.qwen_host}")
    print(f"  --qwen_port: {args.qwen_port}")

    BUFFER_PATH = args.buffer_path
    WORLD_PATH = args.world_path
    WORLD_ID = args.world_idx

    policy_name = args.policy_name
    algorithm = policy_name.split('_')[0].upper()
    print(f"[DEBUG] Extracted algorithm from policy_name: {algorithm}")

    words = os.path.join(*WORLD_PATH.split(os.sep)[-3:])
    print(f"[DEBUG] World path suffix: {words}")

    full_buffer_path = BUFFER_PATH + args.policy_name
    print(f"[DEBUG] Full buffer path: {full_buffer_path}")
    if not os.path.exists(full_buffer_path):
        print(f"[DEBUG] Creating buffer directory: {full_buffer_path}")
        os.makedirs(full_buffer_path, exist_ok=True)
    else:
        print(f"[DEBUG] Buffer directory exists: {full_buffer_path}")

    BUFFER_PATH = full_buffer_path
    id = args.actor_id

    # 🔧 构建 Qwen URL
    qwen_url = f"http://{args.qwen_host}:{args.qwen_port}"
    print(f"\n{'='*60}")
    print(f"Hopper Qwen Evaluation")
    print(f"{'='*60}")
    print(f"Qwen Service: {qwen_url}")
    print(f"Algorithm: {algorithm}")
    print(f"Policy Name: {policy_name}")
    print(f"World: {WORLD_ID}")
    print(f"Actor ID: {id}")
    print(f"Buffer Path: {BUFFER_PATH}")
    print(f"{'='*60}\n")

    print(f"[DEBUG] Calling main() function...")
    main(id, total_worlds=args.total_worlds, use_qwen=True, qwen_url=qwen_url)
