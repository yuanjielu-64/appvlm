import argparse

import GPUtil
import yaml
import numpy as np
import gym
from datetime import datetime
from os.path import join, dirname, abspath, exists
import sys
import os
import shutil
import logging
import collections
import time
from pprint import pformat

import torch

from tensorboardX import SummaryWriter

sys.path.append(dirname(dirname(abspath(__file__))))
from envs import registration
from envs.wrappers import ShapingRewardWrapper, StackFrame
from td3.information_envs import InfoEnv
from td3.net import MLP, CNN
from td3.rl import Actor, Critic, TD3, ReplayBuffer
from td3.collector import CondorCollector, LocalCollector

torch.set_num_threads(8)

import psutil

def restart_gazebo():

    print(">>>>>>>> 正在重启Gazebo...")

    gazebo_processes = ['gazebo', 'gzserver', 'gzclient', 'roslaunch']

    for proc_name in gazebo_processes:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['name'] == proc_name or \
                        (proc.info['cmdline'] and any(proc_name in cmd for cmd in proc.info['cmdline'])):
                    print(f"    >>>> 正在杀死进程: {proc.info['name']} (PID: {proc.info['pid']})")
                    proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass

def initialize_config(config_path, save_path):
    # Load the config files
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config["env_config"]["save_path"] = save_path
    config["env_config"]["config_path"] = config_path

    return config

def initialize_logging(config):
    env_config = config["env_config"]
    training_config = config["training_config"]

    # Config logging
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H_%M")
    save_path = join(
        env_config["save_path"],
        env_config["env_id"],
        training_config['algorithm'],
        dt_string
    )
    print("    >>>> Saving to %s" % save_path)
    if not exists(save_path):
        os.makedirs(save_path)

    writer = SummaryWriter(save_path)

    shutil.copyfile(
        env_config["config_path"],
        join(save_path, "config.yaml")
    )

    return save_path, writer

def getting_model(config):
    env_config = config["env_config"]
    training_config = config["training_config"]
    save_path = join(
        env_config["save_path"],
        env_config["env_id"],
        training_config['algorithm'],
    )

    if not exists(save_path):
        return None

    time_folders = []
    for item in os.listdir(save_path):
        item_path = join(save_path, item)
        if os.path.isdir(item_path):
            try:
                datetime.strptime(item, "%Y_%m_%d_%H_%M")
                time_folders.append(item)
            except ValueError:
                continue

    time_folders.sort(reverse=True)

    for i, folder in enumerate(time_folders):
        model_folder_path = join(save_path, folder)

        actor_file = join(model_folder_path, "policy_actor")
        noise_file = join(model_folder_path, "policy_noise")

        if exists(actor_file) and exists(noise_file):
            return model_folder_path
        elif exists(actor_file):
            return model_folder_path

    return None

def initialize_envs(config):
    env_config = config["env_config"]

    if not env_config["use_condor"]:
        env = gym.make(env_config["env_id"], **env_config["kwargs"])
        if env_config["shaping_reward"]:
            env = ShapingRewardWrapper(env)
        env = StackFrame(env, stack_frame=env_config["stack_frame"])
    else:
        # If use condor, we want to avoid initializing env instance from the central learner
        # So here we use a fake env with obs_space and act_space information
        print("    >>>> Using actors on Condor")
        env = InfoEnv(config)

    return env

def seed(config):
    env_config = config["env_config"]

    np.random.seed(env_config['seed'])
    torch.manual_seed(env_config['seed'])

def initialize_policy(config, env):
    training_config = config["training_config"]

    state_dim = env.observation_space.shape
    action_dim = np.prod(env.action_space.shape)
    action_space_low = env.action_space.low
    action_space_high = env.action_space.high
    devices = GPUtil.getAvailable(order='first', limit=1, maxLoad=0.8, maxMemory=0.8, includeNan=False, excludeID=[],
                                  excludeUUID=[])
    device = "cuda:%d" % (devices[0]) if len(devices) > 0 else "cpu"
    print("    >>>> Running on device %s" % (device))

    state_preprocess = CNN(config["env_config"]["stack_frame"]) if training_config["network"] == "cnn" else None

    input_dim = state_preprocess.feature_dim if state_preprocess else np.prod(state_dim)
    actor = Actor(
        state_preprocess=state_preprocess,
        head=MLP(input_dim, training_config['num_layers'], training_config['hidden_layer_size']),
        action_dim=action_dim
    ).to(device)
    actor_optim = torch.optim.Adam(
        actor.parameters(),
        lr=training_config['actor_lr']
    )

    state_preprocess = CNN(config["env_config"]["stack_frame"]) if training_config["network"] == "cnn" else None
    input_dim += np.prod(action_dim)
    critic = Critic(
        state_preprocess,
        head=MLP(input_dim, training_config['num_layers'], training_config['hidden_layer_size'])
    ).to(device)
    critic_optim = torch.optim.Adam(
        critic.parameters(),
        lr=training_config['critic_lr']
    )

    policy = TD3(
        actor, actor_optim,
        critic, critic_optim,
        action_range=[action_space_low, action_space_high],
        device=device,
        **training_config["policy_args"]
    )

    buffer = ReplayBuffer(state_dim, action_dim, training_config['buffer_size'], device=device)

    return policy, buffer

def train(env, policy, buffer, config):
    env_config = config["env_config"]
    training_config = config["training_config"]

    save_path, writer = initialize_logging(config)

    if config["env_config"]["use_pretrain"] is not None:
        load_model_path = getting_model(config)
        if load_model_path != None:
            policy.load(load_model_path, "policy")

    training_args = training_config["training_args"]

    if env_config["use_condor"]:
        collector = CondorCollector(policy, env, buffer, BUFFER_PATH, training_config['use_actor'])
    else:
        collector = LocalCollector(policy, env, buffer, save_path)

    print("    >>>> Pre-collect experience")
    print(f"    >>>> Target buffer size: {training_config['pre_collect']} steps")

    # collector.collect(n_steps=training_config['pre_collect'], status='test')
    collector.collect(n_steps=training_config['pre_collect'], status='train')

    print(f"    >>>> Buffer filled! Current size: {buffer.size} steps")
    print(f"    >>>> Starting policy training...")
    print("=" * 80)

    n_steps = 0
    n_iter = 0
    n_ep = 0
    epinfo_buf = collections.deque(maxlen=BUFFER_SIZE)
    world_ep_buf = collections.defaultdict(lambda: collections.deque(maxlen=30))
    t0 = time.time()

    tensorboard_step = 0

    best_episode_length = float('inf')
    best_episode_nav = float('-inf')

    while n_steps < training_args["max_step"]:
        # Linear decaying exploration noise from "start" -> "end"
        policy.exploration_noise = \
            - (training_config["exploration_noise_start"] - training_config["exploration_noise_end"]) \
            * n_steps / training_args["max_step"] + training_config["exploration_noise_start"]

        steps, epinfo = collector.collect(training_args["collect_per_step"], status = 'train')

        n_steps += steps
        n_iter += 1
        n_ep += len(epinfo)
        epinfo_buf.extend(epinfo)

        for d in epinfo:
            world = d["world"].split("/")[-1]
            world_ep_buf[world].append(d)

        actor_grad_norms = []
        critic_grad_norms = []
        actor_losses = []
        critic_losses = []
        for _ in range(training_args["update_per_step"]):
            actor_grad_norm, critic_grad_norm, actor_loss, critic_loss = policy.train(buffer,
                                                                                      training_args["batch_size"])
            if actor_loss is not None:
                actor_grad_norms.append(actor_grad_norm)
                actor_losses.append(actor_loss)

            critic_grad_norms.append(critic_grad_norm)
            critic_losses.append(critic_loss)

        t1 = time.time()

        test_steps, test_epinfo = collector.collect(n_steps=training_args['collect_per_step'], status='test')

        status_counts = {"success": 0, "flip": 0, "timeout": 0}
        total_episodes = len(epinfo_buf)

        for epinfo in epinfo_buf:
            status = epinfo.get("ep_status", "unknown")
            if status in status_counts:
                status_counts[status] += 1

        success_rate = 100.0 * status_counts["success"] / total_episodes if total_episodes > 0 else 0.0
        flip_rate = 100.0 * status_counts["flip"] / total_episodes if total_episodes > 0 else 0.0
        timeout_rate = 100.0 * status_counts["timeout"] / total_episodes if total_episodes > 0 else 0.0

        nav_metric_score = np.mean([epinfo["nav_metric"] for epinfo in epinfo_buf])

        nav_metrics = [ep['nav_metric'] for ep in test_epinfo]
        avg_nav_metric = sum(nav_metrics) / len(nav_metrics) if nav_metrics else 0

        ep_times = [ep['ep_time'] for ep in test_epinfo]
        avg_ep_time = sum(ep_times) / len(ep_times) if ep_times else 0

        ep_time_steps = [ep['ep_len'] for ep in test_epinfo]
        avg_ep_len = sum(ep_time_steps) / len(ep_time_steps) if ep_time_steps else 0

        log = {
            "Episode_reward": np.mean([epinfo["ep_rew"] for epinfo in epinfo_buf]),
            "Episode_length": np.mean([epinfo["ep_len"] for epinfo in epinfo_buf]),
            "Episode_nav_metric": nav_metric_score,
            "Test_nav_metric": avg_nav_metric,
            "Test_time" : avg_ep_time,
            "Test_length": avg_ep_len,
            "Test_counts": len(nav_metrics),
            "Success_rate": success_rate,
            "Flip_rate": flip_rate,
            "Timeout_rate": timeout_rate,
            "Status_counts": total_episodes,
            "Time": np.mean([epinfo["ep_time"] for epinfo in epinfo_buf]),
            "Collision": np.mean([epinfo["collision"] for epinfo in epinfo_buf]),
            "Actor_grad_norm": np.mean(actor_grad_norms),
            "Critic_grad_norm": np.mean(critic_grad_norms),
            "Actor_loss": np.mean(actor_losses),
            "Critic_loss": np.mean(critic_losses),
            "fps": n_steps / (t1 - t0),
            "n_episode": n_ep,
            "Steps": n_steps,
            "Exploration_noise": policy.exploration_noise
        }

        logging.info(pformat(log))
        
        # Only start performance evaluation and logging when epinfo_buf is full
        if len(epinfo_buf) >= 0:
            # Check for best performance (high episode_reward, low episode_length)
            current_episode_reward = log["Episode_reward"]
            current_episode_length = log["Test_length"]
            current_episode_nav_metric = log["Test_nav_metric"]

            if (current_episode_nav_metric > best_episode_nav):
                policy_name = f"policy_step_{tensorboard_step}"
                policy.save(save_path, policy_name)
                with open(join(save_path, f"best_performance_step_{tensorboard_step}.txt"), 'w') as f:
                    f.write(f"Best Performance at TensorBoard Step {tensorboard_step}:\n")
                    f.write(f"Training Step: {n_steps}\n")
                    f.write(f"Episode Reward: {current_episode_reward:.3f}\n")
                    f.write(f"Episode Length: {current_episode_length:.3f}\n")
                    f.write(f"Success Rate: {success_rate:.3f}%\n")
                best_episode_nav = current_episode_nav_metric
                best_episode_length = current_episode_length

            if n_iter % training_config["log_intervals"] == 0:
                for k in log.keys():
                    writer.add_scalar('train/' + k, log[k], global_step=tensorboard_step)

                for k in world_ep_buf.keys():
                    writer.add_scalar(k + "/Episode_reward", np.mean([epinfo["ep_rew"] for epinfo in world_ep_buf[k]]),
                                      global_step=tensorboard_step)
                    writer.add_scalar(k + "/Episode_length", np.mean([epinfo["ep_len"] for epinfo in world_ep_buf[k]]),
                                      global_step=tensorboard_step)
                    writer.add_scalar(k + "/Episode_nav_metric", np.mean([epinfo["nav_metric"] for epinfo in world_ep_buf[k]]),
                                      global_step=tensorboard_step)
                    writer.add_scalar(k + "/Success_rate", success_rate,
                                      global_step=tensorboard_step)
                    writer.add_scalar(k + "/Flip_rate", flip_rate,
                                      global_step=tensorboard_step)
                    writer.add_scalar(k + "/Timeout_rate", timeout_rate,
                                      global_step=tensorboard_step)
                    writer.add_scalar(k + "/Status_counts", total_episodes,
                                      global_step=tensorboard_step)
                    writer.add_scalar(k + "/Time", np.mean([epinfo["ep_time"] for epinfo in world_ep_buf[k]]),
                                      global_step=tensorboard_step)
                    writer.add_scalar(k + "/Collision", np.mean([epinfo["collision"] for epinfo in world_ep_buf[k]]),
                                      global_step=tensorboard_step)
            
            # Increment tensorboard_step only when buffer is full
            tensorboard_step += steps

    if env_config["use_condor"]:
        shutil.rmtree(BUFFER_PATH, ignore_errors=True)  # a way to force all the actors to stop
    else:
        train_envs.close()

if __name__ == "__main__":
    # restart_gazebo()

    parser = argparse.ArgumentParser(description = 'Start training')
    parser.add_argument('--config_path', dest='config_path', default="../script/applr/configs/")
    parser.add_argument('--config_file', dest='config_file', default="DDP_cluster")
    parser.add_argument('--buffer_path', dest='buffer_path', default="../buffer/")
    parser.add_argument('--logging_path', dest='logging_path', default="../logging/")
    parser.add_argument('--buffer_size', dest='buffer_size', default= 350)
    parser.add_argument('--device', dest='device', default=None)
    parser.add_argument('--policy_name', dest='policy_name', default="ddp_heurstic_new",
                        help='Policy name (e.g., ddp_heurstic_new). Will load checkpoint from buffer_path/policy_name if exists.')

    logging.getLogger().setLevel("INFO")
    args = parser.parse_args()

    BUFFER_PATH = args.buffer_path
    BUFFER_SIZE = args.buffer_size
    SAVE_PATH = args.logging_path

    # Determine policy name
    if args.policy_name:
        POLICY_NAME = args.policy_name
        print(f">>>>>>>> Using policy name: {POLICY_NAME}")
    else:
        # Will be set after loading config
        POLICY_NAME = None

    # Create/check buffer directory for this policy
    if POLICY_NAME:
        BUFFER_PATH_FULL = join(BUFFER_PATH, POLICY_NAME)

        # Check if config.yaml exists in policy directory
        policy_config_path = join(BUFFER_PATH_FULL, "config.yaml")
        if exists(policy_config_path):
            print(f">>>>>>>> Found existing config at {policy_config_path}")
            print(f">>>>>>>> Loading configuration from policy directory")
            CONFIG_PATH = policy_config_path
        else:
            print(f">>>>>>>> No config found in policy directory")
            print(f">>>>>>>> Using default config from {args.config_path + args.config_file + '.yaml'}")
            CONFIG_PATH = args.config_path + args.config_file + ".yaml"
    else:
        CONFIG_PATH = args.config_path + args.config_file + ".yaml"

    print(f">>>>>>>> Loading the configuration from {CONFIG_PATH}")
    config = initialize_config(CONFIG_PATH, SAVE_PATH)
    ACTION_TYPE = config["env_config"]["action_type"]

    # Set POLICY_NAME if not specified
    if not POLICY_NAME:
        POLICY_NAME = ACTION_TYPE
        print(f">>>>>>>> Using default policy name (ACTION_TYPE): {POLICY_NAME}")

    # Create buffer directory for this policy
    BUFFER_PATH_FULL = join(BUFFER_PATH, POLICY_NAME)
    if not exists(BUFFER_PATH_FULL):
        os.makedirs(BUFFER_PATH_FULL)
        print(f">>>>>>>> Created buffer directory: {BUFFER_PATH_FULL}")
    else:
        print(f">>>>>>>> Buffer directory exists: {BUFFER_PATH_FULL}")

    BUFFER_PATH = BUFFER_PATH_FULL

    seed(config)
    print(">>>>>>>> Creating the environments")
    train_envs = initialize_envs(config)
    env = train_envs  # if config["env_config"]["use_condor"] else train_envs.env[0]

    print(">>>>>>>> Initializing the policy")
    policy, buffer = initialize_policy(config, env)

    # Load checkpoint if exists in buffer_path/policy_name
    checkpoint_path = join(BUFFER_PATH, "policy_actor")
    if exists(checkpoint_path):
        print(f">>>>>>>> Found checkpoint at {BUFFER_PATH}")
        print(f">>>>>>>> Loading checkpoint from {checkpoint_path}")
        try:
            policy.load(BUFFER_PATH, "policy")
            print(">>>>>>>> Successfully loaded checkpoint!")
        except Exception as e:
            print(f">>>>>>>> Warning: Failed to load checkpoint: {e}")
            print(">>>>>>>> Starting from scratch...")
    else:
        print(f">>>>>>>> No checkpoint found at {checkpoint_path}")
        print(">>>>>>>> Starting training from scratch...")

    print(">>>>>>>> Start training")
    train(train_envs, policy, buffer, config)