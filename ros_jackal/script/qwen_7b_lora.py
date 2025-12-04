import os
import sys
import time
import json
import logging
import argparse
import pickle
from os.path import join, dirname, abspath, exists

import yaml
import gym
import numpy as np
import random
import torch
from PIL import Image
import rospy
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

sys.path.append(dirname(dirname(abspath(__file__))))

from envs import registration  # noqa: F401
from envs.wrappers import ShapingRewardWrapper, StackFrame
from gpt_evaluation import (
    PROMPT_TEMPLATE,
    ALGORITHM_PARAMS,
    generate_output_format,
)

os.environ["JACKAL_LASER"] = "1"
os.environ["JACKAL_LASER_MODEL"] = "ust10"
os.environ["JACKAL_LASER_OFFSET"] = "-0.065 0 0.01"


class QwenVlmEvaluator:
    """Load Qwen2.5-VL + LoRA once, run repeated single-image inference."""

    def __init__(
        self,
        base_model="Qwen/Qwen2.5-VL-7B-Instruct",
        lora_path="",
        device="auto",
        max_new_tokens=500,
        algorithm="DWA",
        init_params=None,
        img_dir=None,
    ):
        self.base_model = base_model
        self.lora_path = lora_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.img_dir = img_dir
        self.img_id = 0
        self.algorithm = algorithm
        self.init_params = init_params

        if algorithm not in ALGORITHM_PARAMS:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        self.param_config = ALGORITHM_PARAMS[algorithm]
        self.output_format = generate_output_format(self.param_config)
        self.param_order = list(self.param_config.keys())

        self.model, self.processor = self._load_model()

    def _load_model(self):
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.base_model,
            torch_dtype=dtype,
            device_map=self.device,
        )
        model = PeftModel.from_pretrained(model, self.lora_path)
        model.eval()
        processor = AutoProcessor.from_pretrained(self.base_model)
        return model, processor

    def _get_current_image_path(self):
        image_name = f"VLM_{self.img_id:06d}.png"
        return os.path.join(self.img_dir, image_name)

    def _build_prompt(self, linear_vel, angular_vel):
        return PROMPT_TEMPLATE.format(
            number=len(self.param_config),
            algorithm=self.algorithm,
            linear_vel=round(linear_vel, 4),
            angular_vel=round(angular_vel, 4),
            output_format=self.output_format,
        )

    def _parse_result_to_array(self, result):
        try:
            cleaned = result.strip()
            if cleaned.startswith("```"):
                import re

                match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
                if match:
                    cleaned = match.group(1)
                else:
                    cleaned = cleaned.replace("```json", "").replace("```", "").strip()

            params = json.loads(cleaned)
            return [params[key] for key in self.param_order if key in params]
        except Exception as e:
            print(f"[ERROR] Parse failed: {e}")
            return self.init_params

    def evaluate_single(self, linear_vel=0.0, angular_vel=0.0):
        try:
            image_path = self._get_current_image_path()
            if not os.path.exists(image_path):
                print(f"[WARNING] Image not found: {image_path}")
                self.img_id += 1
                return None

            image = Image.open(image_path).convert("RGB")
            prompt = self._build_prompt(linear_vel, angular_vel)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            chat_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                images=[image],
                text=chat_text,
                return_tensors="pt",
            ).to(self.model.device)

            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )

            result = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
            self.img_id += 1
            return self._parse_result_to_array(result)
        except Exception as e:
            print(f"[ERROR] VLM evaluation failed: {e}")
            self.img_id += 1
            return self.init_params

    def reset(self):
        self.img_id = 0


def write_episode(traj, ep, id, path, opt_time, nav_metric, csv_path):
    """Persist one episode's trajectory and metrics."""
    if not traj or len(traj) <= 1 or len(traj[-1]) < 5:
        return

    info_dict = traj[-1][4]

    if (info_dict['recovery'] == 1.0 and info_dict['status'] == 'timeout') or (info_dict['time'] >= 70):
        error_dir = os.path.join(BUFFER_PATH, 'actor_error')
        os.makedirs(error_dir, exist_ok=True)

        error_file = os.path.join(error_dir, f'{id}.txt')

        with open(error_file, 'a') as f:
            f.write(
                f"Environment {id} and World_name {info_dict['world']} has KeyError in info_dict, time: {info_dict['time']}, recovery: {info_dict['recovery']}, status: {info_dict['status']}\n")

        return

    total_reward = sum([step[2] for step in traj])

    with open(join(path, "trajectory_results.txt"), 'a') as f:

        f.write(
            f"Eval: Collision: {info_dict['collision']}, Recovery: {info_dict['recovery']:.6f}, Smoothness: {info_dict['smoothness']:.6f}, Status: {info_dict['status']}, Time: {info_dict['time']:.3f} , Reward: {total_reward:.3f}, Opt_time: {opt_time:.3f} , Nav_Metric: {nav_metric:.3f} , World: {info_dict['world']}\n")

    with open(join(path, f'traj_{ep}.pickle'), 'wb') as f:
        try:
            pickle.dump(traj, f)
        except OSError as e:
            logging.exception('Failed to dump the trajectory! %s', e)
            pass

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    csv_exists = os.path.exists(csv_path)
    with open(csv_path, 'a') as f:
        if not csv_exists:
            f.write("Method,Collision,Recovery,Smoothness,Status,Time,World,optimal_time,nav_metric\n")
        f.write(
            f"Qwen2.5-VL-LoRA,{info_dict['collision']},{info_dict['recovery']},"
            f"{info_dict['smoothness']},{info_dict['status']},{info_dict['time']},"
            f"{info_dict['world']},{opt_time},{nav_metric}\n"
        )


def initialize_actor(id, BUFFER_PATH):
    rospy.logwarn(">>>>>>>>>>>>>>>>>> actor id: %s <<<<<<<<<<<<<<<<<<" %(str(id)))
    assert os.path.exists(BUFFER_PATH), BUFFER_PATH
    actor_path = join(BUFFER_PATH, 'actor_%s' %(str(id)))

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


def compute_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def path_coord_to_gazebo_coord(x, y):
    RADIUS = 0.075
    r_shift = -RADIUS - (30 * RADIUS * 2)
    c_shift = RADIUS + 5

    gazebo_x = x * (RADIUS * 2) + r_shift
    gazebo_y = y * (RADIUS * 2) + c_shift

    return (gazebo_x, gazebo_y)


def get_score(INIT_POSITION, GOAL_POSITION, status, time_cost, world):

    success = status == "success"
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
    actual_time = time_cost
    nav_metric = int(success) * optimal_time / np.clip(actual_time, 2 * optimal_time, 8 * optimal_time)

    return optimal_time, nav_metric


def get_world_name(config, id):
    world_name = config["condor_config"]["worlds"][id]
    if isinstance(world_name, int):
        world_name = "world_%d.world" %(world_name)
    return world_name


def _debug_print_robot_status(env, count, rew, actions):
    Y = env.jackal_ros.get_robot_state()[1]
    X = env.jackal_ros.get_robot_state()[0]
    p = env.gazebo_sim.get_model_state().pose.position
    print(actions)
    print('current step: %d, X position: %f(world_frame), %f(odem_frame), Y position: %f(world_frame), %f(odom_frame), rew: %f' %(count, p.x, X, p.y, Y , rew))


def _update_reward(traj):
    failure_reward = traj[-1][2]
    failure_steps = min(4, len(traj))

    for i in range(failure_steps):
        step_idx = len(traj) - 1 - i

        penalty_ratio = 0.5 ** i
        adjusted_reward = failure_reward * penalty_ratio

        traj[step_idx][2] = adjusted_reward

    return traj


def main(id, base_model, lora_path, device, max_new_tokens, num_episodes):

    actor_dir = join(BUFFER_PATH, 'actor_%s' % (str(id)))
    os.makedirs(actor_dir, exist_ok=True)

    config = initialize_actor(id, BUFFER_PATH)
    env_config = config['env_config']
    algorithm = config.get("policy_name", "DWA").split('_')[0].upper()
    world_name = get_world_name(config, id)
    env_config["kwargs"]["world_name"] = world_name
    env_config["kwargs"]["WORLD_PATH"] = words

    env_config["kwargs"]["img_dir"] = actor_dir
    env_config["kwargs"]["pid"] = id
    env_config["kwargs"]["use_vlm"] = True

    init_pos = env_config["kwargs"]["init_position"]
    goal_pos = env_config["kwargs"]["goal_position"]

    param_init = env_config["kwargs"].get("param_init")
    csv_dir = join(BUFFER_PATH, "test_qwen_7b_lora")
    csv_path = join(csv_dir, f"test_results_{id}.csv")

    qwen = QwenVlmEvaluator(
        base_model=base_model,
        lora_path=lora_path,
        device=device,
        max_new_tokens=max_new_tokens,
        algorithm=algorithm,
        init_params=param_init,
        img_dir=actor_dir,
    )

    env = gym.make(env_config["env_id"], **env_config["kwargs"])
    if env_config["shaping_reward"]:
        env = ShapingRewardWrapper(env)
    env = StackFrame(env, stack_frame=env_config["stack_frame"])

    print(">>>>>>>>>>>>>> Running on %s with Qwen2.5-VL LoRA <<<<<<<<<<<<<<<<" %(world_name))

    for ep in range(1, num_episodes + 1):
        state = env.reset()

        traj = []
        done = False

        while not done:
            act = qwen.evaluate_single(state[0][0], state[0][1])
            print(str(state[0][0]) + "-- " + str(state[0][1]))
            state, rew, done, info = env.step(act)
            info["world"] = world_name
            traj.append([state, act, rew, done, info, 0, 0])

        info_dict = traj[-1][4]
        opt_time, nav_metric = get_score(init_pos, goal_pos, info_dict['status'], info_dict['time'], info_dict['world'])

        traj[-1][5] = opt_time
        traj[-1][6] = nav_metric

        if traj[-1][3] == False or traj[-1][4]['collision'] >= 1:
            traj = _update_reward(traj)

        write_episode(traj, ep, id, actor_dir, opt_time, nav_metric, csv_path)

    print(f"Completed {num_episodes} episodes on {world_name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='start an actor with Qwen2.5-VL LoRA')
    parser.add_argument('--id', dest='actor_id', type=int, default=0)
    parser.add_argument('--policy_name', dest='policy_name', default="ddp_test")
    parser.add_argument('--buffer_path', dest='buffer_path', default="../buffer/")
    parser.add_argument('--world_path', dest='world_path', default="../jackal_helper/worlds/BARN1/")
    parser.add_argument('--base_model', type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument('--lora_path', type=str, required=True)
    parser.add_argument('--device', type=str, default="auto")
    parser.add_argument('--max_new_tokens', type=int, default=500)
    parser.add_argument('--num_episodes', type=int, default=300, help='Number of eval episodes before exit.')

    args = parser.parse_args()
    BUFFER_PATH = args.buffer_path
    WORLD_PATH = args.world_path

    policy_name = args.policy_name

    words = os.path.join(*WORLD_PATH.split(os.sep)[-3:])

    if (os.path.exists(BUFFER_PATH + args.policy_name) == False):
        os.makedirs(BUFFER_PATH + args.policy_name, exist_ok=True)

    BUFFER_PATH = BUFFER_PATH + args.policy_name
    id = args.actor_id
    main(id, args.base_model, args.lora_path, args.device, args.max_new_tokens, args.num_episodes)
