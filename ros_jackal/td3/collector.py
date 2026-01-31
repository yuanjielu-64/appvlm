from os.path import exists, join
import numpy as np
import yaml
import os
import torch
import time
import logging
import re
import pickle
import shutil
import logging
import sys

from sympy.physics.units import length

# Patch for numpy version compatibility (numpy 1.x vs 2.x)
# numpy 2.x uses numpy._core, but older pickle files from numpy 1.x reference it
if not hasattr(np, '_core'):
    sys.modules['numpy._core'] = np.core
    if hasattr(np.core, '_multiarray_umath'):
        sys.modules['numpy._core._multiarray_umath'] = np.core._multiarray_umath


class LocalCollector(object):
    def __init__(self, policy, env, replaybuffer, logging_file):
        self.policy = policy
        self.env = env

        self.buffer = replaybuffer

        self.start_id = 0
        self.end_id = 0

        self.last_obs = None

        self.global_episodes = 0
        self.global_steps = 0

        self.ddp = 1

        self.logging_file = logging_file

    def collect(self, n_steps, status):

        n_steps_curr = 0
        env = self.env
        policy = self.policy
        results = []

        ep_rew = 0
        ep_len = 0

        if self.last_obs is not None:
            obs = self.last_obs
        else:
            obs = env.reset()

        while n_steps_curr < n_steps:
            act = policy.select_action(obs)
            if self.ddp <= 1:
                act = self.env.unwrapped.param_init
            obs_new, rew, done, info = env.step(act)
            obs = obs_new
            ep_rew += rew
            ep_len += 1
            n_steps_curr += 1
            self.global_steps += 1

            world = int(info['world'].split(
                "_")[-1].split(".")[0])
            if self.ddp <= 1:
                a = 10
            else:
                self.buffer.add(obs, act,
                                obs_new, rew,
                                done, world)

            if done:
                obs = env.reset()
                if self.ddp <= 1:
                    type = 'ddp'
                else:
                    type = 'adp'
                    if info['status'] == 'flip' or info['status'] == 'timeout' or info['collision'] >= 1:
                        self.buffer.update(rew)

                    results.append(dict(
                        ep_rew=ep_rew,
                        ep_len=ep_len,
                        ep_time=info['time'],
                        ep_status=info['status'],
                        world=info['world'],
                        collision=info['collision']
                    ))

                with open(join(self.logging_file, "trajectory_results.txt"), 'a') as f:
                    f.write(
                        f"Type: {type}, Collision: {info['collision']}, Recovery: {info['recovery']:.6f}, Smoothness: {info['smoothness']:.6f}, Status: {info['status']}, Time: {info['time']:.3f}, Reward: {ep_rew:.3f}, World: {info['world']}\n")

                ep_rew = 0
                ep_len = 0
                self.ddp += 1
                self.global_episodes += 1
                self.start_id = self.end_id

            print("n_episode: %d, n_steps: %d" %(self.global_episodes, self.global_steps), end="\r")

        self.last_obs = obs

        return n_steps_curr, results

class CondorCollector(object):
    def __init__(self, policy, env, replaybuffer, buffer_path, use_actor_id):
        '''
        it's a fake tianshou Collector object with the same api
        '''
        super().__init__()
        self.policy = policy
        self.num_actor = env.config['condor_config']['num_actor']
        self.ids = list(range(self.num_actor))
        self.ep_count = [0]*self.num_actor
        self.buffer = replaybuffer

        self.use_actor_id = use_actor_id

        self.buffer_path = buffer_path
        self.actor_id = env.config['condor_config']['worlds']

        self.sync_dir = join(buffer_path, 'sync')
        os.makedirs(self.sync_dir, exist_ok=True)

        self.test_sync_dir = join(buffer_path, 'test_sync')
        os.makedirs(self.test_sync_dir, exist_ok=True)

        self.continue_file = join(self.sync_dir, 'continue.signal')

        with open(self.continue_file, 'w') as f:
            f.write('')

        self.status = 'train'

        # save the current policy
        self.update_policy()
        # save the env config the actor should read from
        src_config = env.config["env_config"]["config_path"]
        dst_config = join(self.buffer_path, "config.yaml")

        # Only copy if source and destination are different files
        if os.path.abspath(src_config) != os.path.abspath(dst_config):
            shutil.copyfile(src_config, dst_config)
        else:
            print(f"    >>>> Config already exists at {dst_config}, skipping copy")

    def buffer_expand(self, traj):
        for i in range(len(traj)):
            obs, act, rew, done, info, opt_time, nav_metric = traj[i]
            obs_new = traj[i+1][0] if i < len(traj)-1 else traj[i][0]
            world = int(info['world'].split(
                "_")[-1].split(".")[0])  # task index
            self.buffer.add(obs, act,
                            obs_new, rew,
                            done, world)

    def natural_keys(self, text):
        return int(re.split(r'(\d+)', text)[1])

    def sort_traj_name(self, traj_files):
        ep_idx = np.array([self.natural_keys(fn) for fn in traj_files])
        idx = np.argsort(ep_idx)
        return np.array(traj_files)[idx]

    def update_policy(self):
        self.policy.save(self.buffer_path,  "policy_copy")
        # To prevent failure of actors when reading the saved policy
        shutil.move(
            join(self.buffer_path, "policy_copy_actor"),
            join(self.buffer_path, "policy_actor")
        )
        shutil.move(
            join(self.buffer_path, "policy_copy_noise"),
            join(self.buffer_path, "policy_noise")
        )

    def update_signal(self, status):
        with open(self.continue_file, 'w') as f:
            f.write(status)

    def cleaning(self, status):
        if status == 'train':
            for id in range(0, len(self.actor_id)):
                base = join(self.buffer_path, 'actor_%d' % (id))

                if os.path.exists(base):

                    for filename in os.listdir(base):
                        # Preserve learned directory, learned_history.txt, and trajectory_results.txt
                        if filename in ['trajectory_results.txt', 'learned', 'learned_history.txt']:
                            continue

                        file_path = os.path.join(base, filename)
                        try:
                            if os.path.isfile(file_path):
                                os.remove(file_path)
                            elif os.path.isdir(file_path):
                                import shutil
                                shutil.rmtree(file_path)
                        except OSError as e:
                            print(f"删除失败: {file_path}, 错误: {e}")
        else:
            base = self.test_sync_dir
            if os.path.exists(base):
                for filename in os.listdir(base):
                    file_path = os.path.join(base, filename)
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                        elif os.path.isdir(file_path):
                            import shutil
                            shutil.rmtree(file_path)
                    except OSError as e:
                        print(f"删除失败: {file_path}, 错误: {e}")

    def collect(self, n_steps, status = 'train'):

        if status == 'test':
            return self.collect_actor()
        else:
            return self.collect_n_steps(n_steps)

    def collect_actor(self):
        self.update_policy()
        self.update_signal('test')

        time.sleep(1)

        self.cleaning('train')

        steps = 0
        results = []
        ids = list(range(len(self.actor_id)))
        start_time = time.time()

        while ids:

            if time.time() - start_time > 60:
                print(f"⚠ collect_actor 超时（60秒）")
                print(f"总共 {len(self.actor_id)} 个actor，已完成 {len(self.actor_id) - len(ids)} 个")
                print(f"未完成的actor IDs: {sorted(ids)}")
                break

            try:
                files = os.listdir(self.test_sync_dir)

                for filename in files:
                    if filename.startswith('test_') and filename.endswith('.pickle'):
                        parts = filename.split('_')
                        if len(parts) >= 2:
                            file_id = int(parts[1])
                            if file_id in ids:
                                target = join(self.test_sync_dir, filename)
                                with open(target, 'rb') as f:
                                    try:

                                        if os.path.getsize(target) == 0:
                                            print(f"⚠ 空文件，跳过: {target}")
                                            continue

                                        traj = pickle.load(f)

                                        ep_rew = sum([t[2] for t in traj])
                                        ep_len = len(traj)
                                        status = traj[-1][4]['status']
                                        ep_time = traj[-1][4]['time']
                                        world = traj[-1][4]['world']
                                        collision = traj[-1][4]['collision']
                                        opt_time = traj[-1][5]
                                        nav_metric = traj[-1][6]
                                        results.append(
                                            dict(ep_rew=ep_rew, ep_len=ep_len, ep_status=status, ep_time=ep_time,
                                                 world=world, collision=collision, opt_time=opt_time,
                                                 nav_metric=nav_metric))
                                        steps += ep_len

                                        os.remove(target)
                                        ids.remove(file_id)

                                    except EOFError:
                                        print(f"⚠ 读取失败（EOFError），跳过: {target}")
                                        bad_dir = join(self.buffer_path, "bad")
                                        os.makedirs(bad_dir, exist_ok=True)
                                        shutil.move(target, join(bad_dir, filename))
                                        continue

            except (OSError, ValueError) as e:
                print(f"处理文件时出错: {e}")

            time.sleep(1)

        return steps, results

    def collect_n_steps(self, n_steps):
        """ This method searches the buffer folder and collect all the saved trajectories
        """
        # collect happens after policy is updated

        self.update_policy()
        self.update_signal('train')

        time.sleep(1)

        self.cleaning('test')

        steps = 0
        results = []

        if os.path.exists(self.buffer_path):
            actor_folders = [d for d in os.listdir(self.buffer_path)
                             if os.path.isdir(join(self.buffer_path, d)) and d.startswith('actor_')]
            if not actor_folders:
                print("Wating for actor...")
                while not actor_folders:
                    time.sleep(2)
                    if os.path.exists(self.buffer_path):
                        actor_folders = [d for d in os.listdir(self.buffer_path)
                                         if os.path.isdir(join(self.buffer_path, d)) and d.startswith('actor_')]

        print("The actor is activated, we start collecting experience!")
        print(f"Target: {n_steps} steps, Current: {steps} steps")

        while steps < n_steps:
            time.sleep(1)
            np.random.shuffle(self.ids)
            progress_pct = (steps / n_steps) * 100 if n_steps > 0 else 0
            print(f"Buffer progress: {steps}/{n_steps} steps ({progress_pct:.1f}%) | Buffer size: {self.buffer.size}")
            for id in self.ids:

                base = join(self.buffer_path, 'actor_%d' % (id))

                try:
                    traj_files = os.listdir(base)
                except:
                    traj_files = []

                # Filter out non-pickle files and special files
                traj_files = [f for f in traj_files
                             if f.endswith('.pickle')
                             and f != 'trajectory_results.txt'
                             and not f.startswith('.')]

                # Create learned cache directory
                learned_dir = join(base, 'learned')
                if not os.path.exists(learned_dir):
                    os.makedirs(learned_dir)

                # Load learned history
                learned_history_file = join(base, 'learned_history.txt')
                if os.path.exists(learned_history_file):
                    with open(learned_history_file, 'r') as f:
                        learned_set = set(line.strip() for line in f)
                else:
                    learned_set = set()

                traj_files = self.sort_traj_name(traj_files)[:]
                for p in traj_files:
                    try:
                        target = join(base, p)

                        # Skip if already learned
                        if p in learned_set:
                            print(f"Skipping already learned: {p}")
                            continue

                        if os.path.getsize(target) > 0:
                            if steps < n_steps:  # if reach the target steps, don't put the experinece into the buffer
                                with open(target, 'rb') as f:
                                    traj = pickle.load(f)
                                    ep_rew = sum([t[2] for t in traj])
                                    ep_len = len(traj)
                                    status = traj[-1][4]['status']
                                    ep_time = traj[-1][4]['time']
                                    world = traj[-1][4]['world']
                                    collision = traj[-1][4]['collision']
                                    opt_time = traj[-1][5]
                                    nav_metric = traj[-1][6]
                                    results.append(dict(ep_rew=ep_rew, ep_len=ep_len, ep_status=status, ep_time=ep_time, world=world, collision=collision, opt_time = opt_time, nav_metric = nav_metric))
                                    self.buffer_expand(traj)
                                    steps += ep_len

                                # Move to learned directory instead of deleting
                                learned_path = join(learned_dir, p)
                                shutil.move(target, learned_path)

                                # Record in learned history
                                with open(learned_history_file, 'a') as f:
                                    f.write(f"{p}\n")
                                learned_set.add(p)

                                print(f"Learned and archived: {p} | Progress: {steps}/{n_steps} steps")
                    except:
                        logging.exception('')
                        print("failed to load actor_%s:%s" % (id, p))
                        pass

        print(f"\n{'='*80}")
        print(f"Collection complete! Collected {steps} steps from {len(results)} trajectories")
        print(f"Final buffer size: {self.buffer.size}")
        print(f"{'='*80}\n")

        return steps, results