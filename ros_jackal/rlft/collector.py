"""
VLM Collector for FTRL

专为VLM+DPT设计的经验收集器，处理:
1. Costmap图像的存储和加载
2. VLM推理的异步调用
3. Replay buffer的高效管理
"""
from os.path import exists, join
import numpy as np
import os
import torch
import time
import pickle
import shutil
import logging
from PIL import Image
import io

# Numpy兼容性补丁
import sys
if not hasattr(np, '_core'):
    sys.modules['numpy._core'] = np.core
    if hasattr(np.core, '_multiarray_umath'):
        sys.modules['numpy._core._multiarray_umath'] = np.core._multiarray_umath


# DDP 工具函数
from rlft.ddp_utils import is_ddp_enabled as _is_ddp_enabled
from rlft.ddp_utils import is_main_process as _is_main_process
from rlft.ddp_utils import rank0_print


class VLMLocalCollector(object):
    """
    本地VLM收集器 - 专为VLM+DPT设计

    关键特性:
    1. 存储costmap图像（而非激光雷达数据）
    2. 支持图像的压缩存储（节省磁盘空间）
    3. 兼容VLM的输入格式（PIL.Image）

    Args:
        policy: VLM_DPT_Actor + TD3 policy
        env: Gym环境
        replaybuffer: ReplayBuffer实例
        logging_file: 日志目录
        save_images: 是否保存原始图像（调试用）
        image_compression: 图像压缩质量（1-95，95=高质量）
    """
    def __init__(
        self,
        policy,
        env,
        replaybuffer,
        logging_file,
        save_images=False,
        image_compression=85
    ):
        self.policy = policy
        self.env = env
        self.buffer = replaybuffer

        self.save_images = save_images
        self.image_compression = image_compression

        self.start_id = 0
        self.end_id = 0

        self.last_obs = None

        self.global_episodes = 0
        self.global_steps = 0

        self.logging_file = logging_file

        # 创建图像保存目录（如果需要）
        if self.save_images:
            self.image_dir = join(logging_file, "collected_images")
            os.makedirs(self.image_dir, exist_ok=True)

    def _get_costmap_image(self, obs):
        """
        从observation中提取costmap图像

        Args:
            obs: 环境返回的observation

        Returns:
            PIL.Image: Costmap图像
        """
        # 检查obs类型
        if isinstance(obs, dict) and 'costmap' in obs:
            # 如果环境返回dict，包含'costmap'键
            costmap = obs['costmap']
        elif isinstance(obs, np.ndarray):
            # 如果直接是numpy array
            costmap = obs
        else:
            raise ValueError(f"Unsupported observation type: {type(obs)}")

        # 转换为PIL Image
        if isinstance(costmap, np.ndarray):
            # 假设costmap是[H, W, 3] RGB图像
            if costmap.dtype == np.float32 or costmap.dtype == np.float64:
                costmap = (costmap * 255).astype(np.uint8)
            image = Image.fromarray(costmap)
        elif isinstance(costmap, Image.Image):
            image = costmap
        else:
            raise ValueError(f"Unsupported costmap type: {type(costmap)}")

        return image

    def _compress_image(self, image):
        """
        压缩图像为bytes（节省内存）

        Args:
            image: PIL.Image

        Returns:
            bytes: JPEG压缩后的图像数据
        """
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=self.image_compression)
        return buffer.getvalue()

    def _decompress_image(self, image_bytes):
        """
        解压缩图像

        Args:
            image_bytes: bytes

        Returns:
            PIL.Image
        """
        buffer = io.BytesIO(image_bytes)
        return Image.open(buffer)

    def collect(self, n_steps, status='train'):
        """
        收集n_steps的经验

        Args:
            n_steps: 收集的步数
            status: 'train' 或 'test'

        Returns:
            n_steps_curr: 实际收集的步数
            results: Episode统计信息列表
        """
        n_steps_curr = 0
        env = self.env
        policy = self.policy
        results = []

        ep_rew = 0
        ep_len = 0

        # 初始化或恢复obs
        if self.last_obs is not None:
            obs = self.last_obs
        else:
            obs = env.reset()

        while n_steps_curr < n_steps:
            # 1. 提取costmap图像
            try:
                costmap_image = self._get_costmap_image(obs)
            except Exception as e:
                rank0_print(f"Warning: Failed to get costmap image: {e}")
                rank0_print(f"Observation type: {type(obs)}")
                # Fallback: 使用obs本身
                costmap_image = obs

            # 2. 选择动作 (VLM推理)
            # 注意: policy.select_action需要接受PIL.Image输入
            act = policy.select_action(costmap_image)

            # 3. 环境交互
            obs_new, rew, done, info = env.step(act)

            # 4. 存储经验
            # 这里我们存储压缩后的图像（节省内存）
            # ReplayBuffer会存储在内存中
            if status == 'train':
                try:
                    costmap_image_new = self._get_costmap_image(obs_new)

                    # 压缩图像（可选）
                    # 如果不压缩，直接存PIL.Image会占用大量内存
                    # 但压缩/解压会增加计算开销
                    # 这里提供两种选择
                    if self.image_compression < 100:
                        # 方案A: 存压缩后的bytes
                        obs_compressed = self._compress_image(costmap_image)
                        obs_new_compressed = self._compress_image(costmap_image_new)
                        # 注意: buffer.add需要修改以支持bytes
                        # 或者在这里解压回PIL.Image
                        obs_to_store = costmap_image  # 暂时存PIL.Image
                        obs_new_to_store = costmap_image_new
                    else:
                        # 方案B: 直接存PIL.Image
                        obs_to_store = costmap_image
                        obs_new_to_store = costmap_image_new

                    world = int(info['world'].split("_")[-1].split(".")[0])
                    self.buffer.add(
                        obs_to_store,
                        act,
                        obs_new_to_store,
                        rew,
                        done,
                        world
                    )
                except Exception as e:
                    rank0_print(f"Warning: Failed to add to buffer: {e}")

            # 5. 可选: 保存图像到磁盘（调试用）
            if self.save_images:
                image_path = join(
                    self.image_dir,
                    f"step_{self.global_steps:06d}.jpg"
                )
                costmap_image.save(image_path, quality=95)

            # 6. 更新状态
            obs = obs_new
            ep_rew += rew
            ep_len += 1
            n_steps_curr += 1
            self.global_steps += 1

            # 7. Episode结束处理
            if done:
                obs = env.reset()

                # 记录episode统计
                results.append(dict(
                    ep_rew=ep_rew,
                    ep_len=ep_len,
                    ep_time=info.get('time', 0),
                    ep_status=info.get('status', 'unknown'),
                    world=info.get('world', 'unknown'),
                    collision=info.get('collision', 0),
                    nav_metric=info.get('nav_metric', 0)
                ))

                # 写入日志
                with open(join(self.logging_file, "trajectory_results.txt"), 'a') as f:
                    f.write(
                        f"Episode {self.global_episodes}: "
                        f"Status={info.get('status', 'unknown')}, "
                        f"Time={info.get('time', 0):.3f}s, "
                        f"Reward={ep_rew:.3f}, "
                        f"Collision={info.get('collision', 0)}, "
                        f"World={info.get('world', 'unknown')}\n"
                    )

                ep_rew = 0
                ep_len = 0
                self.global_episodes += 1

            # 进度显示
            if n_steps_curr % 100 == 0:
                rank0_print(f"Collected: {n_steps_curr}/{n_steps} steps "
                      f"| Episodes: {self.global_episodes} "
                      f"| Buffer size: {self.buffer.size}",
                      end="\r")

        self.last_obs = obs
        rank0_print()  # 换行

        return n_steps_curr, results

    def load_trajectories_from_buffer(self, buffer_path, max_trajectories=None):
        """
        从buffer目录加载pickle轨迹文件

        Args:
            buffer_path: buffer路径 (e.g., "../buffer/ftrl_vlm")
            max_trajectories: 最多加载多少条轨迹 (None表示全部加载)

        Returns:
            loaded_count: 加载的轨迹数量
        """
        import glob
        import re

        rank0_print(f"Loading trajectories from {buffer_path}...")

        # 查找所有pickle文件
        # 模式: traj_1.pickle, traj_2.pickle (training data)
        traj_pattern = join(buffer_path, "actor_*/traj_*.pickle")
        traj_files = glob.glob(traj_pattern)

        if not traj_files:
            rank0_print(f"Warning: No trajectory files found in {buffer_path}")
            return 0

        # 按文件名排序
        traj_files = self._sort_traj_files(traj_files)

        # 限制加载数量
        if max_trajectories is not None:
            traj_files = traj_files[:max_trajectories]

        rank0_print(f"Found {len(traj_files)} trajectory files")

        loaded_count = 0
        for traj_file in traj_files:
            try:
                # 检查文件是否为空，如果为空则删除
                if os.path.getsize(traj_file) == 0:
                    rank0_print(f"Deleting empty file: {traj_file}")
                    os.remove(traj_file)
                    continue

                # 加载轨迹
                with open(traj_file, 'rb') as f:
                    traj = pickle.load(f)

                # 验证轨迹格式
                if not traj or len(traj) == 0:
                    rank0_print(f"Skipping empty trajectory: {traj_file}")
                    continue

                # 加载到buffer
                self.buffer.load_from_trajectory(traj)
                loaded_count += 1

                if loaded_count % 10 == 0:
                    rank0_print(f"Loaded {loaded_count}/{len(traj_files)} trajectories "
                          f"| Buffer size: {self.buffer.size}",
                          end="\r")

            except Exception as e:
                rank0_print(f"Error loading {traj_file}: {e}")
                continue

        rank0_print(f"\nSuccessfully loaded {loaded_count} trajectories")
        rank0_print(f"Final buffer size: {self.buffer.size}")

        return loaded_count

    def _sort_traj_files(self, traj_files):
        """按文件名中的数字排序"""
        import re

        def natural_key(text):
            # 提取数字部分: traj_123.pickle -> 123
            match = re.search(r'traj_(\d+)\.pickle', text)
            return int(match.group(1)) if match else 0

        return sorted(traj_files, key=natural_key)


class VLMReplayBuffer(object):
    """
    专为VLM设计的Replay Buffer

    关键改进:
    1. 存储PIL.Image而非numpy array
    2. 支持图像的懒加载（节省内存）
    3. 批量采样时自动转换为tensor

    Args:
        state_dim: 状态维度（VLM模式下忽略）
        action_dim: 动作维度
        max_size: buffer最大容量
        device: torch device
        image_size: costmap图像尺寸 (H, W) - 默认 (400, 600)
        num_history_frames: 历史帧数量
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        max_size=int(1e6),
        device="cpu",
        image_size=(400, 600),  # Height x Width
        num_history_frames=2,  # 历史2帧（与训练时一致）
        sample_dtype=torch.float32  # 🔧 新增：采样时的 dtype，配合 DeepSpeed bf16 使用
    ):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.device = device
        self.image_size = image_size
        self.num_history_frames = num_history_frames
        self.sample_dtype = sample_dtype  # 🔧 存储采样 dtype

        # 存储图像（当前帧 costmap）
        # 注意: 直接存PIL.Image会占用大量内存
        # 生产环境建议存路径，使用时lazy load
        self.img = [None] * max_size  # 当前帧图像 (PIL.Image)
        self.action = np.zeros((max_size, action_dim))  # 7个参数
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))
        self.task = np.zeros((max_size, 1))  # world index

        # 存储历史帧图像 (每个元素是长度为4的列表)
        self.history_imgs = [None] * max_size  # List[List[PIL.Image]]

        # 存储当前速度状态 (用于VLM prompt)
        self.linear_vel = np.zeros((max_size, 1))   # 当前线速度
        self.angular_vel = np.zeros((max_size, 1))  # 当前角速度

        self.mean, self.std = 0.0, 1.0  # reward统计

        # DDP: 记录上次同步的 buffer size（用于增量广播）
        self.last_sync_size = 0

    def add(self, img, action, reward, done, task, history_images=None,
            linear_vel=0.0, angular_vel=0.0):
        """
        添加经验

        Args:
            img: PIL.Image (当前帧costmap)
            action: np.array (7/8个参数)
            reward: float
            done: bool
            task: int (world index)
            history_images: List[PIL.Image] (历史4帧，长度为num_history_frames)
            linear_vel: float (当前线速度 m/s)
            angular_vel: float (当前角速度 rad/s)
        """
        self.img[self.ptr] = img  # 当前帧 PIL.Image
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        self.task[self.ptr] = task

        # 存储历史帧
        if history_images is not None:
            self.history_imgs[self.ptr] = history_images  # List[PIL.Image], 长度=4
        else:
            self.history_imgs[self.ptr] = None

        # 存储当前速度
        self.linear_vel[self.ptr] = linear_vel
        self.angular_vel[self.ptr] = angular_vel

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        # 更新reward统计
        if self.ptr == 1000:
            rew = self.reward[:1000]
            self.mean, self.std = rew.mean(), rew.std()
            if np.isclose(self.std, 0, 1e-2):
                self.mean, self.std = 0.0, 1.0

    def sample(self, batch_size):
        """
        采样batch

        Returns (匹配TD3接口):
            state: List[PIL.Image] (当前帧图像)
            action: torch.Tensor (batch_size, action_dim)
            next_state: List[PIL.Image] (下一帧图像，暂时返回None，由n_step_return计算)
            reward: torch.Tensor (batch_size, 1)
            not_done: torch.Tensor (batch_size, 1)
            task: torch.Tensor (batch_size, 1)
            ind: np.array (索引)
        """
        ind = np.random.randint(0, self.size, size=batch_size)

        # 提取数据
        # 🔧 修复：使用 self.sample_dtype 确保与模型 dtype 一致（支持 bf16）
        imgs = [self.img[i] for i in ind]  # List[PIL.Image]
        action = torch.tensor(self.action[ind], dtype=self.sample_dtype, device=self.device)
        reward = torch.tensor(self.reward[ind], dtype=self.sample_dtype, device=self.device)
        not_done = torch.tensor(self.not_done[ind], dtype=self.sample_dtype, device=self.device)
        task = torch.tensor(self.task[ind], dtype=self.sample_dtype, device=self.device)

        # 提取历史帧 (每个元素是List[PIL.Image]，长度=4)
        history_imgs = [self.history_imgs[i] for i in ind]  # List[List[PIL.Image]]

        # 提取当前速度
        linear_vels = torch.tensor(self.linear_vel[ind], dtype=self.sample_dtype, device=self.device)  # [B, 1]
        angular_vels = torch.tensor(self.angular_vel[ind], dtype=self.sample_dtype, device=self.device)  # [B, 1]

        # 构造 VLM 输入: (当前帧, 历史帧, 线速度, 角速度) 的元组
        state = [(imgs[i], history_imgs[i], linear_vels[i].item(), angular_vels[i].item())
                 for i in range(len(imgs))]

        # next_state 暂时返回None，会在n_step_return中重新计算
        next_state = None

        # 返回顺序匹配 TD3.train() 的期望: state, action, next_state, reward, not_done, task, ind
        return state, action, next_state, reward, not_done, task, ind

    def n_step_return(self, n_step, ind, gamma):
        """
        计算N-step return

        Returns (匹配TD3接口):
            next_state: List[Tuple[PIL.Image, List[PIL.Image]]] - VLM输入格式
            reward: torch.Tensor
            not_done: torch.Tensor
            gammas: torch.Tensor
        """
        reward = []
        not_done = []
        gammas = []
        next_state = []

        for i in ind:
            n = 0
            r = 0
            for _ in range(n_step):
                idx = (i + n) % self.size
                r += (self.reward[idx] - self.mean) / self.std * gamma ** n
                if not self.not_done[idx]:
                    break
                n = n + 1

            # 计算next_state索引
            next_idx = (i + n) % self.size

            # 获取next_state的图像、历史帧和速度
            next_img = self.img[next_idx]
            next_history = self.history_imgs[next_idx] if self.history_imgs[next_idx] is not None else []
            next_linear_vel = self.linear_vel[next_idx].item()
            next_angular_vel = self.angular_vel[next_idx].item()

            next_state.append((next_img, next_history, next_linear_vel, next_angular_vel))
            not_done.append(self.not_done[idx])
            reward.append(r)
            gammas.append([gamma ** (n + 1)])

        # next_state 保持为 List (VLM 需要)
        # 🔧 修复：使用 self.sample_dtype 确保与模型 dtype 一致（支持 bf16）
        not_done = torch.tensor(np.array(not_done), dtype=self.sample_dtype, device=self.device)
        reward = torch.tensor(np.array(reward), dtype=self.sample_dtype, device=self.device)
        gammas = torch.tensor(np.array(gammas), dtype=self.sample_dtype, device=self.device)

        return next_state, reward, not_done, gammas

    def load_from_trajectory(self, traj):
        """
        从轨迹数据加载到buffer

        Args:
            traj: list of [obs, act, rew, done, info, opt_time, nav_metric]
                  其中obs可能是costmap图像路径或numpy array
        """
        for i in range(len(traj)):
            obs, act, rew, done, info, opt_time, nav_metric = traj[i]

            # 获取下一个状态
            obs_new = traj[i+1][0] if i < len(traj)-1 else traj[i][0]

            # 提取world index
            world = int(info['world'].split("_")[-1].split(".")[0])

            # 处理observation (可能是图像路径、PIL.Image或numpy array)
            obs_image = self._convert_to_pil_image(obs)
            obs_new_image = self._convert_to_pil_image(obs_new)

            # 添加到buffer
            self.add(obs_image, act, obs_new_image, rew, done, world)

    def _convert_to_pil_image(self, obs):
        """
        将observation转换为PIL.Image

        Args:
            obs: 可能是str (图像路径), PIL.Image, 或 numpy.ndarray

        Returns:
            PIL.Image
        """
        if isinstance(obs, str):
            # 如果是路径，加载图像
            return Image.open(obs).convert('RGB')
        elif isinstance(obs, Image.Image):
            # 已经是PIL.Image
            return obs
        elif isinstance(obs, np.ndarray):
            # numpy array，转换为PIL.Image
            if obs.dtype == np.float32 or obs.dtype == np.float64:
                obs = (obs * 255).astype(np.uint8)
            return Image.fromarray(obs)
        else:
            raise ValueError(f"Unsupported observation type: {type(obs)}")

    def broadcast_from_rank0(self, full=False):
        """
        从 rank0 广播 buffer 数据到所有进程（用于 DDP）

        Args:
            full: 是否广播完整 buffer（True）还是只广播增量（False）

        策略：
        1. 广播 buffer 元数据（size, ptr, mean, std）
        2. 广播 numpy arrays（action, reward, etc.）
        3. 广播 PIL.Image（转为 bytes）
        """
        import torch.distributed as dist
        import io

        is_rank0 = dist.get_rank() == 0

        # 1. 广播元数据
        if is_rank0:
            metadata = [self.size, self.ptr, self.mean, self.std, self.last_sync_size]
        else:
            metadata = [None, None, None, None, None]

        dist.broadcast_object_list(metadata, src=0)
        buffer_size, buffer_ptr, buffer_mean, buffer_std, last_sync_size = metadata

        if not is_rank0:
            self.size = buffer_size
            self.ptr = buffer_ptr
            self.mean = buffer_mean
            self.std = buffer_std

        if buffer_size == 0:
            rank0_print("    >>>> [DDP] Buffer 为空，跳过广播")
            return

        # 确定广播范围
        if full:
            start_idx = 0
            end_idx = buffer_size
            rank0_print(f"    >>>> [DDP] 广播完整 buffer [0:{buffer_size}]")
        else:
            start_idx = last_sync_size
            end_idx = buffer_size
            if start_idx >= end_idx:
                rank0_print(f"    >>>> [DDP] 没有新数据，跳过广播")
                return
            rank0_print(f"    >>>> [DDP] 广播增量 buffer [{start_idx}:{end_idx}] ({end_idx - start_idx} steps)")

        sync_size = end_idx - start_idx

        # 2. 广播 numpy arrays（只广播增量部分）
        for array_name in ['action', 'reward', 'not_done', 'task', 'linear_vel', 'angular_vel']:
            array_data = getattr(self, array_name)[start_idx:end_idx]
            tensor = torch.from_numpy(array_data).cuda()

            dist.broadcast(tensor, src=0)

            if not is_rank0:
                getattr(self, array_name)[start_idx:end_idx] = tensor.cpu().numpy()

        # 3. 广播 PIL.Image（转为 bytes 列表）
        if is_rank0:
            # 主进程：将 PIL.Image 转为 bytes
            img_bytes_list = []
            for i in range(start_idx, end_idx):
                if self.img[i] is not None:
                    buf = io.BytesIO()
                    self.img[i].save(buf, format='PNG')
                    img_bytes_list.append(buf.getvalue())
                else:
                    img_bytes_list.append(None)

            # 广播 history_imgs
            history_bytes_list = []
            for i in range(start_idx, end_idx):
                if self.history_imgs[i] is not None:
                    hist_bytes = []
                    for hist_img in self.history_imgs[i]:
                        if hist_img is not None:
                            buf = io.BytesIO()
                            hist_img.save(buf, format='PNG')
                            hist_bytes.append(buf.getvalue())
                        else:
                            hist_bytes.append(None)
                    history_bytes_list.append(hist_bytes)
                else:
                    history_bytes_list.append(None)
        else:
            img_bytes_list = [None] * sync_size
            history_bytes_list = [None] * sync_size

        # 广播图像 bytes
        dist.broadcast_object_list(img_bytes_list, src=0)
        dist.broadcast_object_list(history_bytes_list, src=0)

        # 非主进程：将 bytes 转回 PIL.Image
        if not is_rank0:
            for i, idx in enumerate(range(start_idx, end_idx)):
                if img_bytes_list[i] is not None:
                    self.img[idx] = Image.open(io.BytesIO(img_bytes_list[i]))
                else:
                    self.img[idx] = None

                if history_bytes_list[i] is not None:
                    hist_imgs = []
                    for hist_bytes in history_bytes_list[i]:
                        if hist_bytes is not None:
                            hist_imgs.append(Image.open(io.BytesIO(hist_bytes)))
                        else:
                            hist_imgs.append(None)
                    self.history_imgs[idx] = hist_imgs
                else:
                    self.history_imgs[idx] = None

        # 更新同步点
        if is_rank0:
            self.last_sync_size = self.size
        else:
            self.last_sync_size = end_idx


class VLMCondorCollector(object):
    """
    VLM Condor收集器 - 专为FTRL离线训练设计

    新架构 (2024):
    1. 预先启动多个FTRL服务器 (server_urls)
    2. 通过 ftrl_collect_singularity.sh 启动容器收集数据
    3. 从buffer/actor_X/读取轨迹进行训练

    关键特性:
    - 支持启动多个Singularity容器并行收集数据
    - 每个容器连接到独立的FTRL服务器
    - 自动管理环境分配和端口隔离

    Args:
        policy: VLM_DPT_Actor + TD3 policy
        env: InfoEnv (只提供space信息)
        replaybuffer: VLMReplayBuffer实例
        buffer_path: buffer路径 (e.g., "../buffer/ftrl_vlm")
        ftrl_config: FTRL配置字典（可选，如果不提供则从env.config读取）
    """
    def __init__(self, policy, env, replaybuffer, buffer_path, ftrl_config=None):
        super().__init__()
        self.policy = policy
        self.buffer = replaybuffer
        self.buffer_path = buffer_path  # 完整路径: buffer/ddp_rlft

        # DDP: 标记是否是第一次 collect（用于判断是否需要完整广播）
        self._first_collect = True

        # 从buffer_path提取policy_name (例如: "buffer/ddp_rlft" -> "ddp_rlft")
        self.policy_name = os.path.basename(buffer_path)

        # 获取基础目录，传给脚本 (例如: "../buffer/ddp_rlft" -> "buffer/")
        # 容器内工作目录是 /jackal_ws/src/ros_jackal，需要移除 ../
        buffer_base = os.path.dirname(buffer_path)
        # 移除 ../ 前缀，让路径相对于 ros_jackal 目录
        if buffer_base.startswith('../'):
            buffer_base = buffer_base[3:]  # "../buffer" -> "buffer"
        self.buffer_base_path = buffer_base + "/" if buffer_base else ""

        # 从参数或config获取FTRL服务器配置
        # 优先使用传递的 ftrl_config，否则从 env.config 读取
        if ftrl_config is None:
            ftrl_config = env.config.get('ftrl_config', {})
            rank0_print("    >>>> VLMCondorCollector: Loading ftrl_config from env.config")
        else:
            rank0_print("    >>>> VLMCondorCollector: Using provided ftrl_config")

        self.server_urls = ftrl_config.get('server_urls', ['http://localhost:6000'])

        # 支持 train_assignments 和 test_assignments 分离
        # 向后兼容: 如果只有 world_assignments，两者都使用它
        if 'train_assignments' in ftrl_config and 'test_assignments' in ftrl_config:
            self.train_assignments = ftrl_config['train_assignments']
            self.test_assignments = ftrl_config['test_assignments']
            rank0_print("    >>>> Using separate train_assignments and test_assignments")
        else:
            # 向后兼容
            world_assignments = ftrl_config.get('world_assignments', ['0,1,2,3,4,5,6,7,8,9'])
            self.train_assignments = world_assignments
            self.test_assignments = world_assignments
            rank0_print("    >>>> Using unified world_assignments for both train and test")

        # 验证配置一致性
        assert len(self.server_urls) == len(self.train_assignments), \
            f"server_urls 数量 ({len(self.server_urls)}) 必须等于 train_assignments 数量 ({len(self.train_assignments)})"
        assert len(self.server_urls) == len(self.test_assignments), \
            f"server_urls 数量 ({len(self.server_urls)}) 必须等于 test_assignments 数量 ({len(self.test_assignments)})"

        self.num_servers = len(self.server_urls)

        # 从 train_assignments 提取所有 world_id 列表（用于 train）
        self.train_world_ids = []
        for assignment in self.train_assignments:
            world_list = [int(w.strip()) for w in assignment.split(',')]
            self.train_world_ids.extend(world_list)

        # 从 test_assignments 提取所有 world_id 列表（用于 test）
        self.test_world_ids = []
        for assignment in self.test_assignments:
            world_list = [int(w.strip()) for w in assignment.split(',')]
            self.test_world_ids.extend(world_list)

        # 读取 world_path 配置（train 和 test 可以使用不同的 world 目录）
        self.train_world_path = ftrl_config.get('train_world_path', 'jackal_helper/worlds/BARN/')
        self.test_world_path = ftrl_config.get('test_world_path', 'jackal_helper/worlds/BARN/')
        rank0_print(f"    >>>> Train world path: {self.train_world_path}")
        rank0_print(f"    >>>> Test world path:  {self.test_world_path}")

        # 存储 tmux session 名称
        self.collection_tmux_sessions = []

        # 当前使用的 assignments（打乱后的顺序，供 _wait_for_containers_ready 使用）
        self.current_assignments = list(self.train_assignments)  # 初始化为原始顺序

        # 同步目录（保留兼容性）
        self.sync_dir = join(buffer_path, 'sync')
        os.makedirs(self.sync_dir, exist_ok=True)

        self.test_sync_dir = join(buffer_path, 'test_sync')
        os.makedirs(self.test_sync_dir, exist_ok=True)

        self.continue_file = join(self.sync_dir, 'continue.signal')

        # 初始化信号文件
        with open(self.continue_file, 'w') as f:
            f.write('train')

        self.status = 'train'

        # 脚本路径（使用相对路径）
        self.collect_script = '../script/ft_qwen/ftrl_collect_singularity.sh'

        # Debug 模式：不清理文件，不打乱 world 顺序
        self.no_cleanup = ftrl_config.get('no_cleanup', False)

        rank0_print("\n" + "=" * 60)
        rank0_print("  VLMCondorCollector 初始化完成")
        rank0_print("=" * 60)
        rank0_print(f"  Policy 名称:    {self.policy_name}")
        rank0_print(f"  Buffer 路径:    {self.buffer_path}")
        rank0_print(f"  Buffer 基础:    {self.buffer_base_path}")
        rank0_print(f"  服务器数量:     {self.num_servers}")
        rank0_print(f"  Train 环境数:   {len(self.train_world_ids)}")
        rank0_print(f"  Test 环境数:    {len(self.test_world_ids)}")
        rank0_print(f"  收集脚本:       {self.collect_script}")
        if self.no_cleanup:
            rank0_print(f"  🔧 DEBUG 模式:   no_cleanup=True (不清理文件，不打乱world)")
        rank0_print("=" * 60 + "\n")

    def buffer_expand(self, traj, actor_dir):
        """
        将轨迹数据加载到buffer

        Args:
            traj: list of [state, act, rew, done, info, opt_time, nav_metric]
                  state: (linear_vel, angular_vel) - 机器人速度，不是图像！
                  info['img_PIL']: 当前帧PIL Image对象
            actor_dir: actor目录路径 (保留参数兼容性，但不再使用)
        """
        for i in range(len(traj)):
            state, act, rew, done, info, opt_time, nav_metric = traj[i]

            full_action = np.concatenate([act, np.ravel(state)])

            world = int(info['world'].split("_")[-1].split(".")[0])

            # 直接从info中获取PIL Image
            img_PIL = info.get('img_PIL')

            if img_PIL is None:
                rank0_print(f"Warning: img_PIL is None in traj[{i}], skipping...")
                continue

            if not isinstance(img_PIL, Image.Image):
                rank0_print(f"Warning: img_PIL is not a PIL Image (type={type(img_PIL).__name__}), skipping...")
                continue

            # 当前帧图像（已经是PIL.Image格式）
            current_img = img_PIL

            # 构建历史4帧 (从新到旧: i-1, i-2, i-3, i-4)
            # 根据轨迹的顺序直接获取，无需通过文件名查找
            # 使用智能fallback: 优先使用最近的可用帧，而不是直接用当前帧
            history_imgs = []
            last_available_img = None  # 记录最近的可用历史帧

            for offset in [1, 2, 3, 4]:  # 从新到旧
                hist_idx = i - offset

                if hist_idx >= 0:
                    # 历史帧存在于轨迹中
                    hist_info = traj[hist_idx][4]  # info在第5个位置
                    hist_img_PIL = hist_info.get('img_PIL')

                    if hist_img_PIL is not None and isinstance(hist_img_PIL, Image.Image):
                        # 该帧有效，使用它
                        history_imgs.append(hist_img_PIL)
                        last_available_img = hist_img_PIL  # 更新最近可用帧
                    else:
                        # 该历史帧没有图像，使用最近的可用帧填充
                        if last_available_img is not None:
                            history_imgs.append(last_available_img)
                        else:
                            # 连之前的帧都不存在，用当前帧填充
                            history_imgs.append(current_img)
                            last_available_img = current_img
                else:
                    # 帧号<0（轨迹开始前），使用最近可用帧填充
                    if last_available_img is not None:
                        history_imgs.append(last_available_img)
                    else:
                        # 如果没有任何历史帧，用当前帧填充
                        history_imgs.append(current_img)
                        last_available_img = current_img

            # 提取当前真实速度 (来自 info['last_state'])
            last_state = info.get('last_state', [0.0, 0.0])  # [linear_vel, angular_vel]
            current_linear_vel = last_state[0] if len(last_state) >= 1 else 0.0
            current_angular_vel = last_state[1] if len(last_state) >= 2 else 0.0

            # 添加到buffer (包含当前图像、历史图像、action、reward、当前速度等)
            # 注意: full_action 包含下一时刻速度，而linear_vel/angular_vel是当前速度
            self.buffer.add(
                img=current_img,
                action=full_action,
                reward=rew,
                done=done,
                task=world,
                history_images=history_imgs,  # 历史4帧
                linear_vel=current_linear_vel,    # 当前线速度
                angular_vel=current_angular_vel   # 当前角速度
            )

    def _cleanup_trajectory_files(self, traj_file, traj, actor_dir):
        """
        删除已加载的 pickle 文件和对应的图像文件

        Args:
            traj_file: pickle 文件路径
            traj: 轨迹数据（用于提取图像文件名）
            actor_dir: actor 目录路径
        """
        # Debug 模式：跳过清理
        if self.no_cleanup:
            return

        import glob

        # 1. 删除 pickle 文件
        try:
            os.remove(traj_file)
        except OSError as e:
            logging.warning(f"Failed to delete pickle: {traj_file}, error: {e}")

        # 2. 删除对应的图像文件 (从 info['img_label'])
        first_img_label = None
        for transition in traj:
            if len(transition) < 5:
                continue
            info = transition[4]
            img_label = info.get('img_label')

            if img_label and isinstance(img_label, str):
                # 记录第一个 img_label
                if first_img_label is None:
                    first_img_label = img_label

                img_path = join(actor_dir, img_label)
                if exists(img_path):
                    try:
                        os.remove(img_path)
                    except OSError:
                        pass

        # 3. 删除第一个 img_label 的上一帧（初始帧）
        # 例如: FTRL_000005.png -> FTRL_000004.png
        if first_img_label:
            import re
            match = re.match(r'(\w+_)(\d+)(\.png)', first_img_label)
            if match:
                prefix, num_str, suffix = match.groups()
                prev_num = int(num_str) - 1
                if prev_num >= 0:
                    prev_img_label = f"{prefix}{prev_num:06d}{suffix}"
                    prev_img_path = join(actor_dir, prev_img_label)
                    if exists(prev_img_path):
                        try:
                            os.remove(prev_img_path)
                        except OSError:
                            pass

    def _convert_to_pil_image(self, obs):
        """将observation转换为PIL.Image"""
        if isinstance(obs, str):
            return Image.open(obs).convert('RGB')
        elif isinstance(obs, Image.Image):
            return obs
        elif isinstance(obs, np.ndarray):
            if obs.dtype == np.float32 or obs.dtype == np.float64:
                obs = (obs * 255).astype(np.uint8)
            return Image.fromarray(obs)
        else:
            raise ValueError(f"Unsupported observation type: {type(obs)}")

    def natural_keys(self, text):
        """提取文件名中的数字用于排序"""
        import re
        match = re.search(r'(\d+)', text)
        return int(match.group(1)) if match else 0

    def sort_traj_name(self, traj_files):
        """按轨迹编号排序"""
        ep_idx = np.array([self.natural_keys(fn) for fn in traj_files])
        idx = np.argsort(ep_idx)
        return np.array(traj_files)[idx]

    def update_policy(self):
        """
        更新policy文件供actor使用（单进程版本，用于非 ZeRO-3）

        将当前policy完整保存到buffer_path/policy/
        包含：LoRA, DPT Head, History Encoder, normalization 等
        这样 qwen_server.py 可以加载最新的policy
        """
        policy_dir = join(self.buffer_path, "policy")
        policy_tmp_dir = join(self.buffer_path, "policy_tmp")

        # 1. 保存到临时目录
        self.policy.save(self.buffer_path, "policy_tmp")

        # 2. 原子性替换：删除旧目录，重命名新目录
        if exists(policy_dir):
            shutil.rmtree(policy_dir)
        shutil.move(policy_tmp_dir, policy_dir)

        rank0_print(f"[Collector] Policy 已保存: {policy_dir}")

        # 4. 通知所有 server 重新加载 checkpoint
        self._reload_servers()

    def _save_policy_all_ranks(self):
        """
        ZeRO-3 兼容的 policy 保存（所有进程参与）

        ZeRO-3 模式下，参数被分片到不同 GPU，需要所有进程参与 gather。
        但只有 rank 0 执行实际的文件写入和目录操作。
        """
        import torch.distributed as dist

        policy_dir = join(self.buffer_path, "policy")
        policy_tmp_dir = join(self.buffer_path, "policy_tmp")

        # 1. 所有进程参与 save（内部会 gather 参数，只有 rank 0 写文件）
        self.policy.save(self.buffer_path, "policy_tmp")

        # 2. barrier 确保 save 完成
        if _is_ddp_enabled() and dist.is_initialized():
            dist.barrier()

        # 3. 只有 rank 0 执行目录操作和 reload
        if _is_main_process():
            if exists(policy_dir):
                shutil.rmtree(policy_dir)
            shutil.move(policy_tmp_dir, policy_dir)
            rank0_print(f"[Collector] Policy 已保存: {policy_dir}")
            self._reload_servers()

        # 4. barrier 确保 reload 完成
        if _is_ddp_enabled() and dist.is_initialized():
            dist.barrier()

    def _reload_servers(self):
        """
        通知所有 VLM server 重新加载 checkpoint
        """
        import requests

        rank0_print(f"[Collector] 通知 {len(self.server_urls)} 个 server 重新加载...")

        success_count = 0
        for i, server_url in enumerate(self.server_urls):
            reload_url = f"{server_url}/reload"
            try:
                response = requests.post(reload_url, timeout=120)  # reload 可能需要较长时间
                if response.status_code == 200:
                    result = response.json()
                    rank0_print(f"    ✓ Server {i} ({server_url}): {result.get('message', 'OK')}")
                    success_count += 1
                else:
                    rank0_print(f"    ✗ Server {i} ({server_url}): HTTP {response.status_code}")
            except requests.exceptions.Timeout:
                rank0_print(f"    ✗ Server {i} ({server_url}): 超时")
            except requests.exceptions.ConnectionError:
                rank0_print(f"    ✗ Server {i} ({server_url}): 连接失败")
            except Exception as e:
                rank0_print(f"    ✗ Server {i} ({server_url}): {e}")

        rank0_print(f"[Collector] Reload 完成: {success_count}/{len(self.server_urls)} 成功")

    def _collect_test_after_save(self, skip_actual_test=False, timeout=900):
        """
        Save 之后的 test 逻辑（只有 rank 0 执行）

        在 _save_policy_all_ranks() 之后调用。
        包含步骤 3-7：清理目录、启动 test containers、收集结果。

        Args:
            skip_actual_test: 是否跳过实际的 test 运行
            timeout: 等待 test 完成的超时时间

        Returns:
            steps: 总步数
            results: Episode统计
        """
        # 3. 清理 test_sync 目录
        rank0_print("  [3/7] 清理 test_sync 目录...")
        self._clean_test_sync_dir()

        # 4. 清理 train actor 目录中残留的 traj 和图像
        rank0_print("  [4/7] 清理 train actor 目录...")
        self._clean_train_actor_dirs()

        if skip_actual_test:
            # 跳过实际的 test 运行（步骤 5-7）
            rank0_print("  [5-7/7] 跳过实际 test 运行 (--skip_test)...")
            rank0_print("         - 不启动 test containers")
            rank0_print("         - 不等待 test 结果")
            rank0_print("         - 不收集 test 数据")

            # 直接重新启动 train containers
            rank0_print("  [恢复] 重新启动 train containers...")
            self.start_collection_containers(mode='train')

            return (0, [])
        else:
            # 正常执行 test（步骤 5-7）
            # 5. 重新启动 containers (test 模式，会自动设置 signal)
            rank0_print("  [5/7] 重新启动 containers (test 模式)...")
            self.start_collection_containers(mode='test')

            # 6. 等待所有 world_assignments 都有 test 结果
            rank0_print("  [6/7] 等待所有 world 完成 test...")
            self._wait_for_all_test_results(timeout=timeout)

            # 7. 收集结果
            rank0_print("  [7/7] 收集 test 结果...")
            steps, results = self._collect_test_results()

            self.start_collection_containers(mode='train')

            # 计算统计（如果有结果）
            if results:
                nav_metrics = [ep.get('nav_metric', 0) for ep in results if 'nav_metric' in ep]
                if nav_metrics:
                    mean_metric = np.mean(nav_metrics)
                    std_metric = np.std(nav_metrics)
                    rank0_print(f"  📊 Test nav_metric: {mean_metric:.4f} ± {std_metric:.4f}")

            return (steps, results)

    def update_signal(self, status):
        """更新continue.signal文件"""
        with open(self.continue_file, 'w') as f:
            f.write(status)

    def cleaning(self, status):
        """
        清理已使用的轨迹文件（可选）

        在生产环境中可以保留，用于debugging
        """
        if status == 'train':
            for actor_id in range(self.num_servers):
                base = join(self.buffer_path, 'actor_%d' % actor_id)

                if os.path.exists(base):
                    for filename in os.listdir(base):
                        # 保留重要文件
                        if filename in ['trajectory_results.txt', 'learned', 'learned_history.txt']:
                            continue

                        file_path = os.path.join(base, filename)
                        try:
                            if os.path.isfile(file_path) and filename.endswith('.pickle'):
                                pass  # 暂时不删除pickle文件
                                # os.remove(file_path)
                        except OSError as e:
                            rank0_print(f"清理失败: {file_path}, 错误: {e}")
        else:
            base = self.test_sync_dir
            if os.path.exists(base):
                for filename in os.listdir(base):
                    if filename.endswith('.pickle'):
                        file_path = os.path.join(base, filename)
                        try:
                            pass  # 暂时不删除
                            # os.remove(file_path)
                        except OSError as e:
                            rank0_print(f"清理失败: {file_path}, 错误: {e}")

    def collect(self, n_steps, status='train', skip_actual_test=False):
        """
        收集经验（Condor模式）

        DDP 模式（train）：
        1. 主进程收集数据到 buffer
        2. 主进程将 buffer 保存到磁盘
        3. 非主进程从磁盘加载 buffer
        4. 所有进程独立采样不同的 batch

        DDP 模式（test）：
        - 只有主进程执行，非主进程等待

        Args:
            n_steps: 目标步数（用于兼容，实际加载所有可用轨迹）
            status: 'train' 或 'test'
            skip_actual_test: 是否跳过实际的 test 运行（仅对 status='test' 有效）

        Returns:
            steps: 实际加载的步数
            results: Episode统计列表
        """
        if status == 'test':
            # no_cleanup 模式：只保存 policy，跳过容器操作
            if self.no_cleanup:
                rank0_print(f"\n{'='*60}")
                rank0_print(f"  开始 Test 阶段 (no_cleanup 模式)")
                rank0_print(f"{'='*60}")
                rank0_print("  [no_cleanup] 保存 policy，跳过容器操作...")
                self._save_policy_all_ranks()
                rank0_print(f"{'='*60}\n")
                return 0, []

            # test 模式：ZeRO-3 需要所有进程参与 save() 操作
            # 但只有主进程执行其他 test 逻辑（停止容器、reload servers 等）

            # === 步骤 1: 所有进程同步，准备 save ===
            if _is_ddp_enabled():
                import torch.distributed as dist
                if dist.is_initialized():
                    dist.barrier()

            # === 步骤 2: 所有进程参与 save（ZeRO-3 gather 需要） ===
            rank0_print(f"\n{'='*60}")
            rank0_print(f"  开始 Test 阶段")
            rank0_print(f"{'='*60}")
            rank0_print("  [1/7] 停止 train containers...")
            if _is_main_process():
                self.stop_collection_containers()

            rank0_print("  [2/7] 保存最新 policy（所有进程参与 gather）...")
            # 所有进程都调用 save，但只有 rank 0 写文件
            self._save_policy_all_ranks()

            # === 步骤 3: 只有主进程执行其他 test 逻辑 ===
            if _is_main_process():
                result = self._collect_test_after_save(skip_actual_test=skip_actual_test)
            else:
                result = (0, [])

            # === 步骤 4: 同步所有进程 ===
            if _is_ddp_enabled():
                import torch.distributed as dist
                if dist.is_initialized():
                    dist.barrier()

            return result
        else:
            # train 模式：主进程收集，广播 buffer 到其他进程
            if _is_main_process():
                # 主进程：收集数据到 buffer
                result = self.collect_train(n_steps)
            else:
                # 非主进程：等待主进程完成收集
                result = (0, [])

            # DDP: 广播 buffer 数据到所有进程
            if _is_ddp_enabled():
                import torch.distributed as dist
                if dist.is_initialized():
                    # 第一次 collect（pre-collect）：广播完整 buffer
                    # 之后的 collect：只广播增量
                    if self._first_collect:
                        rank0_print(f"    >>>> [DDP] 首次 collect，广播完整 buffer...")
                        self.buffer.broadcast_from_rank0(full=True)
                        self._first_collect = False
                    else:
                        rank0_print(f"    >>>> [DDP] 广播增量 buffer...")
                        self.buffer.broadcast_from_rank0(full=False)

                    rank0_print(f"    >>>> [DDP] Buffer 广播完成，总 size={self.buffer.size}")

            return result

    def start_collection_containers(self, mode='train'):
        """
        启动数据收集容器 (使用 tmux 管理)

        为每个FTRL服务器启动一个 tmux session
        每个 session 运行 ftrl_collect_singularity.sh

        Args:
            mode: 'train' (无限循环) 或 'test' (只运行一轮)
        """
        # DDP 模式：只在主进程启动容器
        if _is_ddp_enabled() and not _is_main_process():
            return  # 非主进程直接返回

        import subprocess
        import requests

        rank0_print(f"\n{'='*60}")
        rank0_print(f"  启动数据收集容器 (tmux) - {mode} 模式")
        rank0_print(f"{'='*60}")
        rank0_print(f"首先检查所有 FTRL 服务器连接...")

        # 检查所有服务器是否可访问
        available_servers = []
        for server_idx in range(self.num_servers):
            server_url = self.server_urls[server_idx]

            try:
                health_url = f"{server_url}/health"
                response = requests.get(health_url, timeout=5)

                if response.status_code == 200:
                    rank0_print(f"  ✓ 服务器 {server_idx}: {server_url} - 连接成功")
                    available_servers.append(server_idx)
                else:
                    rank0_print(f"  ✗ 服务器 {server_idx}: {server_url} - HTTP {response.status_code}")

            except requests.exceptions.Timeout:
                rank0_print(f"  ✗ 服务器 {server_idx}: {server_url} - 超时")
            except requests.exceptions.ConnectionError:
                rank0_print(f"  ✗ 服务器 {server_idx}: {server_url} - 连接失败")
            except Exception as e:
                rank0_print(f"  ✗ 服务器 {server_idx}: {server_url} - 错误: {e}")

        if len(available_servers) == 0:
            raise RuntimeError(
                f"❌ 错误: 没有可用的 FTRL 服务器！\n"
                f"请确保在 {self.server_urls[0].split(':')[1]}:{self.server_urls[0].split(':')[2]} 上启动了服务器。\n"
                f"启动命令: python script_HPC/qwen_server.py --port 6000 --checkpoint /path/to/checkpoint"
            )

        rank0_print(f"\n找到 {len(available_servers)}/{self.num_servers} 个可用服务器")

        # 清理之前的 tmux sessions
        self.stop_collection_containers()

        # 设置 signal (train/test)
        rank0_print(f"  设置 signal 为 '{mode}'...")
        self.update_signal(mode)

        # 启动 tmux sessions
        self.collection_tmux_sessions = []

        # 根据 mode 选择使用哪个 assignments 和 world_path
        assignments = self.train_assignments if mode == 'train' else self.test_assignments
        world_path = self.train_world_path if mode == 'train' else self.test_world_path

        # Train 模式：随机打乱每个 server 的 world 顺序
        # Test 模式或 no_cleanup 模式：保持原顺序（便于追踪和复现）
        if mode == 'train' and not self.no_cleanup:
            import random
            shuffled_assignments = []
            rank0_print(f"\n  [World 随机化] Train 模式 - 打乱 world 顺序:")
            for server_idx, assignment in enumerate(assignments):
                world_list = [int(w.strip()) for w in assignment.split(',')]
                random.shuffle(world_list)
                shuffled_str = ','.join(str(w) for w in world_list)
                shuffled_assignments.append(shuffled_str)
                # 记录打乱后的顺序（便于检测和调试）
                display_str = shuffled_str if len(shuffled_str) <= 50 else shuffled_str[:50] + "..."
                rank0_print(f"    Server {server_idx}: {display_str}")
            assignments = shuffled_assignments
            # 保存打乱后的顺序，供 _wait_for_containers_ready 使用
            self.current_assignments = shuffled_assignments
        elif mode == 'train' and self.no_cleanup:
            rank0_print(f"\n  [World 顺序] Train 模式 + no_cleanup - 保持原顺序 (DEBUG)")
            self.current_assignments = list(assignments)
        else:
            rank0_print(f"\n  [World 顺序] Test 模式 - 保持原顺序")
            self.current_assignments = list(assignments)  # test 模式保持原顺序

        for server_idx in available_servers:
            server_url = self.server_urls[server_idx]
            world_list = assignments[server_idx]
            server_port = server_url.split(':')[-1]

            # tmux session 名称
            session_name = f"collect_{self.policy_name}_{server_idx}"

            # 构建命令（运行完自动退出，不等待输入）
            cmd = f"""bash {self.collect_script} \
    --server_url {server_url} \
    --world_lists {world_list} \
    --policy_name {self.policy_name} \
    --buffer_path {self.buffer_base_path} \
    --mode {mode} \
    --world_path {world_path} \
    --train_limit 5 \
    --test_limit 1"""

            # 创建 tmux session（静默启动）
            try:
                subprocess.run(
                    ["tmux", "new-session", "-d", "-s", session_name, "bash", "-c", cmd],
                    check=True
                )
                self.collection_tmux_sessions.append(session_name)
            except subprocess.CalledProcessError as e:
                rank0_print(f"    ✗ 容器 #{server_idx} 启动失败: {e}")

        rank0_print(f"  ✓ 已启动 {len(self.collection_tmux_sessions)} 个收集容器 (tmux ls | grep collect_)")

        if len(available_servers) < self.num_servers:
            unavailable_count = self.num_servers - len(available_servers)
            rank0_print(f"⚠️  警告: {unavailable_count} 个服务器不可用")
            rank0_print(f"    不可用的服务器索引: {[i for i in range(self.num_servers) if i not in available_servers]}")

        # 等待容器开始收集数据（仅 train 模式）
        # train 模式: 60% 容器就绪即可开始（允许部分容器启动慢）
        # test 模式: 在 _wait_for_all_test_results() 中等待 100% 完成
        if mode == 'train':
            self._wait_for_containers_ready(available_servers, timeout=300, ready_threshold=0.6)

    def _wait_for_containers_ready(self, server_indices, timeout=600, ready_threshold=1.0):
        """
        等待所有容器开始收集数据

        检测方式: 每个 server 对应的第一个 world 的 actor 目录中是否有新的 pickle 文件
        当 ready_threshold (默认100%) 以上的容器就绪时，进入下一步

        Args:
            server_indices: 已启动的服务器索引列表
            timeout: 超时时间（秒）
            ready_threshold: 就绪阈值 (0.0-1.0)，默认 1.0 表示 100%
        """
        import glob

        rank0_print(f"\n{'='*60}")
        rank0_print(f"  等待容器开始收集数据 (检测新 pickle 文件)...")
        rank0_print(f"  就绪阈值: {ready_threshold*100:.0f}%")
        rank0_print(f"{'='*60}")

        # 记录启动时间，用于判断是否为新文件
        launch_time = time.time()

        # 获取每个 server 的第一个 world_id (使用打乱后的 current_assignments)
        first_worlds = []
        for server_idx in server_indices:
            # 使用打乱后的顺序（由 start_collection_containers 设置）
            world_list = self.current_assignments[server_idx]
            first_world = int(world_list.split(',')[0].strip())
            first_worlds.append((server_idx, first_world))

        total_servers = len(first_worlds)
        required_ready = int(total_servers * ready_threshold)

        rank0_print(f"  监控的 actor 目录:")
        for server_idx, world_id in first_worlds:
            rank0_print(f"    Server {server_idx} -> actor_{world_id}")
        rank0_print(f"\n  需要 {required_ready}/{total_servers} 个容器就绪")
        rank0_print()

        ready_servers = set()

        while len(ready_servers) < required_ready:
            # 检查超时
            elapsed = time.time() - launch_time
            if elapsed > timeout:
                rank0_print(f"\n  ⚠️ 超时 ({timeout}s)!")
                rank0_print(f"  已就绪: {len(ready_servers)}/{total_servers}")
                not_ready = [s for s, w in first_worlds if s not in ready_servers]
                rank0_print(f"  未就绪的服务器: {not_ready[:10]}{'...' if len(not_ready) > 10 else ''}")
                break

            # 检查每个未就绪的 server
            for server_idx, world_id in first_worlds:
                if server_idx in ready_servers:
                    continue

                actor_dir = join(self.buffer_path, f'actor_{world_id}')
                pickle_pattern = join(actor_dir, 'traj_*.pickle')
                pickle_files = glob.glob(pickle_pattern)

                # 检查是否有新的 pickle 文件（修改时间 > 启动时间）
                for pkl_file in pickle_files:
                    try:
                        if os.path.getmtime(pkl_file) > launch_time:
                            ready_servers.add(server_idx)
                            rank0_print(f"  ✓ Server {server_idx} (actor_{world_id}) 已开始收集")
                            break
                    except OSError:
                        continue

            # 显示进度
            rank0_print(f"  等待中... {len(ready_servers)}/{total_servers} 就绪 "
                  f"(需要 {required_ready}), 已等待 {int(elapsed)}s", end="\r")
            time.sleep(5)

        rank0_print(f"\n")
        rank0_print(f"{'='*60}")
        ready_pct = len(ready_servers) / total_servers * 100
        rank0_print(f"  ✓ {len(ready_servers)}/{total_servers} ({ready_pct:.0f}%) 个容器已开始收集")
        rank0_print(f"  总等待时间: {int(time.time() - launch_time)}s")
        rank0_print(f"{'='*60}\n")

    def stop_collection_containers(self):
        """停止所有数据收集容器 (tmux sessions)"""
        # DDP 模式：只在主进程停止容器
        if _is_ddp_enabled() and not _is_main_process():
            return  # 非主进程直接返回

        import subprocess

        # 查找所有匹配的 tmux sessions
        try:
            result = subprocess.run(
                ["tmux", "ls"],
                capture_output=True,
                text=True
            )
            existing_sessions = result.stdout
        except Exception:
            existing_sessions = ""

        # 关闭已知的 sessions
        sessions_to_kill = []

        # 从 self.collection_tmux_sessions 获取
        if hasattr(self, 'collection_tmux_sessions'):
            sessions_to_kill.extend(self.collection_tmux_sessions)

        # 同时查找匹配 collect_{policy_name}_ 模式的 sessions
        for line in existing_sessions.split('\n'):
            if f"collect_{self.policy_name}_" in line:
                session_name = line.split(':')[0]
                if session_name not in sessions_to_kill:
                    sessions_to_kill.append(session_name)

        if not sessions_to_kill:
            return

        rank0_print(f"\n{'='*60}")
        rank0_print(f"  关闭数据收集容器 (tmux)")
        rank0_print(f"{'='*60}")

        closed_count = 0
        for session_name in sessions_to_kill:
            try:
                subprocess.run(
                    ["tmux", "kill-session", "-t", session_name],
                    check=True,
                    capture_output=True
                )
                rank0_print(f"  [OK] Closed: {session_name}")
                closed_count += 1
            except subprocess.CalledProcessError:
                pass  # session 可能已经不存在

        if hasattr(self, 'collection_tmux_sessions'):
            self.collection_tmux_sessions = []

        rank0_print(f"\n  总共关闭了 {closed_count} 个 tmux session")
        rank0_print(f"{'='*60}\n")

    def wait_for_collection(self, timeout=None):
        """
        等待所有收集容器完成

        Args:
            timeout: 超时时间（秒），None表示无限等待

        Returns:
            bool: 是否所有容器都成功完成
        """
        import time as time_module

        import subprocess

        if not self.collection_tmux_sessions:
            rank0_print("⚠️ 没有正在运行的收集容器")
            return True

        rank0_print(f"\n等待 {len(self.collection_tmux_sessions)} 个收集容器完成...")
        rank0_print("（按 Ctrl+C 可以提前终止）\n")

        start_time = time_module.time()

        try:
            while True:
                # 检查所有 tmux session 状态
                running = []
                completed = []

                for idx, session_name in enumerate(self.collection_tmux_sessions):
                    result = subprocess.run(
                        ["tmux", "has-session", "-t", session_name],
                        capture_output=True
                    )
                    if result.returncode == 0:
                        running.append(idx)
                    else:
                        completed.append(idx)

                # 显示进度
                rank0_print(f"进度: {len(completed)}/{len(self.collection_tmux_sessions)} 容器完成", end='\r')

                # 所有 session 完成
                if not running:
                    rank0_print(f"\n✓ 所有收集容器已完成")
                    break

                # 检查超时
                if timeout and (time_module.time() - start_time) > timeout:
                    rank0_print(f"\n⚠️ 超时 ({timeout}s)，停止所有容器")
                    self.stop_collection_containers()
                    return False

                time_module.sleep(10)

        except KeyboardInterrupt:
            rank0_print(f"\n\n⚠️ 用户中断，停止所有容器")
            self.stop_collection_containers()
            return False

        return True

    def collect_train(self, n_steps):
        """
        从actor目录加载轨迹直到达到n_steps

        新流程:
        1. 启动收集容器（如果还没启动）
        2. **循环等待**新的trajectory文件生成
        3. 加载新文件到buffer
        4. 达到n_steps后退出
        """
        import glob
        import time

        # 检查 train containers 是否真正在运行（不只是检查列表）
        # 因为 test containers 可能已退出但列表未更新
        import subprocess

        running_sessions = []
        for session_name in self.collection_tmux_sessions:
            try:
                result = subprocess.run(
                    ["tmux", "has-session", "-t", session_name],
                    capture_output=True,
                    timeout=5
                )
                if result.returncode == 0:
                    running_sessions.append(session_name)
            except:
                pass

        if not running_sessions and not self.no_cleanup:
            rank0_print("⚠️ 警告: 数据收集容器未运行，正在启动 train containers...")
            self.start_collection_containers(mode='train')
        elif not running_sessions and self.no_cleanup:
            rank0_print("ℹ️  no_cleanup 模式: 跳过启动容器，从已有数据加载")

        # 清理 test 残留文件
        self._clean_test_actor_dirs()
        self._clean_test_sync_dir()

        rank0_print(f"\n从 buffer 加载轨迹数据（目标: {n_steps} steps）...")

        steps = 0
        results = []
        processed_files = set()  # 记录已处理的文件，避免重复加载

        # **循环等待**直到收集到足够的步数
        while steps < n_steps:
            new_data_found = False

            # 遍历所有 train actor 目录 (使用实际的actor_id)
            for actor_id in self.train_world_ids:
                actor_dir = join(self.buffer_path, f'actor_{actor_id}')

                if not exists(actor_dir):
                    continue

                # 查找该actor的所有轨迹文件
                traj_pattern = join(actor_dir, 'traj_*.pickle')
                traj_files = glob.glob(traj_pattern)

                if not traj_files:
                    continue

                # 排序
                traj_files = self.sort_traj_name(traj_files)

                # 加载**新的**轨迹文件（跳过已处理的）
                for traj_file in traj_files:
                    # 跳过已处理的文件
                    if traj_file in processed_files:
                        continue
                    try:
                        # 检查文件大小，如果为空则删除（除非 no_cleanup）
                        if os.path.getsize(traj_file) == 0:
                            if not self.no_cleanup:
                                rank0_print(f"Deleting empty file: {traj_file}")
                                os.remove(traj_file)
                            processed_files.add(traj_file)
                            continue

                        # 加载pickle
                        with open(traj_file, 'rb') as f:
                            traj = pickle.load(f)

                        if not traj or len(traj) == 0:
                            processed_files.add(traj_file)
                            continue

                        self.buffer_expand(traj, actor_dir)
                        processed_files.add(traj_file)
                        new_data_found = True

                        steps += len(traj)

                        # 删除 pickle 文件和对应的图像
                        self._cleanup_trajectory_files(traj_file, traj, actor_dir)

                        # 提取episode信息
                        if len(traj) > 0 and len(traj[-1]) >= 5:
                            info = traj[-1][4]  # 最后一个transition的info
                            results.append(dict(
                                ep_rew=sum(t[2] for t in traj),  # 累积reward
                                ep_len=len(traj),
                                ep_time=info.get('time', 0),
                                ep_status=info.get('status', 'unknown'),
                                world=info.get('world', 'unknown'),
                                collision=info.get('collision', 0),
                                nav_metric=traj[-1][6]
                            ))

                        if len(results) % 5 == 0:
                            rank0_print(f"Loaded {len(results)} trajectories | "
                                  f"{steps} steps | "
                                  f"Buffer size: {self.buffer.size}",
                                  end="\r")

                        if steps >= n_steps:
                            break

                    except Exception as e:
                        rank0_print(f"\nError loading {traj_file}: {e}")
                        processed_files.add(traj_file)
                        continue

                if steps >= n_steps:
                    break

            if not new_data_found and steps < n_steps:
                rank0_print(f"\n等待新数据... (当前: {steps}/{n_steps} steps)", end="\r")
                time.sleep(5)

        rank0_print(f"\n✓ Collected {steps} steps from {len(results)} episodes")

        # 注意：DDP barrier 在 collect() 函数末尾统一处理，这里不需要

        return steps, results

    def collect_test(self, timeout=900, skip_actual_test=False):
        """
        收集测试数据

        流程:
        1. 保存最新 policy
        2. 停止 train containers
        3. 清理 test_sync 目录
        4. 清理 train actor 目录
        5. 重新启动 containers (test 模式) ← skip_actual_test=True 时跳过
        6. 等待所有 world 完成 test ← skip_actual_test=True 时跳过
        7. 收集结果 ← skip_actual_test=True 时跳过

        注意:
        - Test containers 运行完会自动退出
        - Train containers 会在下次 collect_train() 时自动检测并启动

        Args:
            timeout: 等待 test 完成的超时时间（秒，默认 15 分钟）
            skip_actual_test: 是否跳过实际的 test 运行（步骤 5-7），只执行保存和清理（步骤 1-4）

        Returns:
            steps: 总步数
            results: Episode统计
        """
        import glob

        rank0_print(f"\n{'='*60}")
        rank0_print(f"  开始 Test 阶段")
        rank0_print(f"{'='*60}")

        # no_cleanup 模式：只保存 policy，跳过 container 操作
        if self.no_cleanup:
            rank0_print("  [no_cleanup] 保存 policy，跳过 container 操作...")
            self.update_policy()
            rank0_print(f"{'='*60}\n")
            return 0, []

        # 1. 停止 train containers（必须先停，否则 reload 时 server 会被请求阻塞）
        rank0_print("  [1/7] 停止 train containers...")
        self.stop_collection_containers()

        # 2. 保存最新 policy 并 reload servers
        rank0_print("  [2/7] 保存最新 policy 并 reload servers...")
        self.update_policy()

        # 3. 清理 test_sync 目录
        rank0_print("  [3/7] 清理 test_sync 目录...")
        self._clean_test_sync_dir()

        # 4. 清理 train actor 目录中残留的 traj 和图像
        rank0_print("  [4/7] 清理 train actor 目录...")
        self._clean_train_actor_dirs()

        if skip_actual_test:
            # 跳过实际的 test 运行（步骤 5-7）
            rank0_print("  [5-7/7] 跳过实际 test 运行 (--skip_test)...")
            rank0_print("         - 不启动 test containers")
            rank0_print("         - 不等待 test 结果")
            rank0_print("         - 不收集 test 数据")

            # 直接重新启动 train containers
            rank0_print("  [恢复] 重新启动 train containers...")
            self.start_collection_containers(mode='train')

            steps = 0
            results = []
        else:
            # 正常执行 test（步骤 5-7）
            # 5. 重新启动 containers (test 模式，会自动设置 signal)
            rank0_print("  [5/7] 重新启动 containers (test 模式)...")
            self.start_collection_containers(mode='test')

            # 6. 等待所有 world_assignments 都有 test 结果
            rank0_print("  [6/7] 等待所有 world 完成 test...")
            self._wait_for_all_test_results(timeout=timeout)

            # 7. 收集结果
            rank0_print("  [7/7] 收集 test 结果...")
            steps, results = self._collect_test_results()

            self.start_collection_containers(mode='train')

        # 注意: test containers 运行完会自动退出
        # train containers 会在下次 collect_train() 时自动启动

        # 计算 nav_metric 统计
        if results:
            nav_metrics = [r.get('nav_metric', 0) for r in results]
            avg_nav_metric = np.mean(nav_metrics)
            success_count = sum(1 for r in results if r.get('ep_status') == 'success')
            success_rate = success_count / len(results) * 100
        else:
            avg_nav_metric = 0
            success_rate = 0

        rank0_print(f"\n{'='*60}")
        if skip_actual_test:
            rank0_print(f"  ✓ Test 阶段完成 (跳过实际运行)")
            rank0_print(f"  ✓ 已保存 policy 并清理残留数据")
            rank0_print(f"  ✓ Train containers 已重新启动")
        else:
            rank0_print(f"  ✓ Test 完成: {len(results)} episodes, {steps} steps")
            rank0_print(f"  ✓ Nav Metric: {avg_nav_metric:.4f} (avg)")
            rank0_print(f"  ✓ Success Rate: {success_rate:.1f}%")
            rank0_print(f"  ✓ Train containers 已重新启动")
        rank0_print(f"{'='*60}\n")

        return steps, results

    def _clean_test_sync_dir(self):
        """清理 test_sync 目录中的旧文件和 sync 目录中的 actor_*.done 文件"""
        # Debug 模式：跳过清理
        if self.no_cleanup:
            rank0_print(f"    [no_cleanup] 跳过清理 test_sync 目录")
            return

        import glob

        # 1. 清理 test_sync 目录中的旧 test 结果
        test_pattern = join(self.test_sync_dir, 'test_*.pickle')
        old_test_files = glob.glob(test_pattern)

        for f in old_test_files:
            try:
                os.remove(f)
            except OSError:
                pass

        # 2. 清理 sync 目录中的 actor_*.done 文件（重置 episode 计数）
        done_pattern = join(self.sync_dir, 'actor_*.done')
        old_done_files = glob.glob(done_pattern)

        for f in old_done_files:
            try:
                os.remove(f)
            except OSError:
                pass

        rank0_print(f"    清理了 {len(old_test_files)} 个旧 test 文件, {len(old_done_files)} 个 actor done 文件")

    def _clean_train_actor_dirs(self):
        """清理 train actor 目录中残留的 traj 和图像"""
        # Debug 模式：跳过清理
        if self.no_cleanup:
            rank0_print(f"    [no_cleanup] 跳过清理 train actor 目录")
            return

        import glob

        traj_count = 0
        img_count = 0

        for world_id in self.train_world_ids:
            actor_dir = join(self.buffer_path, f'actor_{world_id}')
            if not exists(actor_dir):
                continue

            # 清理 traj_*.pickle
            for f in glob.glob(join(actor_dir, 'traj_*.pickle')):
                try:
                    os.remove(f)
                    traj_count += 1
                except OSError:
                    pass

            # 清理 FTRL_*.png
            for f in glob.glob(join(actor_dir, 'FTRL_*.png')):
                try:
                    os.remove(f)
                    img_count += 1
                except OSError:
                    pass

        rank0_print(f"    清理了 {traj_count} 个 traj 文件, {img_count} 个图像文件")

    def _clean_test_actor_dirs(self):
        """清理 test actor 目录中残留的 traj 和图像"""
        # Debug 模式：跳过清理
        if self.no_cleanup:
            rank0_print(f"    [no_cleanup] 跳过清理 test actor 目录")
            return

        import glob

        traj_count = 0
        img_count = 0

        for world_id in self.test_world_ids:
            actor_dir = join(self.buffer_path, f'actor_{world_id}')
            if not exists(actor_dir):
                continue

            # 清理 traj_*.pickle
            for f in glob.glob(join(actor_dir, 'traj_*.pickle')):
                try:
                    os.remove(f)
                    traj_count += 1
                except OSError:
                    pass

            # 清理 FTRL_*.png
            for f in glob.glob(join(actor_dir, 'FTRL_*.png')):
                try:
                    os.remove(f)
                    img_count += 1
                except OSError:
                    pass

        if traj_count > 0 or img_count > 0:
            rank0_print(f"    清理了 test actor 目录: {traj_count} 个 traj 文件, {img_count} 个图像文件")

    def _wait_for_all_test_results(self, timeout=900, completion_threshold=1.0):
        """
        等待所有 world_assignments 中的 world 都有 test 结果

        检测方式: test_sync 目录中是否有 test_{world_id}_*.pickle 文件
        退出条件: 1) 所有 world 完成（100%），或 2) 超时（15分钟）

        注意: test 模式强制要求 100% 完成，不像 train 可以 60% 就开始

        Args:
            timeout: 超时时间（秒，默认 900 = 15 分钟）
            completion_threshold: 完成度阈值（默认 1.0 = 100%，test 模式不建议修改）
        """
        import glob

        # 获取所有需要测试的 world_id (使用 test_assignments)
        all_world_ids = set(self.test_world_ids)

        total_worlds = len(all_world_ids)
        required_worlds = int(total_worlds * completion_threshold)

        rank0_print(f"    需要等待 {total_worlds} 个 world 完成 test")
        rank0_print(f"    退出条件: 完成 {total_worlds} 个（100%）或 {required_worlds} 个（{completion_threshold*100:.0f}%）或超时 {timeout}s")

        start_time = time.time()
        completed_worlds = set()

        while len(completed_worlds) < total_worlds:
            elapsed = time.time() - start_time

            # 检查每个未完成的 world
            for world_id in all_world_ids:
                if world_id in completed_worlds:
                    continue

                # 检查是否有该 world 的 test 文件
                test_pattern = join(self.test_sync_dir, f'test_{world_id}_*.pickle')
                test_files = glob.glob(test_pattern)

                if test_files:
                    completed_worlds.add(world_id)
                    completion_pct = len(completed_worlds) / total_worlds * 100
                    rank0_print(f"    ✓ world_{world_id} 完成 test ({len(completed_worlds)}/{total_worlds}, {completion_pct:.1f}%)")

            # 检查退出条件
            completion_pct = len(completed_worlds) / total_worlds

            # 条件1: 达到完成度阈值
            if len(completed_worlds) >= required_worlds and len(completed_worlds) < total_worlds:
                rank0_print(f"\n    ✓ 完成度达到 {completion_pct*100:.1f}% (>= {completion_threshold*100:.0f}%)")
                not_completed = all_world_ids - completed_worlds
                rank0_print(f"    未完成的 world: {list(not_completed)[:20]}...")
                break

            # 条件2: 超时
            if elapsed > timeout:
                rank0_print(f"\n    ⚠️ 超时 ({timeout}s = {timeout/60:.0f} 分钟)!")
                rank0_print(f"    已完成: {len(completed_worlds)}/{total_worlds} ({completion_pct*100:.1f}%)")
                not_completed = all_world_ids - completed_worlds
                rank0_print(f"    未完成的 world: {list(not_completed)[:20]}...")
                break

            # 显示进度
            rank0_print(f"    等待中... {len(completed_worlds)}/{total_worlds} ({completion_pct*100:.1f}%) 完成, "
                  f"已等待 {int(elapsed)}s", end="\r")
            time.sleep(5)

        final_pct = len(completed_worlds) / total_worlds * 100
        rank0_print(f"\n    ✓ {len(completed_worlds)}/{total_worlds} ({final_pct:.1f}%) 个 world 完成 test")

    def _collect_test_results(self):
        """收集 test_sync 目录中的所有 test 结果"""
        import glob
        import re

        steps = 0
        results = []

        test_pattern = join(self.test_sync_dir, 'test_*.pickle')
        test_files = glob.glob(test_pattern)

        if not test_files:
            rank0_print(f"    Warning: No test files found in {self.test_sync_dir}")
            return steps, results

        test_files = self.sort_traj_name(test_files)

        for test_file in test_files:
            try:
                if os.path.getsize(test_file) == 0:
                    if not self.no_cleanup:
                        os.remove(test_file)
                    continue

                with open(test_file, 'rb') as f:
                    traj = pickle.load(f)

                if not traj:
                    continue

                steps += len(traj)

                if len(traj) > 0 and len(traj[-1]) >= 7:
                    info = traj[-1][4]
                    results.append(dict(
                        ep_rew=sum(t[2] for t in traj),
                        ep_len=len(traj),
                        ep_time=info.get('time', 0),
                        ep_status=info.get('status', 'unknown'),
                        world=info.get('world', 'unknown'),
                        collision=info.get('collision', 0),
                        nav_metric=traj[-1][6]  # 直接从轨迹元素索引6提取（与collect_train一致）
                    ))

                    # 获取 actor 目录 (从 world 名称提取)
                    world_str = info.get('world', '')
                    world_match = re.search(r'(\d+)', world_str)
                    if world_match:
                        world_id = int(world_match.group(1))
                        actor_dir = join(self.buffer_path, f'actor_{world_id}')

                        # 清理图像文件
                        self._cleanup_test_images(traj, actor_dir)

                # 删除已处理的 test 文件
                if not self.no_cleanup:
                    os.remove(test_file)

            except Exception as e:
                rank0_print(f"    Error loading test file {test_file}: {e}")
                continue

        return steps, results

    def _cleanup_test_images(self, traj, actor_dir):
        """清理 test 轨迹对应的图像文件"""
        # Debug 模式：跳过清理
        if self.no_cleanup:
            return

        import re

        if not exists(actor_dir):
            return

        first_img_label = None
        for transition in traj:
            if len(transition) < 5:
                continue
            info = transition[4]
            img_label = info.get('img_label')

            if img_label and isinstance(img_label, str):
                if first_img_label is None:
                    first_img_label = img_label

                img_path = join(actor_dir, img_label)
                if exists(img_path):
                    try:
                        os.remove(img_path)
                    except OSError:
                        pass

        # 删除第一个 img_label 的上一帧
        if first_img_label:
            match = re.match(r'(\w+_)(\d+)(\.png)', first_img_label)
            if match:
                prefix, num_str, suffix = match.groups()
                prev_num = int(num_str) - 1
                if prev_num >= 0:
                    prev_img_label = f"{prefix}{prev_num:06d}{suffix}"
                    prev_img_path = join(actor_dir, prev_img_label)
                    if exists(prev_img_path):
                        try:
                            os.remove(prev_img_path)
                        except OSError:
                            pass


# 为了向后兼容，创建别名
LocalCollector = VLMLocalCollector
CondorCollector = VLMCondorCollector
ReplayBuffer = VLMReplayBuffer
