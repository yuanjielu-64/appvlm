"""
Data Collator for Qwen2.5-VL Regression Task

用于导航参数回归：
- 输入：costmap/场景图像
- 输出：7个导航参数（浮点数）

支持历史帧：
- history_images: 历史帧图像路径列表
- 用轻量 CNN + Transformer 编码，与 VLM 特征融合
"""

import os
import re
from typing import Dict, List, Sequence, Optional
import PIL
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T

from . import register_collator
from .base import BaseDataCollator


SYSTEM_MESSAGE = "You are a navigation parameter prediction assistant."
DEFAULT_IM_START_TOKEN = "<|im_start|>"
DEFAULT_IM_END_TOKEN = "<|im_end|>"


@register_collator("qwen2.5-vl-regression")
class Qwen2_5_VLRegressionDataCollator(BaseDataCollator):
    def __init__(
        self,
        param_mean: np.ndarray = None,
        param_std: np.ndarray = None,
        normalize: bool = True,
        planner: Optional[str] = None,
        use_history: bool = False,
        num_history_frames: int = 2,
        history_image_size: int = 224,
        label_noise_std: float = 0.0,  # 标签噪音的标准差
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.normalize = normalize
        self.planner = planner
        self.use_history = use_history
        self.num_history_frames = num_history_frames
        self.history_image_size = history_image_size
        self.label_noise_std = label_noise_std

        # 历史帧的图像预处理（简单的 resize + normalize）
        # 只在 use_history=True 且 num_history_frames > 0 时初始化
        if use_history and num_history_frames > 0:
            self.history_transform = T.Compose([
                T.Resize((history_image_size, history_image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.history_transform = None
            self.use_history = False  # 强制禁用

        if normalize and (param_mean is None or param_std is None):
            print("[WARNING] normalize=True but param_mean/param_std not provided. Normalization disabled.")
            self.normalize = False
            self.param_mean_tensor = None
            self.param_std_tensor = None
        else:
            self.param_mean_tensor = torch.tensor(param_mean, dtype=torch.float32) if param_mean is not None else None
            self.param_std_tensor = torch.tensor(param_std, dtype=torch.float32) if param_std is not None else None

        self.default_params = None
        if planner is not None:
            try:
                import sys

                parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if parent_dir not in sys.path:
                    sys.path.insert(0, parent_dir)

                from planner_configs import get_param_ranges
                param_ranges = get_param_ranges(planner)
                # 使用参数范围的中间值作为默认值
                self.default_params = [(r[0] + r[1]) / 2.0 for r in param_ranges]
                print(f"[INFO] Loaded default params for {planner}: {self.default_params}")
            except Exception as e:
                print(f"[WARNING] Failed to load planner config for {planner}: {e}")
                self.default_params = None

    def _load_history_images(self, history_paths: List[str], image_folder: str = None) -> torch.Tensor:
        """
        加载并预处理历史帧图像

        Args:
            history_paths: 历史帧图像路径列表
            image_folder: 图像根目录（可选）

        Returns:
            history_tensor: [num_frames, 3, H, W]
        """
        history_tensors = []
        for path in history_paths:
            # 如果提供了 image_folder，拼接路径
            if image_folder and not os.path.isabs(path):
                full_path = os.path.join(image_folder, path)
            else:
                full_path = path

            try:
                img = Image.open(full_path).convert('RGB')
                img_tensor = self.history_transform(img)
                history_tensors.append(img_tensor)
            except Exception as e:
                print(f"[WARNING] Failed to load history image {full_path}: {e}")
                # 使用零张量作为占位符
                img_tensor = torch.zeros(3, self.history_image_size, self.history_image_size)
                history_tensors.append(img_tensor)

        # 如果历史帧不够，用第一帧填充
        while len(history_tensors) < self.num_history_frames:
            if history_tensors:
                history_tensors.insert(0, history_tensors[0].clone())
            else:
                history_tensors.append(torch.zeros(3, self.history_image_size, self.history_image_size))

        # 只保留最近的 num_history_frames 帧
        history_tensors = history_tensors[-self.num_history_frames:]

        return torch.stack(history_tensors, dim=0)  # [num_frames, 3, H, W]

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:

        if "images" in instances[0]:
            is_video = False
        elif "videos" in instances[0]:
            is_video = True
        else:
            raise ValueError("Each instance must contain 'images' or 'videos'")

        if not is_video:
            grid_key = "image_grid_thw"
            pixel_key = "pixel_values"
            videos = None
            images: List[List[PIL.Image.Image]] = [instance["images"] for instance in instances]
        else:
            grid_key = "video_grid_thw"
            pixel_key = "pixel_values_videos"
            images = None
            videos: List[np.ndarray] = [x for instance in instances for x in instance["videos"]]

        # 处理历史帧（如果启用且 num_history_frames > 0）
        batch_history_images = None
        if self.use_history and self.num_history_frames > 0:
            history_list = []
            for instance in instances:
                if "history_images" in instance and instance["history_images"]:
                    # history_images 可以是路径列表或已加载的图像
                    history_paths = instance["history_images"]
                    image_folder = instance.get("image_folder", None)
                    history_tensor = self._load_history_images(history_paths, image_folder)
                else:
                    # 没有历史帧时使用当前帧填充（或零张量）
                    history_tensor = torch.zeros(
                        self.num_history_frames, 3,
                        self.history_image_size, self.history_image_size
                    )
                history_list.append(history_tensor)
            batch_history_images = torch.stack(history_list, dim=0)  # [B, num_frames, 3, H, W]

        parameter_labels = []
        for instance in instances:
            if "parameters" in instance:
                params = instance["parameters"]
                if not isinstance(params, (list, tuple, np.ndarray)):
                    raise ValueError(f"parameters must be a list/array, got {type(params)}")

                expected_num_params = len(self.default_params)
                if len(params) != expected_num_params:
                    raise ValueError(f"Expected {expected_num_params} parameters, got {len(params)}")
                parameter_labels.append(params)
            else:

                if self.default_params is not None:
                    parameter_labels.append(self.default_params)
                else:

                    print("[WARNING] No parameters provided and no default params available. Using zeros.")
                    parameter_labels.append([0.0] * 7)

        labels = torch.tensor(parameter_labels, dtype=torch.float32)

        # 先归一化
        if self.normalize:
            mean = self.param_mean_tensor.to(labels.device)
            std = self.param_std_tensor.to(labels.device)
            labels = (labels - mean) / (std + 1e-8)

        # 在归一化之后添加高斯噪音 (推荐)
        # 这样所有参数受到相同强度的扰动，正则化效果更均匀
        if self.label_noise_std > 0:
            noise = torch.randn_like(labels) * self.label_noise_std
            labels = labels + noise

        system_prompts: List[str] = [
            instance.get("system_prompt", SYSTEM_MESSAGE)
            for instance in instances
        ]

        conversations: List[List] = [
            instance["conversations"]
            for instance in instances
        ]

        max_len = self.tokenizer.model_max_length

        batch_input_ids = []
        batch_pixel_values = []
        batch_vision_grid_thw = []

        for b_idx, (system_prompt, cur_convs) in enumerate(zip(system_prompts, conversations)):
            # 添加 system message（参考 qwen2_5_vl.py Line 90-93）
            system_message = f"{DEFAULT_IM_START_TOKEN}system\n{system_prompt}\n{DEFAULT_IM_END_TOKEN}\n"

            # 回归任务只有单轮对话（用户 prompt），不需要 assistant response
            # 兼容两种格式：
            # 新格式: conversations = ["<image>\nUser prompt..."]
            # 旧格式: conversations = [{"from": "human", "value": "..."}, ...]
            if isinstance(cur_convs[0], dict):
                # 旧格式：提取 "value" 字段
                user_prompt_raw = cur_convs[0]["value"]
            else:
                # 新格式：直接使用字符串
                user_prompt_raw = cur_convs[0]

            user_prompt = replace_image_tokens(user_prompt_raw, is_video=is_video)
            user_input = f"{DEFAULT_IM_START_TOKEN}user\n{user_prompt}\n{DEFAULT_IM_END_TOKEN}\n"

            # 合并 system + user prompt
            full_text = system_message + user_input

            # 使用 processor 处理文本和图像/视频
            if not is_video:
                inputs = self.processor(
                    text=[full_text],
                    images=images[b_idx],
                    videos=None,
                    padding=False,
                    return_tensors='pt'
                )
            else:
                inputs = self.processor(
                    text=[full_text],
                    images=None,
                    videos=videos[b_idx],
                    padding=False,
                    return_tensors='pt'
                )

            cur_input_ids = inputs['input_ids'].squeeze(0)
            cur_pixel_values = inputs[pixel_key]
            cur_vision_grid_thw = inputs[grid_key]

            # manual truncation
            if cur_input_ids.shape[0] > max_len:
                cur_input_ids = cur_input_ids[:max_len]

            cur_input_ids = cur_input_ids.unsqueeze(0)

            # padding
            if cur_input_ids.shape[1] < max_len:
                cur_input_ids = torch.cat([
                    cur_input_ids,
                    torch.full(
                        (cur_input_ids.shape[0], max_len - cur_input_ids.shape[1]),
                        self.PAD_TOKEN_ID,
                        dtype=cur_input_ids.dtype,
                        device=cur_input_ids.device
                    )
                ], dim=1)

            batch_input_ids.append(cur_input_ids)
            batch_pixel_values.append(cur_pixel_values)
            batch_vision_grid_thw.append(cur_vision_grid_thw)

        batch_input_ids = torch.cat(batch_input_ids, dim=0)
        batch_pixel_values = torch.cat(batch_pixel_values, dim=0)
        batch_vision_grid_thw = torch.cat(batch_vision_grid_thw, dim=0)

        data_dict = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_input_ids.ne(self.PAD_TOKEN_ID),
            "labels": labels,
        }
        data_dict[pixel_key] = batch_pixel_values
        data_dict[grid_key] = batch_vision_grid_thw

        # 添加历史帧（如果有）
        if batch_history_images is not None:
            data_dict["history_images"] = batch_history_images

        return data_dict


def replace_image_tokens(input_string, is_video=False):
    if is_video:
        input_string = input_string.replace("<video>"+'\n', "<|vision_start|>"+"<|video_pad|>"+"<|vision_end|>")
        input_string = input_string.replace("<video>", "<|vision_start|>"+"<|video_pad|>"+"<|vision_end|>")
    else:
        input_string = input_string.replace("<image>"+'\n', "<|vision_start|>"+"<|image_pad|>"+"<|vision_end|>")
        input_string = input_string.replace("<image>", "<|vision_start|>"+"<|image_pad|>"+"<|vision_end|>")

    return input_string
