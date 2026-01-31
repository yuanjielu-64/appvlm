"""
Loader for Qwen2.5-VL Regression Model with DPT Head
"""

from typing import Tuple
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, PreTrainedTokenizer, AutoConfig
import sys
from pathlib import Path

# 添加 models 目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))
from models import Qwen2_5_VLForRegression

from . import register_loader
from .base import BaseModelLoader


@register_loader("qwen2.5-vl-regression")
class Qwen2_5_VLRegressionModelLoader(BaseModelLoader):
    """
    加载 Qwen2.5-VL 回归模型

    支持三种 head：
    - 'simple_mlp': 简单 MLP（默认）
    - 'transformer': Transformer attention head
    - 'dpt': DPT 多层融合 head
    """

    def __init__(
        self,
        num_params: int = 7,
        head_type: str = 'simple_mlp',
        num_layers_to_extract: int = 4,
        dropout: float = 0.1,
        use_history: bool = False,
        num_history_frames: int = 2,
        history_dim: int = 256,
        param_weights: list = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_params = num_params
        self.head_type = head_type
        self.num_layers_to_extract = num_layers_to_extract
        self.dropout = dropout
        self.use_history = use_history
        self.num_history_frames = num_history_frames
        self.history_dim = history_dim
        self.param_weights = param_weights

    def load(self, load_model: bool = True) -> Tuple[Qwen2_5_VLForRegression, PreTrainedTokenizer, AutoProcessor, AutoConfig]:
        """
        加载模型、tokenizer、processor 和 config
        """
        if load_model:
            # 1. 加载 base Qwen2.5-VL 模型
            # 移除 device_map，让 DeepSpeed 处理设备分配
            loading_kwargs = self.loading_kwargs.copy()
            loading_kwargs.pop('device_map', None)  # 移除 device_map，避免与 DeepSpeed 冲突

            base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_local_path,
                **loading_kwargs,
            )

            # 2. 包装为带回归头的模型
            # 只在 use_history=True 且 num_history_frames > 0 时启用 history encoder
            use_history_encoder = self.use_history and self.num_history_frames > 0

            model = Qwen2_5_VLForRegression(
                base_model=base_model,
                num_params=self.num_params,
                head_type=self.head_type,
                num_layers_to_extract=self.num_layers_to_extract,
                dropout=self.dropout,
                use_history=use_history_encoder,
                num_history_frames=self.num_history_frames if use_history_encoder else 0,
                history_dim=self.history_dim if use_history_encoder else 0,
                param_weights=self.param_weights,
            )

            print(f"  ✓ Using '{self.head_type}' regression head")
            if use_history_encoder:
                print(f"  ✓ History encoder enabled: {self.num_history_frames} frames, dim={self.history_dim}")
            else:
                print(f"  ✓ History encoder disabled (num_history_frames={self.num_history_frames})")
            if self.param_weights is not None:
                print(f"  ✓ Using weighted MSE loss: {self.param_weights}")

            # 设置 hidden_size (for deepspeed)
            model.config.hidden_size = base_model.config.hidden_size

            # 将 regression head 转换为与 base_model 相同的 dtype
            # 这样可以避免训练时的 dtype 不匹配问题
            if hasattr(base_model, 'dtype'):
                model.regression_head.to(base_model.dtype)
            elif loading_kwargs.get('torch_dtype') is not None:
                model.regression_head.to(loading_kwargs['torch_dtype'])
        else:
            model = None

        # 3. 加载 processor 和 tokenizer
        processor = AutoProcessor.from_pretrained(self.model_hf_path, use_fast=False)
        tokenizer = processor.tokenizer
        config = AutoConfig.from_pretrained(self.model_local_path)

        return model, tokenizer, processor, config
