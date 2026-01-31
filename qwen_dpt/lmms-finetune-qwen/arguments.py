from typing import Dict, Optional, List
from dataclasses import dataclass, field

import transformers

from supported_models import MODEL_HF_PATH, MODEL_FAMILIES


@dataclass
class ModelArguments:
    model_id: str = field(default="llava-1.5-7b")
    model_local_path: Optional[str] = field(default=None)
    planner: Optional[str] = field(
        default=None,
        metadata={"help": "Planner type (dwa/teb/mppi/ddp). Used to determine num_params for regression."}
    )
    num_params: Optional[int] = field(
        default=None,
        metadata={"help": "Number of parameters to predict. If not set, will be inferred from planner."}
    )
    head_type: str = field(
        default="simple_mlp",
        metadata={"help": "Regression head type: simple_mlp, transformer, or dpt"}
    )
    use_history: bool = field(
        default=False,
        metadata={"help": "Whether to use history frames for temporal information"}
    )
    num_history_frames: int = field(
        default=2,
        metadata={"help": "Number of history frames to use (default: 2)"}
    )
    history_dim: int = field(
        default=256,
        metadata={"help": "Dimension of history encoder output (default: 256)"}
    )
    history_image_size: int = field(
        default=224,
        metadata={"help": "Image size for history frames (default: 224)"}
    )

    def __post_init__(self):
        assert self.model_id in MODEL_HF_PATH, f"Unknown model_id: {self.model_id}"
        self.model_hf_path: str = MODEL_HF_PATH[self.model_id]
        assert self.model_id in MODEL_FAMILIES, f"Unknown model_id: {self.model_id}"
        self.model_family_id: str = MODEL_FAMILIES[self.model_id]

        if not self.model_local_path:
            self.model_local_path = self.model_hf_path

        # 为回归模型设置 num_params
        if self.model_family_id == "qwen2.5-vl-regression":
            if self.num_params is None:
                if self.planner is not None:
                    from planner_configs import get_num_params
                    self.num_params = get_num_params(self.planner)
                    print(f"[INFO] Set num_params={self.num_params} for planner '{self.planner}'")
                else:
                    self.num_params = 7  # 默认值
                    print(f"[WARN] planner not specified, using default num_params=7")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data json file."}
    )
    eval_data_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the evaluation data json file."}
    )
    image_folder: Optional[str] = field(default=None)
    video_folder: Optional[str] = field(default=None)
    num_frames: Optional[int] = field(default=8)
    user_key: Optional[str] = field(default="human")
    assistant_key: Optional[str] = field(default="gpt")
    label_noise_std: float = field(
        default=0.0,
        metadata={"help": "Standard deviation of Gaussian noise to add to labels (0.0 = no noise)"}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_max_length: int = field(
        default=1024,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_flash_attn: bool = field(default=False)
    train_vision_encoder: bool = field(default=False)
    train_vision_projector: bool = field(default=False)
    mask_question_tokens: bool = field(default=True)
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of evaluation samples (for faster debugging)"}
    )

    def __post_init__(self):
        super().__post_init__()
        self.remove_unused_columns = False


@dataclass
class LoraArguments:
    use_lora: bool = field(default=True)
    use_vision_lora: bool = field(default=True)
    q_lora: bool = field(default=False)
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    lora_weight_path: str = ""
    lora_bias: str = "none"