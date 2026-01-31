"""
配置文件
"""

import argparse
import os
from dataclasses import dataclass, field
from typing import Optional
from planner_configs import get_num_params


@dataclass
class TrainingConfig:
    """训练配置"""

    # 数据路径（必需字段）
    train_json: str = field(
        metadata={"help": "训练数据 JSON 文件路径"}
    )
    image_folder: str = field(
        metadata={"help": "图像文件夹根目录"}
    )

    # 数据路径（可选字段）
    eval_json: Optional[str] = field(
        default=None,
        metadata={"help": "验证数据 JSON 文件路径"}
    )

    # 输出路径
    output_dir: str = field(
        default="./output",
        metadata={"help": "输出目录"}
    )

    # 模型配置
    num_params: int = field(
        default=8,
        metadata={"help": "输出参数数量"}
    )
    num_history_frames: int = field(
        default=2,
        metadata={"help": "历史帧数量"}
    )
    vision_model: str = field(
        default="vit_base_patch16_224",
        metadata={"help": "Vision Transformer 模型名称 (vit_base_patch16_224, vit_large_patch16_224, vit_huge_patch14_224)"}
    )
    vision_pretrained: bool = field(
        default=True,
        metadata={"help": "是否使用预训练权重"}
    )
    vision_freeze: bool = field(
        default=False,
        metadata={"help": "是否冻结 vision encoder"}
    )
    d_model: int = field(
        default=768,
        metadata={"help": "Transformer 隐藏维度"}
    )
    nhead: int = field(
        default=8,
        metadata={"help": "注意力头数"}
    )
    num_transformer_layers: int = field(
        default=4,
        metadata={"help": "Transformer 层数"}
    )
    dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout 比例"}
    )
    use_velocity: bool = field(
        default=True,
        metadata={"help": "是否使用速度信息"}
    )

    # 数据配置
    image_size: int = field(
        default=224,
        metadata={"help": "图像大小"}
    )
    normalize_params: bool = field(
        default=True,
        metadata={"help": "是否归一化参数"}
    )
    eval_samples: int = field(
        default=2000,
        metadata={"help": "评估时使用的样本数量，0表示使用全部"}
    )

    # 训练超参数
    num_epochs: int = field(
        default=50,
        metadata={"help": "训练轮数"}
    )
    batch_size: int = field(
        default=16,
        metadata={"help": "批次大小"}
    )
    learning_rate: float = field(
        default=1e-4,
        metadata={"help": "学习率"}
    )
    weight_decay: float = field(
        default=1e-4,
        metadata={"help": "权重衰减"}
    )
    warmup_epochs: int = field(
        default=5,
        metadata={"help": "学习率预热轮数"}
    )
    lr_scheduler: str = field(
        default="cosine",
        metadata={"help": "学习率调度器: cosine, step, plateau"}
    )

    # 硬件配置
    device: str = field(
        default="cuda",
        metadata={"help": "设备: cuda, cpu"}
    )
    num_workers: int = field(
        default=4,
        metadata={"help": "DataLoader 工作线程数"}
    )
    mixed_precision: bool = field(
        default=True,
        metadata={"help": "是否使用混合精度训练"}
    )

    # 日志和保存
    log_interval: int = field(
        default=100,
        metadata={"help": "日志打印间隔（步数）"}
    )
    eval_interval: int = field(
        default=1,
        metadata={"help": "验证间隔（轮数）"}
    )
    save_interval: int = field(
        default=5,
        metadata={"help": "模型保存间隔（轮数）"}
    )
    save_steps: int = field(
        default=5000,
        metadata={"help": "模型保存间隔（步数），0表示禁用"}
    )
    save_total_limit: int = field(
        default=3,
        metadata={"help": "最多保留的step checkpoint数量（不包括best和latest）"}
    )
    save_best: bool = field(
        default=True,
        metadata={"help": "是否保存最佳模型"}
    )

    # 其他
    seed: int = field(
        default=42,
        metadata={"help": "随机种子"}
    )
    resume: Optional[str] = field(
        default=None,
        metadata={"help": "从 checkpoint 恢复训练"}
    )


def get_config():
    """从命令行参数获取配置"""
    parser = argparse.ArgumentParser(description="Transformer IL Training")

    # Planner 配置（必选，用于自动设置路径和参数）
    parser.add_argument("--planner", type=str, required=True,
                       choices=["dwa", "teb", "mppi", "ddp"],
                       help="Planner name (dwa, teb, mppi, ddp)")

    # 数据路径（可选，默认根据 planner 自动生成）
    parser.add_argument("--data_root", type=str,
                       default="/home/yuanjielu/robot_navigation/noetic/app_data",
                       help="Data root directory")
    parser.add_argument("--train_json", type=str, default=None,
                       help="Training JSON path (default: auto-generated from planner)")
    parser.add_argument("--eval_json", type=str, default=None,
                       help="Evaluation JSON path (default: auto-generated from planner)")
    parser.add_argument("--image_folder", type=str, default=None,
                       help="Image folder path (default: auto-generated from planner)")
    parser.add_argument("--output_dir", type=str, default="./output",
                       help="Output directory")

    # 模型配置（num_params 会根据 planner 自动设置）
    parser.add_argument("--num_params", type=int, default=None,
                       help="Number of parameters (default: auto-detected from planner)")
    parser.add_argument("--num_history_frames", type=int, default=2)
    parser.add_argument("--vision_model", type=str, default="vit_base_patch16_224")
    parser.add_argument("--vision_pretrained", action="store_true", default=True)
    parser.add_argument("--vision_freeze", action="store_true", default=False)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_transformer_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use_velocity", action="store_true", default=True)

    # 数据配置
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--normalize_params", action="store_true", default=True)
    parser.add_argument("--eval_samples", type=int, default=2000,
                       help="Number of samples for evaluation (0 for all)")

    # 训练超参数
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")

    # 硬件配置
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--mixed_precision", action="store_true", default=True)

    # 日志和保存
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=5000,
                       help="Save checkpoint every N steps (0 to disable)")
    parser.add_argument("--save_total_limit", type=int, default=3,
                       help="Max number of step checkpoints to keep")
    parser.add_argument("--save_best", action="store_true", default=True)

    # 其他
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)

    args = parser.parse_args()

    # 根据 planner 自动设置路径和参数
    planner_lower = args.planner.lower()

    # 1. 自动设置 num_params（如果未指定）
    if args.num_params is None:
        args.num_params = get_num_params(planner_lower)
        print(f"Auto-detected num_params for {planner_lower}: {args.num_params}")

    # 2. 自动设置数据路径（如果未指定）
    planner_data_dir = os.path.join(args.data_root, f"{planner_lower}_heurstic")

    if args.train_json is None:
        args.train_json = os.path.join(planner_data_dir, "splits_200k/chunk_000.json")
        print(f"Auto-set train_json: {args.train_json}")

    if args.eval_json is None:
        args.eval_json = os.path.join(planner_data_dir, "splits_200k/chunk_000.json")
        print(f"Auto-set eval_json: {args.eval_json}")

    if args.image_folder is None:
        args.image_folder = planner_data_dir
        print(f"Auto-set image_folder: {args.image_folder}")

    # 3. 自动设置 output_dir（包含 planner 名称）
    if args.output_dir == "./output":
        args.output_dir = f"./output/{planner_lower}_transformer_il"
        print(f"Auto-set output_dir: {args.output_dir}")

    return args


if __name__ == "__main__":
    config = get_config()
    print(config)
