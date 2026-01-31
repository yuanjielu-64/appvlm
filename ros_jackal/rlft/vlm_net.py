import warnings
# 抑制PEFT的冗长警告
warnings.filterwarnings('ignore', category=UserWarning, module='peft')

import torch
import torch.nn as nn
import sys
import os

# DDP 工具函数（需要在其他导入之前设置日志级别）
from rlft.ddp_utils import rank0_print, is_main_process, get_local_rank
from rlft.zero_utils import fix_model_for_deepspeed, get_zero_stage

# 非主进程禁用所有日志（必须在导入 transformers 之前设置）
_local_rank = get_local_rank()
if _local_rank != 0:
    # 禁用 transformers 日志
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    # 禁用 tqdm 进度条
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    # 禁用 DeepSpeed 日志
    os.environ["DEEPSPEED_LOG_LEVEL"] = "ERROR"
    # 禁用一般日志
    import logging
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("accelerate").setLevel(logging.ERROR)
    logging.getLogger("deepspeed").setLevel(logging.ERROR)

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from transformers import logging as transformers_logging
from peft import PeftModel

# 再次确保非主进程日志被禁用
if _local_rank != 0:
    transformers_logging.set_verbosity_error()

# 添加qwen_dpt模型路径
qwen_dpt_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "qwen_dpt/lmms-finetune-qwen/models"
)
sys.path.append(qwen_dpt_path)

try:
    from qwen2_5_vl_dpt_regression import DPTHead, HistoryEncoder
except ImportError:
    rank0_print(f"Warning: Cannot import from qwen2_5_vl_dpt_regression")
    rank0_print(f"Expected path: {qwen_dpt_path}")


class VLM_DPT_FeatureExtractor(nn.Module):
    """
    VLM+DPT特征提取器 - 从checkpoint加载

    支持两种模式：
    1. 独立模式：加载自己的 VLM + LoRA + DPT + History
    2. 共享模式：共享 VLM base，但独立的 LoRA + DPT + History

    Args:
        checkpoint_path: VLM+DPT监督学习checkpoint路径
        freeze_vlm: 是否冻结VLM参数 (推荐True，因为VLM太大)
        freeze_dpt: 是否冻结DPT head (Actor建议False进行FTRL微调)
        freeze_history: 是否冻结History Encoder (如果使用历史帧)
        device: 设备
        use_4bit: 是否使用4-bit量化 (推荐True节省显存)
        shared_vlm_base: 共享的VLM base模型（如果提供，不会重新加载VLM）
        name: 用于日志的名称标识（如 "Actor" 或 "Critic"）
        use_gradient_checkpointing: 是否使用激活检查点（节省显存，允许更大batch）
    """
    def __init__(
        self,
        checkpoint_path,
        freeze_vlm=True,
        freeze_dpt=False,
        freeze_history=None,  # None表示跟随freeze_dpt，也可以独立设置
        device="cuda",
        use_4bit=True,
        algorithm="DWA",  # 算法类型 (DWA/TEB/MPPI/DDP)
        shared_vlm_base=None,  # 共享的VLM base（用于Critic）
        name="FeatureExtractor",  # 用于日志标识
        use_gradient_checkpointing=False,  # 激活检查点（节省显存）
        compute_dtype=None  # 🔧 新增：计算dtype，None=自动检测（ZeRO-3 bf16 用 bf16，否则 float32）
    ):
        super().__init__()

        self.checkpoint_path = checkpoint_path
        self.device = device
        self.algorithm = algorithm.upper()  # 存储算法类型
        self.name = name
        self._owns_vlm = (shared_vlm_base is None)  # 是否拥有VLM（用于判断是否需要加载LoRA）

        # 获取分布式训练信息
        local_rank = get_local_rank()
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        zero_stage = int(os.environ.get("ZERO_STAGE", 0))

        # 🔧 确定计算 dtype
        # ZeRO + bf16 混合精度：所有可训练层必须使用 bf16
        # 否则会在 backward 时报 "Found dtype Float but expected BFloat16"
        if compute_dtype is not None:
            self.compute_dtype = compute_dtype
        elif zero_stage >= 2:
            # ZeRO-2/ZeRO-3 默认使用 bf16（与 accelerator 的 mixed_precision 一致）
            self.compute_dtype = torch.bfloat16
            rank0_print(f"[{name}] ZeRO-{zero_stage} detected: using bfloat16 for DPT/History")
        else:
            # 普通 DDP 或单卡：使用 float32 保证数值稳定性
            self.compute_dtype = torch.float32
            rank0_print(f"[{name}] Non-ZeRO mode: using float32 for DPT/History")

        # 1. 加载或共享 VLM base
        if shared_vlm_base is not None:
            # 共享模式：使用传入的VLM base
            rank0_print(f"[{name}] 使用共享的 VLM base（不重新加载）")
            self.base_model = shared_vlm_base
            base_vlm_params = sum(p.numel() for p in self.base_model.parameters())
            rank0_print(f"[{name}] ✓ 共享 VLM base: {base_vlm_params:,} parameters ({base_vlm_params/1e9:.2f}B)")
        else:
            # 独立模式：加载自己的VLM
            rank0_print(f"[{name}] Loading VLM from Qwen/Qwen2.5-VL-3B-Instruct...")

            # 使用本地缓存避免网络超时
            local_files_only = os.environ.get("HF_HUB_OFFLINE", "0") == "1"

            if use_4bit and "cuda" in str(device):
                # 4-bit 量化模式：使用 device_map（不兼容 ZeRO-3）
                rank0_print(f"[{name}] Using 4-bit quantization to save memory")
                if world_size > 1:
                    actual_device_map = {"": local_rank}
                    rank0_print(f"[{name}] DDP mode: LOCAL_RANK={local_rank}, device_map={actual_device_map}")
                else:
                    actual_device_map = device

                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                self.base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    "Qwen/Qwen2.5-VL-3B-Instruct",
                    quantization_config=bnb_config,
                    device_map=actual_device_map,
                    local_files_only=local_files_only
                )
            elif zero_stage == 3:
                # ZeRO-3 模式：不使用 device_map，让 DeepSpeed 管理分片
                rank0_print(f"[{name}] ZeRO-3 mode: Loading without device_map (DeepSpeed will shard)")
                self.base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    "Qwen/Qwen2.5-VL-3B-Instruct",
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,  # ZeRO-3 需要这个
                    local_files_only=local_files_only
                )
            else:
                # 普通 DDP 或 ZeRO-2：使用 device_map
                if world_size > 1:
                    actual_device_map = {"": local_rank}
                    rank0_print(f"[{name}] DDP/ZeRO-2 mode: LOCAL_RANK={local_rank}, device_map={actual_device_map}")
                else:
                    actual_device_map = device

                self.base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    "Qwen/Qwen2.5-VL-3B-Instruct",
                    torch_dtype=torch.bfloat16,
                    device_map=actual_device_map,
                    local_files_only=local_files_only
                )

            # 统计base VLM参数
            base_vlm_params = sum(p.numel() for p in self.base_model.parameters())
            rank0_print(f"[{name}] ✓ Base VLM loaded: {base_vlm_params:,} parameters ({base_vlm_params/1e9:.2f}B)")

            # 2. 加载LoRA (只有拥有VLM时才加载)
            lora_path = checkpoint_path
            if os.path.exists(os.path.join(checkpoint_path, "adapter_config.json")):
                rank0_print(f"[{name}] Loading LoRA from {lora_path}...")

                # 检查 adapter_model.safetensors 是否为空
                adapter_weights_path = os.path.join(lora_path, "adapter_model.safetensors")
                if os.path.exists(adapter_weights_path):
                    from safetensors.torch import load_file
                    adapter_state = load_file(adapter_weights_path)
                    non_empty_params = {k: v for k, v in adapter_state.items() if v.numel() > 0}

                    if len(non_empty_params) == 0:
                        rank0_print(f"⚠️  WARNING: adapter_model.safetensors is EMPTY (0 parameters)!")
                        rank0_print(f"⚠️  LoRA will be randomly initialized (not using trained weights)")
                        rank0_print(f"⚠️  This is OK for current testing, but will need to be fixed for full training")
                    else:
                        total_lora_params = sum(v.numel() for v in non_empty_params.values())
                        rank0_print(f"✓ LoRA adapter contains {len(non_empty_params)} keys, {total_lora_params:,} parameters ({total_lora_params/1e6:.2f}M)")

                self.base_model = PeftModel.from_pretrained(
                    self.base_model,
                    lora_path,
                    is_trainable=(not freeze_vlm)  # 如果VLM可训练，保持LoRA可训练
                )

                # 关键决策：是否merge LoRA
                if freeze_vlm:
                    # ZeRO-3 模式下不能 merge（参数被分片，无法直接操作）
                    if zero_stage == 3:
                        # ZeRO-3: 保持 LoRA 独立，但冻结
                        for param in self.base_model.parameters():
                            param.requires_grad = False
                        rank0_print(f"[{name}] ✓ LoRA kept separate (ZeRO-3 mode, frozen)")
                    else:
                        # 非 ZeRO-3：merge LoRA 以节省显存和加速推理
                        self.base_model = self.base_model.merge_and_unload()
                        rank0_print(f"[{name}] ✓ LoRA merged into base model (VLM frozen)")
                else:
                    # VLM可训练：保持LoRA作为独立层，以便训练和保存
                    lora_params = sum(p.numel() for n, p in self.base_model.named_parameters() if 'lora' in n.lower())
                    rank0_print(f"[{name}] ✓ LoRA loaded as trainable layers: {lora_params:,} parameters ({lora_params/1e6:.2f}M)")
            else:
                rank0_print(f"[{name}] No LoRA found, using base VLM")

            # Gradient Checkpointing 启用策略：
            # - ZeRO-3: 延迟到 accelerator.prepare() 之后（DeepSpeed Linear 替换冲突）
            # - ZeRO-2/其他: 直接启用（无冲突）
            if use_gradient_checkpointing:
                if zero_stage == 3:
                    # ZeRO-3: 延迟启用
                    self._pending_gradient_checkpointing = True
                    rank0_print(f"[{name}] Gradient checkpointing will be enabled after accelerator.prepare() (ZeRO-3)")
                else:
                    # ZeRO-2/其他: 直接启用
                    self._pending_gradient_checkpointing = False
                    self._enable_gradient_checkpointing_now(name)
            else:
                self._pending_gradient_checkpointing = False

        # 3. 加载DPT head
        regression_head_path = os.path.join(checkpoint_path, "regression_head", "pytorch_model.bin")
        if not os.path.exists(regression_head_path):
            # Fallback: 尝试不带子目录的路径
            regression_head_path = os.path.join(checkpoint_path, "pytorch_model.bin")

        # 3.1 读取历史帧配置
        history_config_path = os.path.join(checkpoint_path, "history_config.json")
        if os.path.exists(history_config_path):
            import json
            with open(history_config_path) as f:
                history_config = json.load(f)
                self.use_history = history_config.get('use_history', False)
                self.num_history_frames = history_config.get('num_history_frames', 2)
                self.history_dim = history_config.get('history_dim', 256)
            rank0_print(f"[{self.name}] History config: use_history={self.use_history}, num_frames={self.num_history_frames}")
        else:
            self.use_history = False
            self.num_history_frames = 2
            self.history_dim = 256
            rank0_print(f"[{self.name}] No history_config.json found, using use_history=False")

        # 3.2 创建DPT Head
        rank0_print(f"[{self.name}] Loading DPT head from {regression_head_path}...")
        self.dpt_head = DPTHead(
            hidden_size=2048,  # Qwen2.5-VL-3B (与checkpoint匹配)
            num_params=8,      # DDP有8个参数
            feature_dim=256,
            num_layers=4,
            use_history=self.use_history,  # 从配置读取
            history_dim=self.history_dim
        )

        if os.path.exists(regression_head_path):
            state_dict = torch.load(regression_head_path, map_location='cpu')
            # 统计加载的参数数量
            total_params = sum(p.numel() for p in state_dict.values() if p.numel() > 0)
            num_keys = len([k for k, v in state_dict.items() if v.numel() > 0])

            self.dpt_head.load_state_dict(state_dict, strict=False)
            rank0_print(f"[{self.name}] ✓ DPT head loaded: {num_keys} keys, {total_params:,} parameters ({total_params/1e6:.2f}M)")
        else:
            rank0_print(f"[{self.name}] ⚠️  Warning: DPT head not found at {regression_head_path}")
            rank0_print(f"[{self.name}] Using randomly initialized DPT head")

        # 🔧 修复：使用 self.compute_dtype（ZeRO-3 模式下必须用 bf16，否则 backward 报错）
        # ZeRO-3 + bf16 混合精度要求所有参与梯度计算的张量使用 bf16
        # 非 ZeRO-3 模式下使用 float32 保证数值稳定性
        self.dpt_head.to(device=device, dtype=self.compute_dtype)
        rank0_print(f"[{self.name}] ✓ DPT head moved to device={device}, dtype={self.compute_dtype}")

        # 3.3 加载History Encoder (如果使用)
        self.history_encoder = None
        if self.use_history:
            # 监督学习保存格式: history_encoder/pytorch_model.bin
            history_encoder_path = os.path.join(checkpoint_path, "history_encoder", "pytorch_model.bin")

            # Fallback: 尝试旧格式 history_encoder.pth
            if not os.path.exists(history_encoder_path):
                history_encoder_path = os.path.join(checkpoint_path, "history_encoder.pth")

            if os.path.exists(history_encoder_path):
                self.history_encoder = HistoryEncoder(
                    hidden_dim=self.history_dim,
                    num_frames=self.num_history_frames,
                    num_transformer_layers=2
                )
                history_state_dict = torch.load(history_encoder_path, map_location='cpu')

                # 统计参数数量
                total_history_params = sum(p.numel() for p in history_state_dict.values() if p.numel() > 0)
                num_history_keys = len([k for k, v in history_state_dict.items() if v.numel() > 0])

                self.history_encoder.load_state_dict(history_state_dict)
                # 🔧 修复：使用 self.compute_dtype（ZeRO-3 模式下必须用 bf16）
                self.history_encoder.to(device=device, dtype=self.compute_dtype)
                rank0_print(f"[{self.name}] ✓ History encoder loaded: {num_history_keys} keys, {total_history_params:,} parameters ({total_history_params/1e6:.2f}M)")
                rank0_print(f"[{self.name}] ✓ History encoder moved to device={device}, dtype={self.compute_dtype}")
            else:
                rank0_print(f"[{self.name}] ⚠️  Warning: use_history=True but history encoder not found")
                rank0_print(f"[{self.name}] Expected: {os.path.join(checkpoint_path, 'history_encoder/pytorch_model.bin')}")
                self.use_history = False  # 降级到不使用历史帧

        # 4. 冻结策略
        if freeze_vlm:
            for param in self.base_model.parameters():
                param.requires_grad = False
            rank0_print(f"[{self.name}] ✓ VLM frozen")
        else:
            rank0_print(f"[{self.name}] VLM trainable")

        if freeze_dpt:
            for param in self.dpt_head.parameters():
                param.requires_grad = False
            rank0_print(f"[{self.name}] ✓ DPT head frozen")
        else:
            rank0_print(f"[{self.name}] DPT head trainable")

        # History encoder独立冻结策略
        if self.history_encoder is not None:
            # 如果freeze_history未指定，跟随freeze_dpt
            freeze_history_final = freeze_history if freeze_history is not None else freeze_dpt

            if freeze_history_final:
                for param in self.history_encoder.parameters():
                    param.requires_grad = False
                rank0_print(f"[{self.name}] ✓ History encoder frozen")
            else:
                rank0_print(f"[{self.name}] History encoder trainable")

        self.feature_dim = 256  # DPT输出维度

        # 5. 加载processor (用于图像预处理)
        local_files_only = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            use_fast=True,  # 使用快速处理器，避免警告
            local_files_only=local_files_only
        )

        # 6. 参数统计汇总
        rank0_print("\n" + "="*70)
        rank0_print("📊 VLM_DPT_FeatureExtractor Parameter Summary")
        rank0_print("="*70)

        # 检测 ZeRO-3 模式（参数可能被分片，统计需要特殊处理）
        zero_stage = int(os.environ.get("ZERO_STAGE", "0"))
        if zero_stage == 3:
            rank0_print("⚠️  ZeRO-3 模式：参数被分片，以下统计可能不准确")

        # VLM Base Model - 区分 LoRA 和 base 参数
        # 使用 hasattr 检测 PEFT 模型，比 isinstance 更可靠
        is_peft = hasattr(self.base_model, 'peft_config') or hasattr(self.base_model, 'active_adapters')

        if is_peft:
            # PEFT 模式：区分 LoRA 和 base 参数
            lora_params = 0
            lora_trainable = 0
            base_params = 0
            base_trainable = 0

            for name, param in self.base_model.named_parameters():
                # 在 ZeRO-3 中，分片参数的 numel() 可能返回分片后的大小
                # 使用 ds_numel 获取原始大小（如果存在）
                if hasattr(param, 'ds_numel'):
                    numel = param.ds_numel
                else:
                    numel = param.numel()

                if 'lora_' in name.lower():
                    lora_params += numel
                    if param.requires_grad:
                        lora_trainable += numel
                else:
                    base_params += numel
                    if param.requires_grad:
                        base_trainable += numel

            rank0_print(f"VLM Base (Qwen2.5-VL-3B, frozen):")
            rank0_print(f"  ├─ Total:      {base_params:>12,} parameters ({base_params/1e9:.2f}B)")
            rank0_print(f"  ├─ Trainable:  {base_trainable:>12,} parameters ({base_trainable/1e6:.2f}M)")
            rank0_print(f"  └─ Frozen:     {base_params-base_trainable:>12,} parameters ({(base_params-base_trainable)/1e9:.2f}B)")

            rank0_print(f"\nLoRA Adapters:")
            rank0_print(f"  ├─ Total:      {lora_params:>12,} parameters ({lora_params/1e6:.2f}M)")
            rank0_print(f"  ├─ Trainable:  {lora_trainable:>12,} parameters ({lora_trainable/1e6:.2f}M)")
            rank0_print(f"  └─ Frozen:     {lora_params-lora_trainable:>12,} parameters ({(lora_params-lora_trainable)/1e6:.2f}M)")

            vlm_total = base_params + lora_params
            vlm_trainable = base_trainable + lora_trainable
        else:
            # 非 PEFT 模式
            vlm_total = sum(p.ds_numel if hasattr(p, 'ds_numel') else p.numel()
                          for p in self.base_model.parameters())
            vlm_trainable = sum(p.ds_numel if hasattr(p, 'ds_numel') else p.numel()
                               for p in self.base_model.parameters() if p.requires_grad)
            rank0_print(f"VLM Base (Qwen2.5-VL-3B):")
            rank0_print(f"  ├─ Total:      {vlm_total:>12,} parameters ({vlm_total/1e9:.2f}B)")
            rank0_print(f"  ├─ Trainable:  {vlm_trainable:>12,} parameters ({vlm_trainable/1e6:.2f}M)")
            rank0_print(f"  └─ Frozen:     {vlm_total-vlm_trainable:>12,} parameters ({(vlm_total-vlm_trainable)/1e9:.2f}B)")

        # DPT Head
        dpt_total = sum(p.ds_numel if hasattr(p, 'ds_numel') else p.numel()
                       for p in self.dpt_head.parameters())
        dpt_trainable = sum(p.ds_numel if hasattr(p, 'ds_numel') else p.numel()
                          for p in self.dpt_head.parameters() if p.requires_grad)
        rank0_print(f"\nDPT Head:")
        rank0_print(f"  ├─ Total:      {dpt_total:>12,} parameters ({dpt_total/1e6:.2f}M)")
        rank0_print(f"  ├─ Trainable:  {dpt_trainable:>12,} parameters ({dpt_trainable/1e6:.2f}M)")
        rank0_print(f"  └─ Frozen:     {dpt_total-dpt_trainable:>12,} parameters ({(dpt_total-dpt_trainable)/1e6:.2f}M)")

        # History Encoder (如果存在)
        hist_total = 0
        hist_trainable = 0
        if self.history_encoder is not None:
            hist_total = sum(p.ds_numel if hasattr(p, 'ds_numel') else p.numel()
                            for p in self.history_encoder.parameters())
            hist_trainable = sum(p.ds_numel if hasattr(p, 'ds_numel') else p.numel()
                                for p in self.history_encoder.parameters() if p.requires_grad)
            rank0_print(f"\nHistory Encoder:")
            rank0_print(f"  ├─ Total:      {hist_total:>12,} parameters ({hist_total/1e6:.2f}M)")
            rank0_print(f"  ├─ Trainable:  {hist_trainable:>12,} parameters ({hist_trainable/1e6:.2f}M)")
            rank0_print(f"  └─ Frozen:     {hist_total-hist_trainable:>12,} parameters ({(hist_total-hist_trainable)/1e6:.2f}M)")
        else:
            rank0_print(f"\nHistory Encoder: Not loaded")

        # 总计
        total_params = vlm_total + dpt_total + hist_total
        total_trainable = vlm_trainable + dpt_trainable + hist_trainable

        rank0_print(f"\n{'─'*70}")
        rank0_print(f"Total Feature Extractor:")
        rank0_print(f"  ├─ Total:      {total_params:>12,} parameters ({total_params/1e9:.2f}B)")
        rank0_print(f"  ├─ Trainable:  {total_trainable:>12,} parameters ({total_trainable/1e6:.2f}M)")
        rank0_print(f"  ├─ Frozen:     {total_params-total_trainable:>12,} parameters ({(total_params-total_trainable)/1e9:.2f}B)")
        if total_params > 0:
            rank0_print(f"  └─ Trainable%: {100*total_trainable/total_params:>11.2f}%")
        rank0_print("="*70 + "\n")

        # 🔧 ZeRO-3 兼容性修复：在 accelerator.prepare() 之前修复 _parameters
        # 必须在模型完全创建后（包括 PEFT 加载后）立即调用
        if zero_stage == 3:
            rank0_print(f"[{self.name}] Applying ZeRO-3 compatibility fix...")
            total_fixed = 0
            # 修复 base_model（包含 PEFT 层）
            total_fixed += fix_model_for_deepspeed(self.base_model, verbose=False, name=f"{self.name}.base_model")
            # 修复 DPT head
            total_fixed += fix_model_for_deepspeed(self.dpt_head, verbose=False, name=f"{self.name}.dpt_head")
            # 修复 History encoder（如果存在）
            if self.history_encoder is not None:
                total_fixed += fix_model_for_deepspeed(self.history_encoder, verbose=False, name=f"{self.name}.history_encoder")
            rank0_print(f"[{self.name}] ✓ ZeRO-3 fix applied to {total_fixed} modules")

    def _enable_gradient_checkpointing_now(self, name=None):
        """实际启用 gradient checkpointing 的逻辑"""
        from rlft.ddp_utils import rank0_print
        if name is None:
            name = getattr(self, 'name', 'VLM')

        is_peft_model = 'PeftModel' in type(self.base_model).__name__

        if is_peft_model:
            self.base_model.enable_input_require_grads()

            lora_model = self.base_model.base_model
            if hasattr(lora_model, 'model'):
                qwen_model = lora_model.model
                if hasattr(qwen_model, 'gradient_checkpointing_enable'):
                    qwen_model.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False}
                    )
                    rank0_print(f"[{name}] ✓ Gradient checkpointing enabled on Qwen model")
            elif hasattr(lora_model, 'gradient_checkpointing_enable'):
                lora_model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
                rank0_print(f"[{name}] ✓ Gradient checkpointing enabled on LoraModel")
        else:
            if hasattr(self.base_model, 'gradient_checkpointing_enable'):
                self.base_model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
                rank0_print(f"[{name}] ✓ Gradient checkpointing enabled")

    def enable_gradient_checkpointing(self):
        """
        在 accelerator.prepare() 之后调用，启用 gradient checkpointing（ZeRO-3 专用）

        ZeRO-3 需要在 DeepSpeed hooks 设置完成后再启用，否则会有冲突。
        ZeRO-2 及其他模式在模型创建时直接启用，不需要调用此方法。
        """
        if not getattr(self, '_pending_gradient_checkpointing', False):
            return

        from rlft.ddp_utils import rank0_print
        rank0_print(f"[{self.name}] Enabling gradient checkpointing after accelerator.prepare() (ZeRO-3)...")
        self._enable_gradient_checkpointing_now(self.name)
        self._pending_gradient_checkpointing = False

    def forward(self, images, prompt=None, history_images=None,
                linear_vels=None, angular_vels=None, algorithm="DWA"):
        """
        Args:
            images: PIL.Image list 或 [B, 3, H, W] tensor (当前帧)
            prompt: str, 如果为None使用默认导航prompt
            history_images: [B, num_frames, 3, H, W] tensor 或 List[List[PIL.Image]] (历史帧，可选)
            linear_vels: List[float] (当前线速度，可选，用于生成prompt)
            angular_vels: List[float] (当前角速度，可选，用于生成prompt)
            algorithm: str (规划算法名称，默认DWA)

        Returns:
            features: [B, 256] 特征向量
        """
        # System prompt (与训练时一致)
        SYSTEM_PROMPT = """You are a navigation scene analyzer for Clearpath Jackal robot motion planning.
Scene Representation (Costmap Visualization):
- Scale: Each grid cell represents approximately 1 meter.
- Robot: Yellow square (0.508 m x 0.430 m footprint)
- Green arrow: forward driving direction (x-axis)
- Blue line: lateral direction (y-axis)
- Obstacles: Red points from laser scanner measurements
- Global Path: Black curve showing a kinematic reference trajectory that guides direction but does not encode dynamics or feasibility.
- Background: Grey grid is for visualization only and has no obstacle or cost meaning.
Your Role:
Analyze the spatial layout to understand:
- Obstacle distribution and density around the robot
- Available free space and safe clearance margins
- Path geometry (straight corridor, sharp turn, narrow passage)
- Motion feasibility given the current velocity state

Your navigation reasoning will be used downstream to select parameters for the local motion planner."""

        # 构建完整的chat messages
        if not isinstance(images, list):
            # Tensor -> 暂时不支持，因为chat template需要PIL.Image
            raise ValueError("QwenVLMFeatureExtractor only supports PIL.Image list in forward()")

        batch_size = len(images)
        texts = []

        for i in range(batch_size):
            # User prompt (包含速度信息)
            if linear_vels is not None and angular_vels is not None:
                user_prompt = f"""Current robot state:
- Linear velocity: {linear_vels[i]:.3f} m/s
- Angular velocity: {angular_vels[i]:.3f} rad/s

Target local planner: {algorithm}

Use the scene image and robot state to perform navigation reasoning about obstacle proximity, path curvature, free-space structure, and motion constraints."""
            else:
                user_prompt = f"""Target local planner: {algorithm}

Analyze the costmap to understand obstacle distribution and path geometry."""

            # 构建messages (与qwen_server_flash_attn.py一致)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT.strip()},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": images[i]},
                        {"type": "text", "text": user_prompt.strip()},
                    ],
                }
            ]

            # 应用chat template
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False  # 回归任务不生成assistant回复
            )
            texts.append(text)

        # Processor处理 (与qwen_server_flash_attn.py一致)
        inputs = self.processor(
            text=texts,
            images=images,
            videos=None,
            padding=False,  # 与训练时一致
            return_tensors="pt"
        ).to(self.device)

        # VLM前向 (获取hidden states)
        # 🔧 关键：使用 self.base_model(**inputs) 而不是 self.base_model.model(**inputs)
        # - base_model (Qwen2_5_VLForConditionalGeneration) 会先通过 vision encoder 处理 pixel_values
        # - base_model.model (Qwen2_5_VLModel) 不接受 pixel_values，只接受处理后的 inputs_embeds
        # - 参考: script/qwen/qwen_server.py:180-184
        is_vlm_frozen = not any(p.requires_grad for p in self.base_model.parameters())

        with torch.no_grad() if is_vlm_frozen else torch.enable_grad():
            outputs = self.base_model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
            multi_layer_hidden_states = outputs.hidden_states[-4:]

            # Debug: check VLM hidden states
            for i, hs in enumerate(multi_layer_hidden_states):
                if torch.isnan(hs).any() or torch.isinf(hs).any():
                    rank0_print(f"    [VLM] WARNING: hidden_state layer {-4+i} contains nan/inf!")
                    rank0_print(f"      shape: {hs.shape}, nan: {torch.isnan(hs).sum()}, inf: {torch.isinf(hs).sum()}")

        # 编码历史帧 (如果使用)
        history_feat = None
        if self.use_history and self.history_encoder is not None and history_images is not None:
            # 转换 List[List[PIL.Image]] -> [B, num_frames, 3, H, W] tensor
            if isinstance(history_images, list) and len(history_images) > 0 and isinstance(history_images[0], list):
                # VLMReplayBuffer format: List[List[PIL.Image]]
                import torchvision.transforms as T
                transform = T.Compose([
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

                batch_history = []
                # 从 history_encoder 读取实际需要的帧数
                required_frames = self.num_history_frames  # 从 __init__ 读取

                for batch_idx, hist_imgs in enumerate(history_images):
                    if hist_imgs is None or len(hist_imgs) == 0:
                        # 用零填充（使用 float32，与 history_encoder 一致）
                        batch_history.append(torch.zeros(required_frames, 3, 224, 224, dtype=torch.float32))
                    else:
                        # 智能处理：截取或填充到 required_frames
                        actual_frames = len(hist_imgs)

                        if actual_frames >= required_frames:
                            # 截取最新的 required_frames 帧
                            selected_imgs = hist_imgs[:required_frames]
                        else:
                            # 填充：用最后一帧重复填充
                            selected_imgs = hist_imgs + [hist_imgs[-1]] * (required_frames - actual_frames)

                        frames = [transform(img) for img in selected_imgs]
                        batch_history.append(torch.stack(frames))  # [required_frames, 3, H, W]

                # 关键：dtype 与 history_encoder 的参数一致（支持 DeepSpeed bf16）
                encoder_dtype = next(self.history_encoder.parameters()).dtype
                history_images = torch.stack(batch_history).to(device=self.device, dtype=encoder_dtype)  # [B, required_frames, 3, H, W]

            is_history_frozen = not any(p.requires_grad for p in self.history_encoder.parameters())

            # 🔧 DEBUG: 检查 history_encoder 输入和参数
            if torch.isnan(history_images).any():
                rank0_print(f"    [HistoryEncoder] WARNING: INPUT history_images contains NaN!")
                rank0_print(f"      shape: {history_images.shape}, nan count: {torch.isnan(history_images).sum()}")

            # 检查 history_encoder 参数是否有 NaN
            for name, param in self.history_encoder.named_parameters():
                if torch.isnan(param).any():
                    rank0_print(f"    [HistoryEncoder] WARNING: PARAM {name} contains NaN!")

            with torch.no_grad() if is_history_frozen else torch.enable_grad():
                history_feat = self.history_encoder(history_images)  # [B, history_dim]

            # 🔧 DEBUG: 检查输出
            if torch.isnan(history_feat).any():
                rank0_print(f"    [HistoryEncoder] WARNING: OUTPUT history_feat contains NaN!")
                rank0_print(f"      history_feat shape: {history_feat.shape}")
                rank0_print(f"      history_images dtype: {history_images.dtype}")
                rank0_print(f"      history_encoder dtype: {next(self.history_encoder.parameters()).dtype}")

        # DPT提取特征 (包含历史特征融合)
        is_dpt_frozen = not any(p.requires_grad for p in self.dpt_head.parameters())

        with torch.no_grad() if is_dpt_frozen else torch.enable_grad():
            features = self._extract_dpt_features(multi_layer_hidden_states, history_feat)

        return features

    def _extract_dpt_features(self, multi_layer_hidden_states, history_feat=None):
        """
        提取DPT的中间特征 (256-d pooled)，不经过最后的MLP回归层
        """
        B, seq_len, _ = multi_layer_hidden_states[0].shape

        # 🔧 关键修复：将 hidden states 转换为与 DPT head 参数一致的 dtype
        # 支持 DeepSpeed bf16 模式
        dpt_dtype = next(self.dpt_head.parameters()).dtype
        multi_layer_hidden_states = [hs.to(dtype=dpt_dtype) for hs in multi_layer_hidden_states]

        # 🔧 DEBUG: 检查 DPT 参数是否有 NaN
        dpt_has_nan = False
        for name, param in self.dpt_head.named_parameters():
            if torch.isnan(param).any():
                rank0_print(f"    [DPT] WARNING: PARAM {name} has NaN! requires_grad={param.requires_grad}")
                dpt_has_nan = True
        if dpt_has_nan:
            rank0_print(f"    [DPT] DPT head parameters have NaN!")

        # Step 1: 投影所有层到统一特征空间
        projected = []
        for i, (proj, hidden_state) in enumerate(zip(self.dpt_head.projections, multi_layer_hidden_states)):
            p = proj(hidden_state)
            if torch.isnan(p).any():
                rank0_print(f"    [DPT] WARNING: projection[{i}] output has NaN!")
            projected.append(p)

        # Step 2: 转换为Conv1d格式 [B, feature_dim, seq_len]
        projected = [p.transpose(1, 2) for p in projected]

        # Step 3: 渐进式融合 (top-down refinement)
        fused = projected[-1]
        for i in range(len(self.dpt_head.fusion_blocks) - 1, -1, -1):
            skip = projected[i]
            fused = self.dpt_head.fusion_blocks[i](fused, skip)
            if torch.isnan(fused).any():
                rank0_print(f"    [DPT] WARNING: fusion_block[{i}] output has NaN!")

        fused = fused.transpose(1, 2)  # [B, seq_len, feature_dim]

        # Step 4: Spatial attention pooling
        attention_weights = self.dpt_head.spatial_attention(fused)  # [B, seq_len, 1]
        if torch.isnan(attention_weights).any():
            rank0_print(f"    [DPT] WARNING: attention_weights has NaN!")
        pooled = (fused * attention_weights).sum(dim=1)  # [B, feature_dim]

        # Step 5: 历史特征融合 (如果使用)
        if self.use_history and history_feat is not None:
            # 🔧 DEBUG: 检查 dtype 和 NaN
            if torch.isnan(pooled).any():
                rank0_print(f"    [DPT] WARNING: pooled contains NaN BEFORE history fusion!")
            if torch.isnan(history_feat).any():
                rank0_print(f"    [DPT] WARNING: history_feat contains NaN!")
                rank0_print(f"      pooled dtype: {pooled.dtype}, history_feat dtype: {history_feat.dtype}")

            # 确保 dtype 一致（与 history_fusion 参数一致，支持 bf16）
            fusion_dtype = self.dpt_head.history_fusion[0].weight.dtype
            pooled = pooled.to(dtype=fusion_dtype)
            history_feat = history_feat.to(dtype=fusion_dtype)

            # 与监督学习保持一致的融合方式
            combined = torch.cat([pooled, history_feat], dim=-1)  # [B, feature_dim + history_dim]

            # history_fusion
            pooled = self.dpt_head.history_fusion(combined)  # [B, feature_dim]

            if torch.isnan(pooled).any():
                rank0_print(f"    [DPT] WARNING: pooled contains NaN AFTER history fusion!")
                # 检查 history_fusion 权重
                for i, layer in enumerate(self.dpt_head.history_fusion):
                    if hasattr(layer, 'weight') and torch.isnan(layer.weight).any():
                        rank0_print(f"      history_fusion[{i}] weight has NaN!")

        return pooled  # [B, 256]


class VLM_DPT_Actor(nn.Module):
    """
    VLM+DPT Actor - 用于FTRL

    架构: VLM+DPT特征提取 → FC层 → Action

    Args:
        feature_extractor: VLM_DPT_FeatureExtractor实例
        action_dim: 动作维度 (7个导航参数)
    """
    def __init__(self, feature_extractor, action_dim=7, algorithm="DWA"):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.action_dim = action_dim
        self.algorithm = algorithm.upper()  # 存储算法类型

        # 输出层: 256-d → action_dim
        self.fc = nn.Linear(feature_extractor.feature_dim, action_dim)

        # 初始化
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

        # 🔧 修复：FC 层使用与 feature_extractor 相同的 dtype（ZeRO-3 需要 bf16）
        self._sync_dtype()

    def _sync_dtype(self):
        """同步FC层dtype与feature_extractor（ZeRO-3 兼容）"""
        if hasattr(self.feature_extractor, 'compute_dtype'):
            target_dtype = self.feature_extractor.compute_dtype
            self.fc = self.fc.to(dtype=target_dtype)
            rank0_print(f"[Actor] FC layer dtype synced to {target_dtype}")

        # 🔧 ZeRO-3 兼容性修复
        if get_zero_stage() == 3:
            fixed = fix_model_for_deepspeed(self, verbose=False, name="Actor")
            if fixed > 0:
                rank0_print(f"[Actor] ✓ ZeRO-3 fix applied to {fixed} modules")

    def forward(self, images, prompt=None, history_images=None):
        """
        Args:
            images: PIL.Image list 或 [B, 3, H, W] tensor (当前帧)
                    或 List[(PIL.Image, List[PIL.Image])] (VLMReplayBuffer格式)
            prompt: str, 可选
            history_images: [B, num_frames, 3, H, W] tensor (历史帧，可选)

        Returns:
            action: [B, action_dim] 归一化到[-1, 1]
        """
        # Handle VLMReplayBuffer format: List[(img, history_imgs, linear_vel, angular_vel)]
        if isinstance(images, list) and len(images) > 0 and isinstance(images[0], tuple):
            # Unpack tuple format
            imgs = [item[0] for item in images]
            hists = [item[1] for item in images]
            # Extract velocities for prompt generation
            linear_vels = [item[2] if len(item) > 2 else 0.0 for item in images]
            angular_vels = [item[3] if len(item) > 3 else 0.0 for item in images]

            features = self.feature_extractor(
                imgs, prompt, hists,
                linear_vels=linear_vels,
                angular_vels=angular_vels,
                algorithm=self.algorithm  # 使用实例属性
            )
        else:
            # Normal format: separate images and history_images
            features = self.feature_extractor(images, prompt, history_images)

        # 转换dtype以匹配FC层 (DeepSpeed bf16模式下FC是bf16)
        fc_dtype = self.fc.weight.dtype
        features = features.to(dtype=fc_dtype)

        # Debug: check VLM features
        if torch.isnan(features).any() or torch.isinf(features).any():
            rank0_print(f"    [Actor] WARNING: VLM features contain nan/inf!")
            rank0_print(f"      features range: [{features.min():.4f}, {features.max():.4f}]")
            rank0_print(f"      nan count: {torch.isnan(features).sum()}, inf count: {torch.isinf(features).sum()}")

        # Debug: check FC weights
        if torch.isnan(self.fc.weight).any() or torch.isnan(self.fc.bias).any():
            rank0_print(f"    [Actor] WARNING: FC layer weights contain nan!")

        action = torch.tanh(self.fc(features))
        return action


class VLM_DPT_Critic(nn.Module):
    """
    VLM+DPT Critic - 用于FTRL

    架构: VLM+DPT特征提取 + Action编码 → 双Q网络

    TD3的Twin Critic设计: 共享特征提取器，独立的Q-head
    (Twin指的是两个独立的Q-value估计，不是两个特征提取器)

    Args:
        feature_extractor: 共享的VLM_DPT_FeatureExtractor (与Actor共享)
        action_dim: 动作维度
        detach_features: 是否detach特征（防止Critic梯度回传到VLM）
    """
    def __init__(
        self,
        feature_extractor,
        action_dim=7,
        detach_features=True  # 默认detach，保护Actor的VLM更新
    ):
        super().__init__()

        # 共享的特征提取器 (与Actor共享，VLM只需要跑1次)
        self.feature_extractor = feature_extractor
        self.detach_features = detach_features

        feature_dim = feature_extractor.feature_dim

        # 共享的Action编码器
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        # Q1网络: [256 + 64] → 1
        self.q1_head = nn.Sequential(
            nn.Linear(feature_dim + 64, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # Q2网络: [256 + 64] → 1
        self.q2_head = nn.Sequential(
            nn.Linear(feature_dim + 64, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # 初始化
        self._init_weights()

        # 🔧 修复：Q-heads 使用与 feature_extractor 相同的 dtype（ZeRO-3 需要 bf16）
        self._sync_dtype()

    def _init_weights(self):
        """权重初始化"""
        for module in [self.action_encoder, self.q1_head, self.q2_head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def _sync_dtype(self):
        """同步所有层的dtype与feature_extractor（ZeRO-3 兼容）"""
        if hasattr(self.feature_extractor, 'compute_dtype'):
            target_dtype = self.feature_extractor.compute_dtype
            self.action_encoder = self.action_encoder.to(dtype=target_dtype)
            self.q1_head = self.q1_head.to(dtype=target_dtype)
            self.q2_head = self.q2_head.to(dtype=target_dtype)
            rank0_print(f"[Critic] Q-heads dtype synced to {target_dtype}")

        # 🔧 ZeRO-3 兼容性修复
        if get_zero_stage() == 3:
            fixed = fix_model_for_deepspeed(self, verbose=False, name="Critic")
            if fixed > 0:
                rank0_print(f"[Critic] ✓ ZeRO-3 fix applied to {fixed} modules")

    def forward(self, images, action, prompt=None, history_images=None):
        """
        Args:
            images: PIL.Image list 或 [B, 3, H, W] tensor (当前帧)
                    或 List[(PIL.Image, List[PIL.Image], float, float)] (VLMReplayBuffer格式)
            action: [B, action_dim] 动作参数
            prompt: str, 可选
            history_images: [B, num_frames, 3, H, W] tensor (历史帧，可选)

        Returns:
            q1, q2: [B, 1] 两个Q值估计
        """
        # Handle VLMReplayBuffer format: List[(img, history_imgs, linear_vel, angular_vel)]
        if isinstance(images, list) and len(images) > 0 and isinstance(images[0], tuple):
            # Unpack tuple format
            imgs = [item[0] for item in images]
            hists = [item[1] for item in images]
            linear_vels = [item[2] if len(item) > 2 else 0.0 for item in images]
            angular_vels = [item[3] if len(item) > 3 else 0.0 for item in images]

            vlm_feat = self.feature_extractor(
                imgs, prompt, hists,
                linear_vels=linear_vels,
                angular_vels=angular_vels,
                algorithm=self.feature_extractor.algorithm  # 从feature_extractor获取
            )
        else:
            # Normal format: separate images and history_images
            vlm_feat = self.feature_extractor(images, prompt, history_images)

        # 关键: detach阻止Critic的梯度回传到VLM，保护Actor的更新
        if self.detach_features:
            vlm_feat = vlm_feat.detach()

        # 转换dtype以匹配各层 (DeepSpeed bf16模式下是bf16)
        # action需要匹配action_encoder的dtype
        ae_dtype = self.action_encoder[0].weight.dtype
        action = action.to(dtype=ae_dtype)
        action_feat = self.action_encoder(action)

        # vlm_feat和action_feat需要匹配q_head的dtype
        q_head_dtype = self.q1_head[0].weight.dtype
        vlm_feat = vlm_feat.to(dtype=q_head_dtype)
        action_feat = action_feat.to(dtype=q_head_dtype)

        q_input = torch.cat([vlm_feat, action_feat], dim=-1)

        # 两个独立的Q-head (TD3的Twin Critic)
        q1 = self.q1_head(q_input)
        q2 = self.q2_head(q_input)

        return q1, q2

    def Q1(self, images, action, prompt=None, history_images=None):
        """
        只计算Q1 (用于actor更新时的梯度计算)

        Args:
            images: PIL.Image list 或 [B, 3, H, W] tensor (当前帧)
                    或 List[(PIL.Image, List[PIL.Image], float, float)] (VLMReplayBuffer格式)
            action: [B, action_dim]
            prompt: str, 可选
            history_images: [B, num_frames, 3, H, W] tensor (历史帧，可选)

        Returns:
            q1: [B, 1]
        """
        # Handle VLMReplayBuffer format: List[(img, history_imgs, linear_vel, angular_vel)]
        if isinstance(images, list) and len(images) > 0 and isinstance(images[0], tuple):
            # Unpack tuple format
            imgs = [item[0] for item in images]
            hists = [item[1] for item in images]
            linear_vels = [item[2] if len(item) > 2 else 0.0 for item in images]
            angular_vels = [item[3] if len(item) > 3 else 0.0 for item in images]

            # 获取算法类型 (从feature_extractor或默认DWA)
            algorithm = getattr(self.feature_extractor, 'algorithm', 'DWA')

            vlm_feat = self.feature_extractor(
                imgs, prompt, hists,
                linear_vels=linear_vels,
                angular_vels=angular_vels,
                algorithm=algorithm
            )
        else:
            # Normal format: separate images and history_images
            vlm_feat = self.feature_extractor(images, prompt, history_images)

        # 关键: detach阻止Critic的梯度回传到VLM，保护Actor的更新
        if self.detach_features:
            vlm_feat = vlm_feat.detach()

        # 转换dtype以匹配各层 (DeepSpeed bf16模式下是bf16)
        # action需要匹配action_encoder的dtype
        ae_dtype = self.action_encoder[0].weight.dtype
        action = action.to(dtype=ae_dtype)
        action_feat = self.action_encoder(action)

        # vlm_feat和action_feat需要匹配q_head的dtype
        q_head_dtype = self.q1_head[0].weight.dtype
        vlm_feat = vlm_feat.to(dtype=q_head_dtype)
        action_feat = action_feat.to(dtype=q_head_dtype)

        q1_input = torch.cat([vlm_feat, action_feat], dim=-1)
        return self.q1_head(q1_input)
