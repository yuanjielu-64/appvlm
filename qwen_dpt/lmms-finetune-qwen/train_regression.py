import os
os.environ["WANDB_PROJECT"]= "vlm-nav-regression"
from dataclasses import asdict
import math
from pathlib import Path
from typing import List, Optional
import yaml

from accelerate.utils import DistributedType
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
import transformers
from transformers.integrations import deepspeed
from transformers.trainer_callback import TrainerCallback

from arguments import ModelArguments, DataArguments, TrainingArguments, LoraArguments
from collators import COLLATORS
from datasets import LazySupervisedDataset, RegressionDataset
from loaders import LOADERS
from supported_models import MODULE_KEYWORDS
from utils import (
    rank0_print, find_all_linear_names, get_peft_state_maybe_zero_3
)
from trainers import RegressionTrainer


class SaveInitialCheckpointCallback(TrainerCallback):
    """在第一个训练步骤后保存 checkpoint-1"""

    def __init__(self):
        self.saved = False

    def on_step_end(self, args, state, control, **kwargs):
        # 只在 step 1 保存一次
        if state.global_step == 1 and not self.saved:
            rank0_print(f"[Callback] Triggering save at step 1...")
            control.should_save = True  # 触发Trainer的保存机制
            self.saved = True
        return control

def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()

    # dumping arguments
    output_dir = getattr(training_args, 'output_dir', None)
    assert output_dir is not None, "output_dir is required"
    args_dir = Path(output_dir) / "arguments"
    args_dir.mkdir(parents=True, exist_ok=True)
    yaml.dump(asdict(model_args), open(args_dir / "model.yaml", "w"))
    yaml.dump(asdict(data_args), open(args_dir / "data.yaml", "w"))
    yaml.dump(asdict(training_args), open(args_dir / "training.yaml", "w"))
    yaml.dump(asdict(lora_args), open(args_dir / "lora.yaml", "w"))

    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    if getattr(training_args, 'deepspeed', None) and getattr(lora_args, 'q_lora', False):
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    device_map = None
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if int(os.environ.get("WORLD_SIZE", 1)) != 1 else None
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            raise ValueError("FSDP or ZeRO3 are not incompatible with QLoRA.")

    # llm quantization config (for q-lora)
    bnb_config = None
    if lora_args.use_lora and lora_args.q_lora:
        from transformers import BitsAndBytesConfig
        rank0_print("Quantization for LLM enabled...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
        )

    # load model, tokenizer, processor
    rank0_print("Loading model, tokenizer, processor...")
    loader_kwargs = {
        "model_hf_path": model_args.model_hf_path,
        "model_local_path": model_args.model_local_path,
        "compute_dtype": compute_dtype,
        "bnb_config": bnb_config,
        "use_flash_attn": training_args.use_flash_attn,
        "device_map": device_map,
    }

    # 为回归模型传递 num_params 和 head_type
    if model_args.model_family_id == "qwen2.5-vl-regression":
        loader_kwargs["num_params"] = model_args.num_params
        loader_kwargs["head_type"] = model_args.head_type

        # 只在 use_history=True 且 num_history_frames > 0 时启用 history encoder
        use_history = getattr(model_args, 'use_history', False)
        num_history_frames = getattr(model_args, 'num_history_frames', 0)
        if use_history and num_history_frames > 0:
            loader_kwargs["use_history"] = True
            loader_kwargs["num_history_frames"] = num_history_frames
            loader_kwargs["history_dim"] = model_args.history_dim
        else:
            loader_kwargs["use_history"] = False
            loader_kwargs["num_history_frames"] = 0

        # 加载参数权重
        if model_args.planner:
            from planner_configs import get_param_weights
            param_weights = get_param_weights(model_args.planner)
            if param_weights:
                loader_kwargs["param_weights"] = param_weights
                rank0_print(f"Loaded param weights for {model_args.planner}: {param_weights}")

        rank0_print(f"Regression model: num_params={model_args.num_params}, head_type={model_args.head_type}")
        if loader_kwargs["use_history"]:
            rank0_print(f"  History encoder enabled: {loader_kwargs['num_history_frames']} frames, dim={loader_kwargs['history_dim']}")
        else:
            rank0_print(f"  History encoder disabled (num_history_frames={num_history_frames})")

    loader = LOADERS[model_args.model_family_id](**loader_kwargs)

    model, tokenizer, processor, config = loader.load()
    tokenizer.model_max_length = training_args.model_max_length

    if training_args.gradient_checkpointing:
        model.base_model.enable_input_require_grads()

    # freeze certain params
    vision_encoder_keys = MODULE_KEYWORDS[model_args.model_family_id]["vision_encoder"]
    if not training_args.train_vision_encoder:
        rank0_print(f"Vision encoder is freezed... including:")
        for module in vision_encoder_keys:
            rank0_print(f"\t{module}")
            eval(f"model.{module}").requires_grad_(False)

    vision_projector_keys = MODULE_KEYWORDS[model_args.model_family_id]["vision_projector"]
    if not training_args.train_vision_projector:
        rank0_print(f"Vision projector is freezed... including:")
        for module in vision_projector_keys:
            rank0_print(f"\t{module}")
            eval(f"model.{module}").requires_grad_(False)

    # other components preparation (e.g., image_newline, vision_resampler)
    # we will just freeze these
    if "others" in MODULE_KEYWORDS[model_args.model_family_id]:
        rank0_print(f"Other multimodal component is freezed... including:")
        for other_key in MODULE_KEYWORDS[model_args.model_family_id]["others"]:
            rank0_print(f"\t{other_key}")
            eval(f"model.{other_key}").requires_grad_(False)

    # lora preparation (apply to base_model only, DPT head is always fully trained)
    llm_keys = MODULE_KEYWORDS[model_args.model_family_id]["llm"]
    # LoRA 用的 keys（不带 base_model. 前缀）
    llm_lora_keys = MODULE_KEYWORDS[model_args.model_family_id].get("llm_lora", llm_keys)

    print("\n" + "="*80, flush=True)
    print(f"🔧 LoRA Configuration:", flush=True)
    print(f"   use_lora = {lora_args.use_lora}", flush=True)
    print(f"   q_lora = {lora_args.q_lora}", flush=True)
    print(f"   lora_r = {lora_args.lora_r}", flush=True)
    print(f"   lora_alpha = {lora_args.lora_alpha}", flush=True)
    print("="*80 + "\n", flush=True)

    if not (lora_args.use_lora or (training_args.train_vision_encoder and lora_args.use_vision_lora)):
        print("❌ No LoRA enabled...", flush=True)
        rank0_print("No LoRA enabled...")
    else:
        named_modules = {n: m for n, m in model.base_model.named_modules()}
        lora_modules = []
        full_modules = []

        if training_args.train_vision_encoder and lora_args.use_vision_lora:
            print("✅ LoRA for vision encoder enabled...", flush=True)
            rank0_print("LoRA for vision encoder enabled...")
            # Vision encoder keys 也需要去掉 base_model. 前缀
            vision_lora_keys = [k.replace("base_model.", "") for k in vision_encoder_keys]
            lora_modules.extend(find_all_linear_names(named_modules, vision_lora_keys))
        elif training_args.train_vision_encoder:
            rank0_print("Vision encoder will be fully trained...")
            full_modules.extend(vision_encoder_keys)

        if lora_args.use_lora:
            print("✅ LoRA for LLM enabled...", flush=True)
            rank0_print("LoRA for LLM enabled...")
            lora_modules.extend(find_all_linear_names(named_modules, llm_lora_keys))
        else:
            rank0_print("LLM will be fully trained...")
            full_modules.extend(llm_keys)

        if training_args.train_vision_projector:
            rank0_print("Vision projector will be fully trained...")
            full_modules.extend(vision_projector_keys)

        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_modules,
            modules_to_save=full_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
        )

        if lora_args.q_lora:
            model.base_model = prepare_model_for_kbit_training(
                model.base_model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

        model.base_model = get_peft_model(model.base_model, lora_config)

    # print trainable parameters for inspection
    print("\n" + "="*80, flush=True)
    print("🎯 Trainable Parameters:", flush=True)
    trainable_params = 0
    all_params = 0
    for name, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(f"   Total params: {all_params:,}", flush=True)
    print(f"   Trainable params: {trainable_params:,}", flush=True)
    print(f"   Trainable %: {100 * trainable_params / all_params:.2f}%", flush=True)
    print("="*80 + "\n", flush=True)

    rank0_print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            rank0_print(f"\t{name}")

    # Load checkpoint weights if resuming (must be done BEFORE creating Trainer)
    if training_args.resume_from_checkpoint is not None:
        checkpoint_path = training_args.resume_from_checkpoint
        rank0_print(f"\n{'='*80}")
        rank0_print(f"Loading checkpoint from: {checkpoint_path}")
        rank0_print(f"{'='*80}\n")

        # 1. Load DPT Head weights
        regression_head_path = os.path.join(checkpoint_path, 'regression_head', 'pytorch_model.bin')
        if os.path.exists(regression_head_path):
            rank0_print(f"✓ Loading regression head from {regression_head_path}")
            state_dict = torch.load(regression_head_path, map_location='cpu')
            model.regression_head.load_state_dict(state_dict, strict=False)
        else:
            rank0_print(f"⚠ Warning: regression head not found at {regression_head_path}")

        # 2. Load History Encoder weights (if using history)
        if hasattr(model, 'history_encoder') and model.history_encoder is not None:
            history_encoder_path = os.path.join(checkpoint_path, 'history_encoder', 'pytorch_model.bin')
            if os.path.exists(history_encoder_path):
                rank0_print(f"✓ Loading history encoder from {history_encoder_path}")
                state_dict = torch.load(history_encoder_path, map_location='cpu')
                model.history_encoder.load_state_dict(state_dict, strict=False)
            else:
                rank0_print(f"⚠ Warning: history encoder not found at {history_encoder_path}")

        # 3. Verify history config matches
        history_config_path = os.path.join(checkpoint_path, 'history_config.json')
        current_use_history = getattr(model_args, 'use_history', False)
        current_num_frames = getattr(model_args, 'num_history_frames', 0)

        if os.path.exists(history_config_path):
            import json
            with open(history_config_path, 'r') as f:
                saved_history_config = json.load(f)

            rank0_print(f"✓ Found history_config.json in checkpoint")
            rank0_print(f"  Checkpoint config: {saved_history_config}")

            saved_use_history = saved_history_config.get('use_history', False)
            saved_num_frames = saved_history_config.get('num_history_frames', 0)
            saved_history_dim = saved_history_config.get('history_dim', 256)

            # 验证配置一致性
            config_mismatch = False
            if saved_use_history != current_use_history:
                rank0_print(f"⚠ WARNING: use_history mismatch!")
                rank0_print(f"  Checkpoint: {saved_use_history}")
                rank0_print(f"  Current: {current_use_history}")
                config_mismatch = True

            if saved_num_frames != current_num_frames:
                rank0_print(f"⚠ WARNING: num_history_frames mismatch!")
                rank0_print(f"  Checkpoint: {saved_num_frames}")
                rank0_print(f"  Current: {current_num_frames}")
                config_mismatch = True

            if saved_use_history and current_use_history:
                current_history_dim = getattr(model_args, 'history_dim', 256)
                if saved_history_dim != current_history_dim:
                    rank0_print(f"⚠ WARNING: history_dim mismatch!")
                    rank0_print(f"  Checkpoint: {saved_history_dim}")
                    rank0_print(f"  Current: {current_history_dim}")
                    config_mismatch = True

            if config_mismatch:
                rank0_print(f"❌ ERROR: Configuration mismatch detected!")
                rank0_print(f"   Please ensure training arguments match the checkpoint.")
                raise ValueError("History configuration mismatch between checkpoint and current settings")
            else:
                rank0_print(f"✓ History configuration matches checkpoint")
        else:
            # Checkpoint中没有history_config.json
            rank0_print(f"⚠ Warning: history_config.json not found in checkpoint")

            # 推断checkpoint的use_history状态
            history_encoder_exists = os.path.exists(os.path.join(checkpoint_path, 'history_encoder', 'pytorch_model.bin'))

            if history_encoder_exists:
                rank0_print(f"  → Detected history_encoder weights in checkpoint (checkpoint was trained with use_history=True)")
                if not current_use_history:
                    rank0_print(f"❌ ERROR: Checkpoint has history_encoder but current use_history=False")
                    raise ValueError("Cannot load checkpoint with history_encoder when use_history=False")
            else:
                rank0_print(f"  → No history_encoder weights found (checkpoint was trained with use_history=False)")
                if current_use_history:
                    rank0_print(f"❌ ERROR: Checkpoint has no history_encoder but current use_history=True")
                    raise ValueError("Cannot load checkpoint without history_encoder when use_history=True")

        rank0_print(f"{'='*80}\n")

    # load data (use RegressionDataset for regression tasks)
    rank0_print("Loading data...")
    train_dataset = RegressionDataset(
        data_path=data_args.data_path,
        image_folder=data_args.image_folder,
        video_folder=data_args.video_folder,
        num_frames=data_args.num_frames,
        model_family_id=model_args.model_family_id,
        user_key=data_args.user_key,
        assistant_key=data_args.assistant_key
    )

    if data_args.eval_data_path:
        eval_dataset = RegressionDataset(
            data_path=data_args.eval_data_path,
            image_folder=data_args.image_folder,
            video_folder=data_args.video_folder,
            num_frames=data_args.num_frames,
            model_family_id=model_args.model_family_id,
            user_key=data_args.user_key,
            assistant_key=data_args.assistant_key
        )

        max_eval_samples = getattr(training_args, 'max_eval_samples', None)
        if max_eval_samples is not None and max_eval_samples > 0:
            original_len = len(eval_dataset)
            eval_dataset.list_data_dict = eval_dataset.list_data_dict[:max_eval_samples]
            rank0_print(f"Limited eval dataset from {original_len} to {len(eval_dataset)} samples")
    else:
        eval_dataset = None
        training_args.eval_strategy = "no"

    # 计算或加载归一化统计量
    import numpy as np
    param_mean = None
    param_std = None

    # 如果从checkpoint恢复，优先从checkpoint加载归一化参数
    if training_args.resume_from_checkpoint is not None:
        checkpoint_path = training_args.resume_from_checkpoint
        normalization_dir = os.path.join(checkpoint_path, 'normalization')
        mean_path = os.path.join(normalization_dir, 'param_mean.npy')
        std_path = os.path.join(normalization_dir, 'param_std.npy')

        if os.path.exists(mean_path) and os.path.exists(std_path):
            rank0_print(f"✓ Loading normalization stats from checkpoint: {normalization_dir}")
            param_mean = np.load(mean_path)
            param_std = np.load(std_path)
            rank0_print("Loaded normalization stats:")
            if model_args.planner:
                from planner_configs import get_param_names
                param_names = get_param_names(model_args.planner)
                for i, name in enumerate(param_names):
                    rank0_print(f"  {name:<25} mean={param_mean[i]:>8.4f}, std={param_std[i]:>8.4f}")
        else:
            rank0_print(f"⚠ Warning: normalization stats not found in checkpoint, will recompute")

    # 如果没有从checkpoint加载，则重新计算
    if param_mean is None or param_std is None:
        rank0_print("Computing normalization statistics from training data...")
        param_mean, param_std = train_dataset.compute_normalization_stats(max_samples=200000)
        rank0_print("Computed normalization stats:")
        if model_args.planner:
            from planner_configs import get_param_names
            param_names = get_param_names(model_args.planner)
            for i, name in enumerate(param_names):
                rank0_print(f"  {name:<25} mean={param_mean[i]:>8.4f}, std={param_std[i]:>8.4f}")

    # data collator
    collator_use_history = use_history and num_history_frames > 0
    data_collator = COLLATORS[model_args.model_family_id](
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        mask_question_tokens=False,  # regression task does not need masking
        param_mean=param_mean,
        param_std=param_std,
        normalize=True,
        planner=model_args.planner,
        use_history=collator_use_history,
        num_history_frames=num_history_frames if collator_use_history else 0,
        history_image_size=getattr(model_args, 'history_image_size', None) if collator_use_history else None,
        label_noise_std=data_args.label_noise_std,
    )

    if data_args.label_noise_std > 0:
        rank0_print(f"Label noise enabled: std={data_args.label_noise_std}")

    # 准备历史encoder配置（用于保存）
    history_config = None
    use_history = getattr(model_args, 'use_history', False)
    num_history_frames = getattr(model_args, 'num_history_frames', 0)
    if use_history and num_history_frames > 0:
        history_config = {
            "use_history": True,
            "num_history_frames": num_history_frames,
            "history_dim": model_args.history_dim,
            "history_image_size": getattr(model_args, 'history_image_size', None),
        }

    # trainer (use RegressionTrainer for MSE loss and regression metrics)
    # 注意：不需要手动传 compute_metrics，RegressionTrainer 的 compute_metrics 方法会自动被调用
    trainer = RegressionTrainer(
        param_mean=param_mean,
        param_std=param_std,
        planner=model_args.planner,  # Pass planner name for parameter naming
        history_config=history_config,  # Pass history encoder config for saving
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Add callback to save checkpoint at step 1 (only for new training, not resume)
    if training_args.resume_from_checkpoint is None:
        save_initial_callback = SaveInitialCheckpointCallback()
        trainer.add_callback(save_initial_callback)
        rank0_print("Added callback to save checkpoint at step 1")

    # Resume from checkpoint if specified
    if training_args.resume_from_checkpoint is not None:
        rank0_print(f"Resuming training from checkpoint: {training_args.resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
        trainer.train()
    trainer.save_state()

    # save model (处理共享 tensor 的问题)
    rank0_print("Saving model...")

    # 解决 Qwen2.5-VL 的 embed_tokens 和 lm_head 共享内存问题
    # 方法：先解绑共享的权重，保存后再恢复绑定
    model_to_save = trainer.model
    if hasattr(model_to_save, 'base_model'):
        base_model = model_to_save.base_model
        if hasattr(base_model, 'model') and hasattr(base_model, 'lm_head'):
            # 保存原始绑定状态
            original_lm_head = base_model.lm_head.weight
            # 创建独立副本
            base_model.lm_head.weight = torch.nn.Parameter(original_lm_head.clone())

            # 保存模型
            trainer.save_model(output_dir)

            # 恢复绑定（节省内存）
            base_model.lm_head.weight = original_lm_head
        else:
            trainer.save_model(output_dir)
    else:
        trainer.save_model(output_dir)


if __name__ == "__main__":
    train()
