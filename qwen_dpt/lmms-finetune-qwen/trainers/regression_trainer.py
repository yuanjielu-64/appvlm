import torch
import torch.nn as nn
from typing import Dict, Optional, Any
from transformers import Trainer
from transformers.trainer import *
import numpy as np
from collections import deque


class RegressionTrainer(Trainer):

    def __init__(
        self,
        param_mean: Optional[np.ndarray] = None,
        param_std: Optional[np.ndarray] = None,
        planner: Optional[str] = None,
        metric_window_size: int = 100,
        history_config: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.param_mean = param_mean
        self.param_std = param_std
        self.metric_window_size = metric_window_size
        self.history_config = history_config

        if param_mean is not None:
            self.param_mean_tensor = torch.tensor(param_mean, dtype=torch.float32)
            self.param_std_tensor = torch.tensor(param_std, dtype=torch.float32)
        else:
            self.param_mean_tensor = None
            self.param_std_tensor = None

        # Get parameter names for clearer metric logging
        self.param_names = None
        if planner is not None:
            try:
                from planner_configs import get_param_names
                self.param_names = get_param_names(planner)
            except Exception as e:
                print(f"Warning: Could not load parameter names for planner '{planner}': {e}")
                self.param_names = None

        # Sliding window containers for metrics
        self.train_loss_window = deque(maxlen=metric_window_size)
        self.train_mae_window = deque(maxlen=metric_window_size)
        self.train_rmse_window = deque(maxlen=metric_window_size)
        self.train_r2_window = deque(maxlen=metric_window_size)

        self.eval_loss_window = deque(maxlen=metric_window_size)
        self.eval_mae_window = deque(maxlen=metric_window_size)
        self.eval_rmse_window = deque(maxlen=metric_window_size)
        self.eval_r2_window = deque(maxlen=metric_window_size)

        # Per-parameter sliding windows (will be initialized when we know number of params)
        self.train_mae_per_param_windows = None
        self.eval_mae_per_param_windows = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        计算loss并记录指标
        - xxx_per_batch: 当前batch的值（会有波动）
        - xxx_per_100: 最近100个batch的平均值（平滑曲线）
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        is_eval = not model.training

        # === 1. 记录当前batch的loss ===
        current_loss = loss.detach().item()
        if is_eval:
            self.eval_loss_window.append(current_loss)
        else:
            self.train_loss_window.append(current_loss)

        # === 2. 检查是否需要记录指标 ===
        current_step = getattr(self.state, 'global_step', 0)
        should_log = is_eval or (current_step > 0 and current_step % 10 == 0)

        if should_log and hasattr(outputs, 'predictions') and outputs.predictions is not None:
            with torch.no_grad():
                predictions = outputs.predictions.detach()

                # 反归一化（恢复到原始参数范围）
                if self.param_mean_tensor is not None:
                    pred_denorm = predictions * (self.param_std_tensor.to(predictions.device) + 1e-8) + \
                                  self.param_mean_tensor.to(predictions.device)
                    label_denorm = labels * (self.param_std_tensor.to(labels.device) + 1e-8) + \
                                   self.param_mean_tensor.to(labels.device)
                else:
                    pred_denorm = predictions
                    label_denorm = labels

                pred_np = pred_denorm.cpu().numpy()
                label_np = label_denorm.cpu().numpy()

                # === 3. 计算当前batch的指标 ===
                current_mae_per_param = np.abs(pred_np - label_np).mean(axis=0)  # 每个参数的MAE
                current_rmse_per_param = np.sqrt(((pred_np - label_np) ** 2).mean(axis=0))
                current_mae_overall = float(current_mae_per_param.mean())  # 所有参数的平均MAE
                current_rmse_overall = float(current_rmse_per_param.mean())

                # 计算R²
                ss_res = ((pred_np - label_np) ** 2).sum()
                ss_tot = ((label_np - label_np.mean(axis=0)) ** 2).sum()
                current_r2 = float(1 - (ss_res / (ss_tot + 1e-8)))

                # === 4. 初始化每个参数的滑动窗口（第一次调用时） ===
                num_params = len(current_mae_per_param)
                if self.train_mae_per_param_windows is None:
                    self.train_mae_per_param_windows = [deque(maxlen=self.metric_window_size) for _ in range(num_params)]
                    self.eval_mae_per_param_windows = [deque(maxlen=self.metric_window_size) for _ in range(num_params)]

                # === 5. 将当前batch的指标加入滑动窗口 ===
                if is_eval:
                    self.eval_mae_window.append(current_mae_overall)
                    self.eval_rmse_window.append(current_rmse_overall)
                    self.eval_r2_window.append(current_r2)
                    for i, mae in enumerate(current_mae_per_param):
                        self.eval_mae_per_param_windows[i].append(float(mae))
                else:
                    self.train_mae_window.append(current_mae_overall)
                    self.train_rmse_window.append(current_rmse_overall)
                    self.train_r2_window.append(current_r2)
                    for i, mae in enumerate(current_mae_per_param):
                        self.train_mae_per_param_windows[i].append(float(mae))

                # === 6. 准备TensorBoard日志 ===
                prefix = "eval" if is_eval else "train"
                metrics = {}

                # 6.1 当前batch的整体指标（会波动）
                metrics[f"{prefix}/loss_per_batch"] = current_loss
                metrics[f"{prefix}/mae_per_batch"] = current_mae_overall
                metrics[f"{prefix}/rmse_per_batch"] = current_rmse_overall
                metrics[f"{prefix}/r2_per_batch"] = current_r2

                # 6.2 当前batch的每个参数MAE（会波动）
                for i, mae in enumerate(current_mae_per_param):
                    param_name = self.param_names[i] if self.param_names else f"param_{i}"
                    metrics[f"{prefix}/{param_name}_mae_per_batch"] = float(mae)

                # 6.3 最近N个batch的平滑指标（平滑曲线）
                windows = {
                    'loss': self.eval_loss_window if is_eval else self.train_loss_window,
                    'mae': self.eval_mae_window if is_eval else self.train_mae_window,
                    'rmse': self.eval_rmse_window if is_eval else self.train_rmse_window,
                    'r2': self.eval_r2_window if is_eval else self.train_r2_window
                }

                for metric_name, window in windows.items():
                    if len(window) > 0:
                        metrics[f"{prefix}/{metric_name}_per_{self.metric_window_size}"] = float(np.mean(window))

                # 6.4 每个参数的平滑MAE
                per_param_windows = self.eval_mae_per_param_windows if is_eval else self.train_mae_per_param_windows
                for i in range(num_params):
                    if len(per_param_windows[i]) > 0:
                        param_name = self.param_names[i] if self.param_names else f"param_{i}"
                        metrics[f"{prefix}/{param_name}_mae_per_{self.metric_window_size}"] = float(np.mean(per_param_windows[i]))

                self.log(metrics)

        return (loss, outputs) if return_outputs else loss

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """
        保存 LoRA adapter + Regression head (PEFT 兼容格式)
        """
        import os
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        model_to_save = self.model

        # 1. 保存 PEFT adapter (LoRA weights)
        if hasattr(model_to_save, 'base_model'):
            base_model = model_to_save.base_model
            if hasattr(base_model, 'save_pretrained'):  # PEFT model
                print(f"Saving LoRA adapter to {output_dir}")

                # 🔧 修复 DeepSpeed ZeRO-3 权重收集问题
                # 使用自定义的 get_peft_state_maybe_zero_3 函数
                if self.is_deepspeed_enabled:
                    print("[FIX] DeepSpeed detected, using get_peft_state_maybe_zero_3...")
                    from utils import get_peft_state_maybe_zero_3

                    if self.args.local_rank == 0 or self.args.local_rank == -1:
                        # 只收集LoRA参数（不收集冻结的VLM）
                        state_dict = get_peft_state_maybe_zero_3(
                            base_model.named_parameters(),
                            bias="none"
                        )
                        base_model.save_pretrained(output_dir, state_dict=state_dict)

                        # 统计保存的LoRA参数
                        num_lora_params = len(state_dict)
                        total_lora_values = sum(p.numel() for p in state_dict.values())
                        print(f"✓ Saved LoRA adapter:")
                        print(f"  - {num_lora_params} parameter tensors")
                        print(f"  - {total_lora_values:,} total values")
                        if total_lora_values == 0:
                            print(f"  ⚠️  WARNING: All LoRA parameters are empty! This likely means the save failed.")
                else:
                    # 非 DeepSpeed 模式正常保存
                    base_model.save_pretrained(output_dir)
                    print(f"✓ Saved LoRA adapter (non-DeepSpeed mode)")

        # 2. 保存 Regression head (单独保存)
        if hasattr(model_to_save, 'regression_head'):
            regression_head_dir = os.path.join(output_dir, 'regression_head')
            os.makedirs(regression_head_dir, exist_ok=True)
            regression_head_path = os.path.join(regression_head_dir, 'pytorch_model.bin')
            print(f"Saving regression head to {regression_head_path}")

            # 🔧 修复 DeepSpeed ZeRO-3 权重收集问题
            if self.is_deepspeed_enabled:
                print("[FIX] DeepSpeed detected, gathering regression head weights...")
                import deepspeed

                # DeepSpeed ZeRO-3 需要收集所有分片
                with deepspeed.zero.GatheredParameters(list(model_to_save.regression_head.parameters()), modifier_rank=0):
                    if self.args.local_rank == 0 or self.args.local_rank == -1:
                        # 只在 rank 0 保存
                        state_dict = model_to_save.regression_head.state_dict()

                        # 🔧 FIX: 克隆每个tensor，避免保存共享的大buffer
                        # ZeRO-3的GatheredParameters会创建一个包含整个模型的大buffer
                        # state_dict中的tensor都是这个buffer的view，直接保存会保存整个buffer
                        # 克隆可以创建独立的tensor，只保存实际需要的参数
                        state_dict_clean = {k: v.clone() for k, v in state_dict.items()}
                        torch.save(state_dict_clean, regression_head_path)

                        # 统计DPT Head参数
                        num_dpt_params = len(state_dict_clean)
                        total_dpt_values = sum(p.numel() for p in state_dict_clean.values())
                        file_size_mb = os.path.getsize(regression_head_path) / 1024 / 1024
                        print(f"✓ Saved DPT Head:")
                        print(f"  - {num_dpt_params} parameter tensors")
                        print(f"  - {total_dpt_values:,} total values")
                        print(f"  - File size: {file_size_mb:.2f} MB")
            else:
                # 非 DeepSpeed 模式正常保存
                state_dict = model_to_save.regression_head.state_dict()
                torch.save(state_dict, regression_head_path)

                # 统计DPT Head参数
                num_dpt_params = len(state_dict)
                total_dpt_values = sum(p.numel() for p in state_dict.values())
                file_size_mb = os.path.getsize(regression_head_path) / 1024 / 1024
                print(f"✓ Saved DPT Head:")
                print(f"  - {num_dpt_params} parameter tensors")
                print(f"  - {total_dpt_values:,} total values")
                print(f"  - File size: {file_size_mb:.2f} MB")

        # 3. 保存归一化参数 (param_mean, param_std)
        if self.param_mean is not None and self.param_std is not None:
            import numpy as np
            normalization_dir = os.path.join(output_dir, 'normalization')
            os.makedirs(normalization_dir, exist_ok=True)

            mean_path = os.path.join(normalization_dir, 'param_mean.npy')
            std_path = os.path.join(normalization_dir, 'param_std.npy')

            np.save(mean_path, self.param_mean)
            np.save(std_path, self.param_std)
            print(f"Saving normalization stats to {normalization_dir}")
            print(f"  param_mean: {self.param_mean}")
            print(f"  param_std: {self.param_std}")

        # 4. 保存历史 encoder (history_encoder)
        if hasattr(model_to_save, 'history_encoder') and model_to_save.history_encoder is not None:
            history_encoder_dir = os.path.join(output_dir, 'history_encoder')
            os.makedirs(history_encoder_dir, exist_ok=True)
            history_encoder_path = os.path.join(history_encoder_dir, 'pytorch_model.bin')
            print(f"Saving history encoder to {history_encoder_path}")

            # 🔧 修复 DeepSpeed ZeRO-3 权重收集问题
            if self.is_deepspeed_enabled:
                print("[FIX] DeepSpeed detected, gathering history encoder weights...")
                import deepspeed

                # DeepSpeed ZeRO-3 需要收集所有分片
                with deepspeed.zero.GatheredParameters(list(model_to_save.history_encoder.parameters()), modifier_rank=0):
                    if self.args.local_rank == 0 or self.args.local_rank == -1:
                        # 只在 rank 0 保存
                        state_dict = model_to_save.history_encoder.state_dict()

                        # 🔧 FIX: 克隆每个tensor，避免保存共享的大buffer
                        # ZeRO-3的GatheredParameters会创建一个包含整个模型的大buffer
                        # state_dict中的tensor都是这个buffer的view，直接保存会保存整个buffer
                        # 克隆可以创建独立的tensor，只保存实际需要的参数
                        state_dict_clean = {k: v.clone() for k, v in state_dict.items()}
                        torch.save(state_dict_clean, history_encoder_path)

                        # 统计History Encoder参数
                        num_history_params = len(state_dict_clean)
                        total_history_values = sum(p.numel() for p in state_dict_clean.values())
                        file_size_mb = os.path.getsize(history_encoder_path) / 1024 / 1024
                        print(f"✓ Saved History Encoder:")
                        print(f"  - {num_history_params} parameter tensors")
                        print(f"  - {total_history_values:,} total values")
                        print(f"  - File size: {file_size_mb:.2f} MB")
            else:
                # 非 DeepSpeed 模式正常保存
                state_dict = model_to_save.history_encoder.state_dict()
                torch.save(state_dict, history_encoder_path)

                # 统计History Encoder参数
                num_history_params = len(state_dict)
                total_history_values = sum(p.numel() for p in state_dict.values())
                file_size_mb = os.path.getsize(history_encoder_path) / 1024 / 1024
                print(f"✓ Saved History Encoder:")
                print(f"  - {num_history_params} parameter tensors")
                print(f"  - {total_history_values:,} total values")
                print(f"  - File size: {file_size_mb:.2f} MB")

        # 5. 保存历史 encoder 配置 (history_config)
        if self.history_config is not None:
            import json
            history_config_path = os.path.join(output_dir, 'history_config.json')
            with open(history_config_path, 'w') as f:
                json.dump(self.history_config, f, indent=2)
            print(f"Saving history config to {history_config_path}")
            print(f"  config: {self.history_config}")

        # 6. 保存 trainer state (由父类处理)
        # 不调用 super()._save()，因为它会保存整个模型
