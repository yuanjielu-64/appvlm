import copy
import os
import pickle
from os.path import join

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# DDP 工具函数
from rlft.ddp_utils import rank0_print, is_main_process, is_ddp_enabled

# ZeRO 兼容工具（支持 ZeRO-2 和 ZeRO-3）
from rlft.zero_utils import ZeROCompatibleTrainer, is_deepspeed_enabled, get_zero_stage


class Actor(nn.Module):
    def __init__(self, state_preprocess, head, action_dim):
        super(Actor, self).__init__()

        self.state_preprocess = state_preprocess
        self.head = head
        self.fc = nn.Linear(self.head.feature_dim, action_dim)

    def forward(self, state):
        a = self.state_preprocess(state) if self.state_preprocess else state
        a = self.head(a)
        return torch.tanh(self.fc(a))


class Critic(nn.Module):
    def __init__(self, state_preprocess, head):
        super(Critic, self).__init__()

        # Q1 architecture
        self.state_preprocess1 = state_preprocess
        self.head1 = head
        self.fc1 = nn.Linear(self.head1.feature_dim, 1)

        # Q2 architecture
        self.state_preprocess2 = copy.deepcopy(state_preprocess)
        self.head2 = copy.deepcopy(head)
        self.fc2 = nn.Linear(self.head2.feature_dim, 1)

    def forward(self, state, action):
        state1 = self.state_preprocess1(
            state) if self.state_preprocess1 else state
        sa1 = torch.cat([state1, action], 1)

        state2 = self.state_preprocess2(
            state) if self.state_preprocess2 else state
        sa2 = torch.cat([state2, action], 1)

        q1 = self.head1(sa1)
        q1 = self.fc1(q1)

        q2 = self.head2(sa2)
        q2 = self.fc2(q2)
        return q1, q2

    def Q1(self, state, action):
        state = self.state_preprocess1(
            state) if self.state_preprocess1 else state
        sa = torch.cat([state, action], 1)

        q1 = self.head1(sa)
        q1 = self.fc1(q1)
        return q1


class TD3(object):
    @staticmethod
    def _unwrap_model(model):
        """解包 DataParallel，返回实际的模型"""
        return model.module if isinstance(model, torch.nn.DataParallel) else model

    def __init__(
            self,
            actor,
            actor_optim,
            critic,
            critic_optim,
            action_range,
            device="cpu",
            gamma=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            n_step=4,
            update_actor_freq=2,
            exploration_noise=0.1,
            param_mean=None,
            param_std=None
    ):

        self.actor = actor
        # 🔧 ZeRO-3 兼容修复: actor_target 组件作为独立模块
        # 不创建 VLM_DPT_Actor 实例，避免持有 DeepSpeed 管理的 feature_extractor 引用
        # 组件存储为 self.actor_target_fc, self.actor_target_lora_state, actor_target_dpt, actor_target_history
        self._create_actor_target(self._unwrap_model(actor))
        self.actor_optimizer = actor_optim

        self.critic = critic
        # 🔧 ZeRO-3 兼容修复: critic_target 的 Q-heads 作为独立模块
        # 不创建完整的 VLM_DPT_Critic 对象，避免与 DeepSpeed 冲突
        self._create_critic_target(self._unwrap_model(critic))
        self.critic_optimizer = critic_optim

        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.update_actor_freq = update_actor_freq
        self.exploration_noise = exploration_noise
        self.device = device
        self.n_step = n_step

        self.total_it = 0
        self.action_range = action_range
        # 🔧 修复：显式指定 dtype=float32，避免 numpy float64 导致的 dtype 不一致
        self._action_scale = torch.tensor(
            (action_range[1] - action_range[0]) / 2.0, dtype=torch.float32, device=self.device)
        self._action_bias = torch.tensor(
            (action_range[1] + action_range[0]) / 2.0, dtype=torch.float32, device=self.device)

        # 参数归一化统计（用于与监督学习保持一致）
        if param_mean is not None and param_std is not None:
            self.param_mean = torch.tensor(param_mean, dtype=torch.float32, device=self.device)
            self.param_std = torch.tensor(param_std, dtype=torch.float32, device=self.device)
            rank0_print(f"[TD3] Using parameter normalization:")
            rank0_print(f"  Mean: {param_mean}")
            rank0_print(f"  Std:  {param_std}")
        else:
            self.param_mean = None
            self.param_std = None
            rank0_print(f"[TD3] No parameter normalization (using raw action space)")

        # DDP/ZeRO 模式：accelerator 和 trainer 会在 initialize_policy 中设置
        self.accelerator = None
        self.zero_trainer = None  # ZeROCompatibleTrainer 实例

    def _create_actor_target(self, actor):
        """
        创建 actor_target，为可训练组件创建独立副本

        策略:
        - VLM base: 共享（4-bit 量化无法 deepcopy，始终冻结）
        - LoRA: 共享（ZeRO-3 下 set_peft_model_state_dict 会破坏参数管理）
        - DPT: 独立副本（如果可训练，需要 soft update）
        - History: 独立副本（如果可训练，需要 soft update）
        - FC: 独立副本（始终可训练，需要 soft update）

        ZeRO-3 兼容：
        - 在 ZeRO-3 模式下，不能使用 copy.deepcopy（ZeROOrderedDict 不支持）
        - 改用创建新实例 + 加载 state_dict 的方式

        ZeRO-3 兼容性关键：
        - 不创建 VLM_DPT_Actor 实例，避免持有 DeepSpeed 管理的 feature_extractor 引用
        - 不为 LoRA 创建独立副本（set_peft_model_state_dict 会替换参数张量，破坏 DeepSpeed 管理）
        - 只存储独立的 FC/DPT/History 组件
        """
        import copy
        import torch.nn as nn
        from peft import PeftModel

        # 检测 ZeRO-3 模式
        zero_stage = get_zero_stage()

        rank0_print(f"[TD3] Creating actor_target components (ZeRO-3 compatible)...")
        if zero_stage == 3:
            rank0_print(f"  [ZeRO-3] Using state_dict copy, no VLM_DPT_Actor instance")

        # ============ 获取设备和 dtype ============
        # 获取设备（ZeRO-3 兼容）
        if zero_stage == 3:
            # ZeRO-3: 使用 local_rank 确定设备
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            device = f"cuda:{local_rank}"
        else:
            device = actor.fc.weight.device

        # ============ LoRA: 不创建独立副本 ============
        # 为了 ZeRO-3 兼容性，不为 target 网络维护独立的 LoRA。
        # 原因：set_peft_model_state_dict 会替换参数张量，导致 DeepSpeed 的
        # ds_active_sub_modules 属性丢失。在 TD3 中，target 网络的稳定性
        # 主要来自 Q-heads 和 FC 层的慢更新，使用当前 LoRA 是可以接受的。
        self.actor_target_lora_state = None
        rank0_print(f"  [LoRA] Sharing with actor (ZeRO-3 compatible, no independent target)")

        # ============ 为 DPT 创建独立副本 ============
        if hasattr(actor.feature_extractor, 'dpt_head') and actor.feature_extractor.dpt_head is not None:
            dpt_trainable = any(p.requires_grad for p in actor.feature_extractor.dpt_head.parameters())
            if dpt_trainable:
                rank0_print(f"  [DPT] Creating independent copy (trainable)...")

                # 获取原始 DPT
                orig_dpt = actor.feature_extractor.dpt_head

                if zero_stage == 3:
                    # ZeRO-3: 创建新实例并加载 state_dict
                    from qwen2_5_vl_dpt_regression import DPTHead
                    import deepspeed

                    self.actor_target_dpt = DPTHead(
                        hidden_size=orig_dpt.hidden_size if hasattr(orig_dpt, 'hidden_size') else 2048,
                        num_params=orig_dpt.num_params if hasattr(orig_dpt, 'num_params') else 8,
                        feature_dim=orig_dpt.feature_dim if hasattr(orig_dpt, 'feature_dim') else 256,
                        num_layers=len(orig_dpt.projections) if hasattr(orig_dpt, 'projections') else 4,
                        use_history=actor.feature_extractor.use_history,
                        history_dim=orig_dpt.history_dim if hasattr(orig_dpt, 'history_dim') else 256,
                    )

                    # 在 GatheredParameters 中加载 state_dict
                    with deepspeed.zero.GatheredParameters(list(orig_dpt.parameters()), modifier_rank=None):
                        state_dict = {k: v.clone().cpu() for k, v in orig_dpt.state_dict().items()}
                    self.actor_target_dpt.load_state_dict(state_dict, strict=False)
                else:
                    # 普通模式（ZeRO-2 等）：使用 deepcopy
                    self.actor_target_dpt = copy.deepcopy(orig_dpt)

                # 获取原始DPT的dtype（bf16模式下是bf16）
                if zero_stage == 3:
                    import deepspeed
                    with deepspeed.zero.GatheredParameters(list(orig_dpt.parameters()), modifier_rank=None):
                        dpt_dtype = next(orig_dpt.parameters()).dtype
                else:
                    dpt_dtype = next(orig_dpt.parameters()).dtype
                self.actor_target_dpt.to(device=device, dtype=dpt_dtype)
                for param in self.actor_target_dpt.parameters():
                    param.requires_grad = False
                rank0_print(f"  [DPT] ✓ Independent target created (dtype={dpt_dtype})")
            else:
                rank0_print(f"  [DPT] Frozen, sharing with actor")
                self.actor_target_dpt = None
        else:
            self.actor_target_dpt = None

        # ============ 为 History Encoder 创建独立副本 ============
        if hasattr(actor.feature_extractor, 'history_encoder') and actor.feature_extractor.history_encoder is not None:
            history_trainable = any(p.requires_grad for p in actor.feature_extractor.history_encoder.parameters())
            if history_trainable:
                rank0_print(f"  [History] Creating independent copy (trainable)...")

                # 获取原始 History Encoder
                orig_hist = actor.feature_extractor.history_encoder

                if zero_stage == 3:
                    # ZeRO-3: 创建新实例并加载 state_dict
                    from qwen2_5_vl_dpt_regression import HistoryEncoder
                    import deepspeed

                    self.actor_target_history = HistoryEncoder(
                        hidden_dim=orig_hist.hidden_dim if hasattr(orig_hist, 'hidden_dim') else 256,
                        num_frames=orig_hist.num_frames if hasattr(orig_hist, 'num_frames') else 2,
                        num_transformer_layers=2,  # 默认值
                    )

                    with deepspeed.zero.GatheredParameters(list(orig_hist.parameters()), modifier_rank=None):
                        state_dict = {k: v.clone().cpu() for k, v in orig_hist.state_dict().items()}
                    self.actor_target_history.load_state_dict(state_dict, strict=False)
                else:
                    # 普通模式（ZeRO-2 等）：使用 deepcopy
                    self.actor_target_history = copy.deepcopy(orig_hist)

                # 获取原始History的dtype（bf16模式下是bf16）
                if zero_stage == 3:
                    import deepspeed
                    with deepspeed.zero.GatheredParameters(list(orig_hist.parameters()), modifier_rank=None):
                        hist_dtype = next(orig_hist.parameters()).dtype
                else:
                    hist_dtype = next(orig_hist.parameters()).dtype
                self.actor_target_history.to(device=device, dtype=hist_dtype)
                for param in self.actor_target_history.parameters():
                    param.requires_grad = False
                rank0_print(f"  [History] ✓ Independent target created (dtype={hist_dtype})")
            else:
                rank0_print(f"  [History] Frozen, sharing with actor")
                self.actor_target_history = None
        else:
            self.actor_target_history = None

        # ============ 创建独立的 FC 层 ============
        # 不使用 VLM_DPT_Actor 实例，直接创建独立的 FC 层
        feature_dim = actor.feature_extractor.feature_dim
        action_dim = actor.action_dim
        self.actor_target_fc = nn.Linear(feature_dim, action_dim)

        # 复制 FC 层权重
        if zero_stage == 3:
            # ZeRO-3: 需要 gather FC 层参数
            import deepspeed
            with deepspeed.zero.GatheredParameters(list(actor.fc.parameters()), modifier_rank=None):
                fc_state_dict = {k: v.clone().cpu() for k, v in actor.fc.state_dict().items()}
            self.actor_target_fc.load_state_dict(fc_state_dict)
        else:
            self.actor_target_fc.load_state_dict(actor.fc.state_dict())

        # 获取原始FC的dtype（bf16模式下是bf16）
        if zero_stage == 3:
            import deepspeed
            with deepspeed.zero.GatheredParameters(list(actor.fc.parameters()), modifier_rank=None):
                fc_dtype = next(actor.fc.parameters()).dtype
        else:
            fc_dtype = next(actor.fc.parameters()).dtype
        self.actor_target_fc = self.actor_target_fc.to(device=device, dtype=fc_dtype)
        for param in self.actor_target_fc.parameters():
            param.requires_grad = False

        rank0_print(f"  [FC] ✓ Copied (dtype={fc_dtype})")
        rank0_print(f"[TD3] Created actor_target components (ZeRO-3 compatible)")
        rank0_print(f"  Note: No VLM_DPT_Actor instance, only independent FC/LoRA/DPT/History")

        # 不返回任何值，所有 target 组件都已存储为 self.actor_target_*
        return None

    def _create_critic_target(self, critic):
        """
        创建 critic_target 的 Q-heads（不创建完整的 VLM_DPT_Critic 对象）

        ZeRO-3 兼容：将 Q-heads 作为独立模块存储，不与 DeepSpeed 管理的模型混合

        原因：如果创建 VLM_DPT_Critic 对象并共享 feature_extractor，
        DeepSpeed 会在 partition 时尝试处理非 DeepSpeed 管理的参数，导致错误。
        """
        import torch.nn as nn

        # 检测 ZeRO-3 模式
        zero_stage = get_zero_stage()

        # 从 action_encoder 推断维度
        action_dim = critic.action_encoder[0].in_features
        feature_dim = critic.q1_head[0].in_features - 64  # feature_dim + action_feat(64)

        # 创建独立的 Q-heads（不作为任何 nn.Module 的子模块）
        self.critic_target_action_encoder = nn.Sequential(
            nn.Linear(action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        self.critic_target_q1_head = nn.Sequential(
            nn.Linear(feature_dim + 64, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.critic_target_q2_head = nn.Sequential(
            nn.Linear(feature_dim + 64, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # 复制权重
        if zero_stage == 3:
            import deepspeed

            with deepspeed.zero.GatheredParameters(list(critic.action_encoder.parameters()), modifier_rank=None):
                ae_state = {k: v.clone().cpu() for k, v in critic.action_encoder.state_dict().items()}
            self.critic_target_action_encoder.load_state_dict(ae_state)

            with deepspeed.zero.GatheredParameters(list(critic.q1_head.parameters()), modifier_rank=None):
                q1_state = {k: v.clone().cpu() for k, v in critic.q1_head.state_dict().items()}
            self.critic_target_q1_head.load_state_dict(q1_state)

            with deepspeed.zero.GatheredParameters(list(critic.q2_head.parameters()), modifier_rank=None):
                q2_state = {k: v.clone().cpu() for k, v in critic.q2_head.state_dict().items()}
            self.critic_target_q2_head.load_state_dict(q2_state)
        else:
            self.critic_target_action_encoder.load_state_dict(critic.action_encoder.state_dict())
            self.critic_target_q1_head.load_state_dict(critic.q1_head.state_dict())
            self.critic_target_q2_head.load_state_dict(critic.q2_head.state_dict())

        # 获取设备和dtype
        if zero_stage == 3:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            device = f"cuda:{local_rank}"
            import deepspeed
            with deepspeed.zero.GatheredParameters(list(critic.q1_head.parameters()), modifier_rank=None):
                target_dtype = next(critic.q1_head.parameters()).dtype
        else:
            device = next(critic.q1_head.parameters()).device
            target_dtype = next(critic.q1_head.parameters()).dtype

        # 移动到设备并设置dtype
        self.critic_target_action_encoder = self.critic_target_action_encoder.to(device=device, dtype=target_dtype)
        self.critic_target_q1_head = self.critic_target_q1_head.to(device=device, dtype=target_dtype)
        self.critic_target_q2_head = self.critic_target_q2_head.to(device=device, dtype=target_dtype)

        # 不需要梯度
        for param in self.critic_target_action_encoder.parameters():
            param.requires_grad = False
        for param in self.critic_target_q1_head.parameters():
            param.requires_grad = False
        for param in self.critic_target_q2_head.parameters():
            param.requires_grad = False

        # 保存 detach_features 设置
        self.critic_target_detach_features = critic.detach_features

        rank0_print(f"[TD3] Created critic_target Q-heads (dtype={target_dtype})")
        rank0_print(f"  Note: feature_extractor is shared with critic, Q-heads are independent")

    def _soft_update_actor(self):
        """
        Soft update actor_target 的可训练组件

        更新: FC + DPT (如果可训练) + History (如果可训练)

        ZeRO-3 兼容：使用 hard update 策略避免 GatheredParameters 冲突
        """
        zero_stage = get_zero_stage()

        # 解包 DataParallel
        actor_model = self._unwrap_model(self.actor)

        if zero_stage == 3:
            # ZeRO-3 多模型训练：GatheredParameters 可能失败（active_sub_modules 冲突）
            # 原因：critic 的参数在 actor_loss 计算后仍被标记为 active
            # 解决方案：使用 hard update 策略，每 N 步完全复制参数
            #
            # Hard update 间隔：基于 tau 计算等效的更新间隔
            # soft update: target = tau * current + (1-tau) * target
            # 经过 N 步后，target ≈ current when N * tau ≈ 1
            # 所以 N ≈ 1/tau
            hard_update_interval = max(1, int(1.0 / self.tau))

            if self.total_it % hard_update_interval == 0:
                # 使用 state_dict 进行 hard update
                # 这不需要 GatheredParameters，因为 target 网络不是 DeepSpeed 管理的

                # 1. 更新 FC 层
                self.actor_target_fc.load_state_dict(actor_model.fc.state_dict())

                # 2. LoRA target：不维护独立的 target LoRA
                # 原因：set_peft_model_state_dict 会替换参数张量，导致 DeepSpeed 的
                # ds_active_sub_modules 属性丢失。在 TD3 中，target 网络的稳定性
                # 主要来自 FC 层的慢更新，使用当前 LoRA 是可以接受的。

                # 3. 更新 DPT target（如果有独立副本）
                if hasattr(self, 'actor_target_dpt') and self.actor_target_dpt is not None:
                    self.actor_target_dpt.load_state_dict(
                        actor_model.feature_extractor.dpt_head.state_dict())

                # 4. 更新 History target（如果有独立副本）
                if hasattr(self, 'actor_target_history') and self.actor_target_history is not None:
                    self.actor_target_history.load_state_dict(
                        actor_model.feature_extractor.history_encoder.state_dict())
        else:
            # 非 ZeRO-3: 直接访问参数进行 soft update
            # 1. 更新 FC 层
            for param, target_param in zip(actor_model.fc.parameters(), self.actor_target_fc.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

            # 2. LoRA target：不维护独立的 target LoRA（同上）

            # 3. 更新 DPT target（如果有独立副本）
            if hasattr(self, 'actor_target_dpt') and self.actor_target_dpt is not None:
                for param, target_param in zip(
                    actor_model.feature_extractor.dpt_head.parameters(),
                    self.actor_target_dpt.parameters()
                ):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data)

            # 4. 更新 History target（如果有独立副本）
            if hasattr(self, 'actor_target_history') and self.actor_target_history is not None:
                for param, target_param in zip(
                    actor_model.feature_extractor.history_encoder.parameters(),
                    self.actor_target_history.parameters()
                ):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data)

    def _actor_target_forward(self, state):
        """
        使用独立的 target 组件计算 actor_target 的输出

        ZeRO-3 兼容：
        - 使用 critic 的 feature_extractor（而不是 actor 的）来避免 DeepSpeed 状态冲突
        - 因为 backward 是在 critic 上进行的，使用 critic 的 VLM 保持一致
        - 使用独立的 target 组件（FC/DPT/History）计算
        """
        import torch
        zero_stage = get_zero_stage()

        # 重要：使用 critic 的 feature_extractor 而不是 actor 的
        # 原因：在 ZeRO-3 模式下，backward 是在 critic 上进行的
        # 如果使用 actor 的 VLM 进行前向传播，会导致 DeepSpeed 参数管理冲突
        critic_model = self._unwrap_model(self.critic)
        fe = critic_model.feature_extractor

        # ========== Step 1: 获取 VLM 特征 ==========
        # 使用 critic 的 feature_extractor 获取 hidden states
        if zero_stage == 3:
            import deepspeed
            # 需要 gather history_encoder 的参数（如果使用 target history）
            if hasattr(self, 'actor_target_history') and self.actor_target_history is not None:
                # target history 不是 DeepSpeed 管理的，不需要 gather
                features = self._get_actor_features_with_target(state, fe)
            else:
                # history 冻结，使用原始的（critic 的 history_encoder）
                if hasattr(fe, 'history_encoder') and fe.history_encoder is not None:
                    history_params = list(fe.history_encoder.parameters())
                    with deepspeed.zero.GatheredParameters(history_params, modifier_rank=None):
                        features = self._get_actor_features_with_target(state, fe)
                else:
                    features = self._get_actor_features_with_target(state, fe)
        else:
            features = self._get_actor_features_with_target(state, fe)

        # ========== Step 2: 使用 target FC 计算 action ==========
        fc_dtype = self.actor_target_fc.weight.dtype
        features = features.to(dtype=fc_dtype)
        action = torch.tanh(self.actor_target_fc(features))

        return action

    def _get_actor_features_with_target(self, state, fe):
        """
        使用 feature_extractor 获取特征，但使用 target 的 DPT/History 组件

        这个方法模拟 VLM_DPT_FeatureExtractor.forward()，但使用 target 组件

        注意：为了 ZeRO-3 兼容性，不再为 target 网络维护独立的 LoRA。
        直接使用当前的 LoRA 来计算特征。在 TD3 中，target 网络的稳定性
        主要来自 Q-heads 和 FC 层的慢更新，使用当前 LoRA 是可以接受的。
        """
        import torch
        from PIL import Image

        # 解析输入格式
        if isinstance(state, list) and len(state) > 0 and isinstance(state[0], tuple):
            # VLMReplayBuffer format: List[(img, history_imgs, linear_vel, angular_vel)]
            imgs = [item[0] for item in state]
            hists = [item[1] for item in state]
            linear_vels = [item[2] if len(item) > 2 else 0.0 for item in state]
            angular_vels = [item[3] if len(item) > 3 else 0.0 for item in state]
            algorithm = fe.algorithm
        else:
            # 简单格式
            imgs = state
            hists = None
            linear_vels = None
            angular_vels = None
            algorithm = fe.algorithm

        # ========== 使用 VLM 获取 hidden states ==========
        # 使用当前的 LoRA（不做 swap，避免 DeepSpeed 参数管理冲突）
        hidden_states = self._run_vlm_forward(fe, imgs, linear_vels, angular_vels, algorithm)

        # ========== 使用 target DPT 处理 hidden states ==========
        if hasattr(self, 'actor_target_dpt') and self.actor_target_dpt is not None:
            dpt_head = self.actor_target_dpt
        else:
            dpt_head = fe.dpt_head

        # ========== 处理 history ==========
        history_feat = None
        if fe.use_history and hists is not None:
            if hasattr(self, 'actor_target_history') and self.actor_target_history is not None:
                history_encoder = self.actor_target_history
            else:
                history_encoder = fe.history_encoder

            if history_encoder is not None:
                # 处理 history 图像
                import torchvision.transforms as T
                transform = T.Compose([
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

                batch_history = []
                required_frames = fe.num_history_frames

                for hist_imgs in hists:
                    if hist_imgs is None or len(hist_imgs) == 0:
                        batch_history.append(torch.zeros(required_frames, 3, 224, 224))
                    else:
                        actual_frames = len(hist_imgs)
                        if actual_frames >= required_frames:
                            selected_imgs = hist_imgs[:required_frames]
                        else:
                            selected_imgs = hist_imgs + [hist_imgs[-1]] * (required_frames - actual_frames)
                        frames = [transform(img) for img in selected_imgs]
                        batch_history.append(torch.stack(frames))

                encoder_dtype = next(history_encoder.parameters()).dtype
                history_images = torch.stack(batch_history).to(device=fe.device, dtype=encoder_dtype)
                history_feat = history_encoder(history_images)

        # ========== 使用 DPT 提取特征 ==========
        features = self._extract_dpt_features_with_head(hidden_states, dpt_head, fe, history_feat)

        return features

    def _run_vlm_forward(self, fe, imgs, linear_vels, angular_vels, algorithm):
        """运行 VLM 前向传播获取 hidden states"""
        import torch

        # 构建 prompt 和处理输入（与 fe.forward 相同）
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

        batch_size = len(imgs)
        texts = []

        for i in range(batch_size):
            if linear_vels is not None and angular_vels is not None:
                user_prompt = f"""Current robot state:
- Linear velocity: {linear_vels[i]:.3f} m/s
- Angular velocity: {angular_vels[i]:.3f} rad/s

Target local planner: {algorithm}

Use the scene image and robot state to perform navigation reasoning about obstacle proximity, path curvature, free-space structure, and motion constraints."""
            else:
                user_prompt = f"""Target local planner: {algorithm}

Analyze the costmap to understand obstacle distribution and path geometry."""

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT.strip()},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": imgs[i]},
                        {"type": "text", "text": user_prompt.strip()},
                    ],
                }
            ]

            text = fe.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)

        inputs = fe.processor(
            text=texts, images=imgs, videos=None, padding=False, return_tensors="pt"
        ).to(fe.device)

        is_vlm_frozen = not any(p.requires_grad for p in fe.base_model.parameters())

        with torch.no_grad() if is_vlm_frozen else torch.enable_grad():
            outputs = fe.base_model(
                **inputs, output_hidden_states=True, return_dict=True
            )
            multi_layer_hidden_states = outputs.hidden_states[-4:]

        return multi_layer_hidden_states

    def _extract_dpt_features_with_head(self, multi_layer_hidden_states, dpt_head, fe, history_feat=None):
        """使用指定的 DPT head 提取特征"""
        import torch

        B, seq_len, _ = multi_layer_hidden_states[0].shape

        # 转换 dtype
        dpt_dtype = next(dpt_head.parameters()).dtype
        multi_layer_hidden_states = [hs.to(dtype=dpt_dtype) for hs in multi_layer_hidden_states]

        # 投影
        projected = [proj(hs) for proj, hs in zip(dpt_head.projections, multi_layer_hidden_states)]
        projected = [p.transpose(1, 2) for p in projected]

        # 融合
        fused = projected[-1]
        for i in range(len(dpt_head.fusion_blocks) - 1, -1, -1):
            skip = projected[i]
            fused = dpt_head.fusion_blocks[i](fused, skip)

        fused = fused.transpose(1, 2)

        # 注意力池化
        attention_weights = dpt_head.spatial_attention(fused)
        pooled = (fused * attention_weights).sum(dim=1)

        # 历史融合
        if fe.use_history and history_feat is not None and hasattr(dpt_head, 'history_fusion'):
            fusion_dtype = dpt_head.history_fusion[0].weight.dtype
            pooled = pooled.to(dtype=fusion_dtype)
            history_feat = history_feat.to(dtype=fusion_dtype)
            combined = torch.cat([pooled, history_feat], dim=-1)
            pooled = dpt_head.history_fusion(combined)

        return pooled

    def _critic_target_forward(self, state, action):
        """
        计算 critic_target 的 Q 值

        使用 critic.feature_extractor 获取特征，然后用独立的 Q-heads 计算 Q 值
        这样避免了 DeepSpeed ZeRO-3 的参数管理冲突

        ZeRO-3 兼容：history_encoder 参数被分片，需要先 gather
        """
        import torch
        zero_stage = get_zero_stage()

        # 解包 critic 模型
        critic_model = self._unwrap_model(self.critic)

        # 使用 critic 的 feature_extractor 获取特征
        if zero_stage == 3:
            import deepspeed
            fe = critic_model.feature_extractor
            if hasattr(fe, 'history_encoder') and fe.history_encoder is not None:
                history_params = list(fe.history_encoder.parameters())
                with deepspeed.zero.GatheredParameters(history_params, modifier_rank=None):
                    vlm_feat = self._get_critic_features(critic_model, state)
            else:
                vlm_feat = self._get_critic_features(critic_model, state)
        else:
            vlm_feat = self._get_critic_features(critic_model, state)

        # detach 特征（与 critic 的 detach_features 设置保持一致）
        if self.critic_target_detach_features:
            vlm_feat = vlm_feat.detach()

        # 使用独立的 Q-heads 计算 Q 值
        ae_dtype = self.critic_target_action_encoder[0].weight.dtype
        action = action.to(dtype=ae_dtype)
        action_feat = self.critic_target_action_encoder(action)

        q_head_dtype = self.critic_target_q1_head[0].weight.dtype
        vlm_feat = vlm_feat.to(dtype=q_head_dtype)
        action_feat = action_feat.to(dtype=q_head_dtype)

        q_input = torch.cat([vlm_feat, action_feat], dim=-1)
        q1 = self.critic_target_q1_head(q_input)
        q2 = self.critic_target_q2_head(q_input)

        return q1, q2

    def _get_critic_features(self, critic_model, state):
        """
        从 critic 的 feature_extractor 获取特征

        这是一个辅助方法，与 VLM_DPT_Critic 的 forward 逻辑一致
        """
        # Handle VLMReplayBuffer format: List[(img, history_imgs, linear_vel, angular_vel)]
        if isinstance(state, list) and len(state) > 0 and isinstance(state[0], tuple):
            imgs = [item[0] for item in state]
            hists = [item[1] for item in state]
            linear_vels = [item[2] if len(item) > 2 else 0.0 for item in state]
            angular_vels = [item[3] if len(item) > 3 else 0.0 for item in state]

            vlm_feat = critic_model.feature_extractor(
                imgs, None, hists,
                linear_vels=linear_vels,
                angular_vels=angular_vels,
                algorithm=critic_model.feature_extractor.algorithm
            )
        else:
            vlm_feat = critic_model.feature_extractor(state, None, None)

        return vlm_feat

    def _soft_update_critic(self):
        """
        Soft update critic_target 的 Q-heads（feature_extractor 是共享的，不需要更新）

        ZeRO-3 兼容：使用 hard update 策略避免 GatheredParameters 冲突
        """
        zero_stage = get_zero_stage()

        # 解包 DataParallel
        critic_model = self._unwrap_model(self.critic)

        if zero_stage == 3:
            # ZeRO-3 多模型训练：GatheredParameters 可能失败（active_sub_modules 冲突）
            # 原因：actor_loss 计算涉及 critic，导致 critic 参数被标记为 active
            # 解决方案：使用 hard update 策略，每 N 步完全复制参数
            #
            # Hard update 间隔：基于 tau 计算等效的更新间隔
            # soft update: target = tau * current + (1-tau) * target
            # 经过 N 步后，target ≈ current when N * tau ≈ 1
            # 所以 N ≈ 1/tau
            hard_update_interval = max(1, int(1.0 / self.tau))

            if self.total_it % hard_update_interval == 0:
                # 使用 state_dict 进行 hard update
                # 这不需要 GatheredParameters，因为 target 网络不是 DeepSpeed 管理的
                self.critic_target_action_encoder.load_state_dict(
                    critic_model.action_encoder.state_dict())
                self.critic_target_q1_head.load_state_dict(
                    critic_model.q1_head.state_dict())
                self.critic_target_q2_head.load_state_dict(
                    critic_model.q2_head.state_dict())
        else:
            # 非 ZeRO-3: 直接访问参数
            # action_encoder
            for param, target_param in zip(critic_model.action_encoder.parameters(),
                                           self.critic_target_action_encoder.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)
            # q1_head
            for param, target_param in zip(critic_model.q1_head.parameters(),
                                           self.critic_target_q1_head.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)
            # q2_head
            for param, target_param in zip(critic_model.q2_head.parameters(),
                                           self.critic_target_q2_head.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

    def select_action(self, state):
        """
        选择动作

        Args:
            state: PIL.Image (VLM模式) 或 np.array (传统模式)

        Returns:
            action: np.array
        """
        # ===== 推理模式（eval mode）=====
        self.actor.eval()

        # 检查state类型，兼容VLM和传统模式
        from PIL import Image
        if isinstance(state, Image.Image):
            # VLM模式: PIL.Image输入
            # actor.forward接受PIL.Image
            with torch.no_grad():
                action = self.actor([state]).cpu().data.numpy().flatten()  # List[PIL.Image]
        elif isinstance(state, list):
            # VLM模式: List[PIL.Image]
            with torch.no_grad():
                action = self.actor(state).cpu().data.numpy().flatten()
        else:
            # 传统模式: numpy array
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            with torch.no_grad():
                action = self.actor(state).cpu().data.numpy().flatten()

        # 添加探索噪声
        action += np.random.normal(0, self.exploration_noise, size=action.shape)

        # 缩放到动作空间
        action *= self._action_scale.cpu().data.numpy()
        action += self._action_bias.cpu().data.numpy()

        return action

    def train(self, replay_buffer, batch_size=256, verbose=False, gradient_accumulation_steps=1):
        import time as time_module
        self.total_it += 1

        # ===== 确保模型在训练模式（gradient checkpointing 需要）=====
        self.actor.train()
        # actor_target 组件设置为 train 模式（虽然不训练，但保持一致性）
        self.actor_target_fc.train()
        if hasattr(self, 'actor_target_dpt') and self.actor_target_dpt is not None:
            self.actor_target_dpt.train()
        if hasattr(self, 'actor_target_history') and self.actor_target_history is not None:
            self.actor_target_history.train()
        self.critic.train()
        # critic_target 的 Q-heads 是独立模块，设置为 train 模式
        self.critic_target_action_encoder.train()
        self.critic_target_q1_head.train()
        self.critic_target_q2_head.train()

        # 每50次迭代打印详细信息
        verbose = verbose or (self.total_it % 50 == 1)

        if verbose:
            rank0_print(f"    [TD3.train] iter={self.total_it}, batch_size={batch_size}, accum_steps={gradient_accumulation_steps}")
            rank0_print(f"               effective_batch={batch_size * gradient_accumulation_steps}")

        t0 = time_module.time()

        # ===== DDP 模式：所有进程独立采样（标准 DDP 做法）=====
        import torch.distributed as dist
        is_ddp = self.accelerator is not None and dist.is_initialized()

        # ===== 梯度累积循环 =====
        accumulated_critic_loss = 0.0
        accumulated_actor_loss = 0.0
        accumulated_actor_grad_norm = 0.0
        accumulated_critic_grad_norm = 0.0

        for accum_step in range(gradient_accumulation_steps):
            # 每个累积步独立采样（数据并行）
            state, action, next_state, reward, not_done, task, ind = replay_buffer.sample(batch_size)
            next_state, reward, not_done, gammas = replay_buffer.n_step_return(self.n_step, ind, self.gamma)

            # 获取模型的dtype（DeepSpeed bf16模式下是bf16）
            # 从Critic的Q-head获取，因为这是参与梯度计算的层
            model_dtype = self.critic.q1_head[0].weight.dtype

            # 将从buffer采样的tensor转换为模型dtype（避免float32与bf16混合）
            if action.dtype != model_dtype:
                action = action.to(dtype=model_dtype)
            if reward.dtype != model_dtype:
                reward = reward.to(dtype=model_dtype)
            if not_done.dtype != model_dtype:
                not_done = not_done.to(dtype=model_dtype)
            if gammas.dtype != model_dtype:
                gammas = gammas.to(dtype=model_dtype)

            if verbose and accum_step == 0:
                rank0_print(f"    [TD3.train] 采样完成: {time_module.time()-t0:.2f}s")
                t1 = time_module.time()

            # ===== Target 网络计算（无梯度）=====
            # 重要：所有 target 网络操作必须在 torch.no_grad() 内
            # 否则会创建包含未注册参数的计算图，导致 ZeRO-3 ds_active_sub_modules 错误
            with torch.no_grad():
                # Select action according to policy and add clipped noise
                # Actor输出[-1, 1]，映射到原始参数空间
                if verbose:
                    rank0_print(f"    [TD3.train] actor_target forward (VLM)...")
                # 使用独立的 DPT/History target（如果有）
                next_action_raw = self._actor_target_forward(next_state)  # [-1, 1]
                if verbose:
                    rank0_print(f"    [TD3.train] actor_target完成: {time_module.time()-t1:.2f}s")

                # Debug: check actor output
                if torch.isnan(next_action_raw).any() or torch.isinf(next_action_raw).any():
                    rank0_print(f"    [TD3.train] WARNING: next_action_raw contains nan/inf!")
                    rank0_print(f"      next_action_raw: {next_action_raw}")

                # 缩放到原始参数空间 (保持dtype一致)
                action_dtype = next_action_raw.dtype
                next_action_param = next_action_raw * self._action_scale.to(dtype=action_dtype) + self._action_bias.to(dtype=action_dtype)

                # 在原始参数空间添加噪声（policy noise）
                noise_param = (
                    torch.randn_like(next_action_param) * self.policy_noise * self.param_std.to(dtype=action_dtype)
                    if self.param_std is not None
                    else torch.randn_like(next_action_param) * self.policy_noise
                ).clamp(-self.noise_clip, self.noise_clip)

                next_action_param = next_action_param + noise_param

                # 归一化后给Critic (保持action的dtype，避免bf16被提升为float32)
                if self.param_mean is not None and self.param_std is not None:
                    action_dtype = next_action_param.dtype
                    next_action = (next_action_param - self.param_mean.to(dtype=action_dtype)) / (self.param_std.to(dtype=action_dtype) + 1e-8)
                else:
                    next_action = next_action_param

                # Compute the target Q value (使用独立的 DPT/History target)
                target_Q1, target_Q2 = self._critic_target_forward(next_state, next_action)

                # Debug: check critic_target output BEFORE min
                if torch.isnan(target_Q1).any() or torch.isnan(target_Q2).any():
                    rank0_print(f"    [TD3.train] WARNING: critic_target output contains nan!")
                    rank0_print(f"      next_action range: [{next_action.min():.4f}, {next_action.max():.4f}]")
                    rank0_print(f"      next_action has nan: {torch.isnan(next_action).any()}")

                target_Q = torch.min(target_Q1, target_Q2)

                # 转换reward, not_done, gammas到与target_Q相同的dtype
                reward = reward.to(dtype=target_Q.dtype)
                not_done = not_done.to(dtype=target_Q.dtype)
                gammas = gammas.to(dtype=target_Q.dtype)

                target_Q = reward + not_done * gammas * target_Q

            # Debug: check for nan in target_Q
            if torch.isnan(target_Q).any():
                rank0_print(f"    [TD3.train] WARNING: target_Q contains nan!")
                rank0_print(f"      reward range: [{reward.min():.4f}, {reward.max():.4f}]")
                rank0_print(f"      target_Q1 range: [{target_Q1.min():.4f}, {target_Q1.max():.4f}]")
                rank0_print(f"      target_Q2 range: [{target_Q2.min():.4f}, {target_Q2.max():.4f}]")

            # Get current Q estimates
            # 归一化action以提高训练稳定性（与监督学习保持一致）
            # 保持action的dtype，避免bf16被float32的param_mean/std提升
            if self.param_mean is not None and self.param_std is not None:
                action_dtype = action.dtype
                action_normalized = (action - self.param_mean.to(dtype=action_dtype)) / (self.param_std.to(dtype=action_dtype) + 1e-8)
            else:
                action_normalized = action

            current_Q1, current_Q2 = self.critic(state, action_normalized)

            # Compute critic loss (scale by accumulation steps)
            critic_loss = (F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)) / gradient_accumulation_steps

            # Skip update if loss is nan (prevent corrupting network)
            if torch.isnan(critic_loss):
                rank0_print(f"    [TD3.train] WARNING: critic_loss is nan at accum_step={accum_step}, skipping entire update!")
                return 0.0, 0.0, None, float('nan')

            # Backward pass (accumulate gradients)
            # 第一步：清空梯度
            if accum_step == 0:
                self.critic_optimizer.zero_grad()

            # ZeRO 兼容模式：使用 zero_trainer.backward()
            # 重要：必须传入 model 参数，让每个模型使用自己的 DeepSpeed 引擎
            # 这是 ZeRO-3 多模型训练的关键
            if self.zero_trainer is not None:
                self.zero_trainer.backward(critic_loss, model=self.critic)
            elif self.accelerator is not None:
                self.accelerator.backward(critic_loss)
            else:
                critic_loss.backward()

            # 累积损失（用于日志）
            accumulated_critic_loss += critic_loss.item()

        # ===== Critic 梯度累积循环结束，执行优化步骤 =====
        # 🔧 ZeRO-3 关键修复：在开始 actor 更新之前，先完成 critic 的更新
        # 这确保两个 DeepSpeedEngine 的操作不会交错
        grad_has_nan = False
        for name, p in self.critic.named_parameters():
            if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                if not grad_has_nan:
                    rank0_print(f"    [TD3.train] WARNING: critic gradients contain nan/inf BEFORE clipping!")
                    grad_has_nan = True
                rank0_print(f"      {name}: grad_max={p.grad.abs().max():.4f}")

        if grad_has_nan:
            rank0_print(f"    [TD3.train] Skipping entire update due to bad critic gradients")
            return 0.0, 0.0, None, accumulated_critic_loss

        # Gradient clipping（ZeRO 兼容）
        if self.zero_trainer is not None:
            self.zero_trainer.clip_grad_norm(self.critic, max_norm=0.5)
        else:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)

        self.critic_optimizer.step()
        accumulated_critic_grad_norm = self.grad_norm(self.critic)

        # ===== Actor 更新（完全分离的循环）=====
        # Delayed policy updates
        actor_updated = False
        if self.total_it % self.update_actor_freq == 0:
            # 重新采样用于 actor 更新（或重用之前的采样）
            # 这里我们重新循环采样，确保与 critic 完全分离
            self.actor_optimizer.zero_grad()

            # 🔧 ZeRO-2 关键修复：在 actor 更新期间冻结 critic 的参数
            # 原因：actor_loss = -critic.Q1(state, actor(state)) 涉及 critic 的参数
            # 当 backward 时，梯度会流过 critic，触发 critic 的 DeepSpeed hooks
            # 但 critic 的优化器状态可能不正确，导致 ipg_buffer 错误
            # 解决方案：冻结 critic 参数，这样 critic 的 hooks 不会被触发
            zero_stage = get_zero_stage()
            critic_params_grad_state = {}
            if zero_stage == 2:
                # 保存并冻结 critic 参数
                for name, param in self.critic.named_parameters():
                    critic_params_grad_state[name] = param.requires_grad
                    param.requires_grad = False

            for accum_step in range(gradient_accumulation_steps):
                # 每个累积步独立采样
                state, action, next_state, reward, not_done, task, ind = replay_buffer.sample(batch_size)

                # 获取模型的dtype
                model_dtype = self.actor.fc.weight.dtype

                # Compute actor loss (scale by accumulation steps)
                # Actor输出[-1, 1]，需要映射到原始参数空间 (保持dtype一致)
                actor_output = self.actor(state)  # [-1, 1]
                action_dtype = actor_output.dtype
                predicted_action = actor_output * self._action_scale.to(dtype=action_dtype) + self._action_bias.to(dtype=action_dtype)

                # 归一化后给Critic（与训练数据保持一致）
                if self.param_mean is not None and self.param_std is not None:
                    action_dtype = predicted_action.dtype
                    predicted_action_normalized = (predicted_action - self.param_mean.to(dtype=action_dtype)) / (self.param_std.to(dtype=action_dtype) + 1e-8)
                else:
                    predicted_action_normalized = predicted_action

                actor_loss = -self.critic.Q1(state, predicted_action_normalized).mean() / gradient_accumulation_steps

                # ZeRO 兼容模式：使用 zero_trainer.backward()
                if self.zero_trainer is not None:
                    # ZeRO-2: critic 已冻结，可以直接用 actor.backward()
                    self.zero_trainer.backward(actor_loss, model=self.actor, cross_model=False)
                elif self.accelerator is not None:
                    self.accelerator.backward(actor_loss)
                else:
                    actor_loss.backward()

                # 累积损失（用于日志）
                accumulated_actor_loss += actor_loss.item()

            # 🔧 恢复 critic 参数的 requires_grad 状态
            if zero_stage == 2:
                for name, param in self.critic.named_parameters():
                    if name in critic_params_grad_state:
                        param.requires_grad = critic_params_grad_state[name]

            # Actor 梯度裁剪和步进
            if self.zero_trainer is not None:
                self.zero_trainer.clip_grad_norm(self.actor, max_norm=0.5)
            else:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)

            self.actor_optimizer.step()
            actor_updated = True
            accumulated_actor_grad_norm = self.grad_norm(self.actor)

        # ===== Soft Update（在所有 backward/step 完成后）=====
        # 重要：在 ZeRO-3 模式下，必须在所有计算完成后才能做 soft update
        # 否则参数可能仍在被计算图引用，GatheredParameters 会失败
        #
        # 清理计算图引用
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        # Soft update target networks
        self._soft_update_critic()
        if actor_updated:
            self._soft_update_actor()

        return (accumulated_actor_grad_norm, accumulated_critic_grad_norm,
                accumulated_actor_loss if accumulated_actor_loss > 0 else None,
                accumulated_critic_loss)

    def grad_norm(self, model):
        total_norm = 0
        for p in model.parameters():
            param_norm = p.grad.data.norm(2).item() if p.grad is not None else 0
            total_norm += param_norm ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm

    def save(self, dir, filename):
        """
        保存Actor模型（VLM+DPT版本）- 统一使用标准格式

        保存到 dir/policy/ 目录：
           - adapter_model.safetensors + adapter_config.json (LoRA)
           - regression_head/pytorch_model.bin (DPT)
           - history_encoder/pytorch_model.bin
           - normalization/param_mean.npy + param_std.npy

        探索噪声单独保存到 dir/policy_noise（TD3 专用）

        策略：
        1. VLM base: 跳过（4-bit量化，从checkpoint重新加载）
        2. LoRA: 使用 PEFT 标准格式（bfloat16）
        3. Regression Head (DPT) + History: 使用 torch.save
        4. 归一化: 使用 numpy

        ZeRO-3 支持：
        - 在 ZeRO-3 模式下，参数被分片到不同 GPU
        - 使用 deepspeed.zero.GatheredParameters 收集参数后再保存
        - 只在 rank 0 执行实际的文件写入
        """
        import numpy as np
        import shutil

        # 获取 ZeRO 阶段和 local_rank
        from rlft.zero_utils import get_zero_stage, save_module_zero3, save_lora_zero3, get_peft_state_maybe_zero_3
        import torch.distributed as dist

        zero_stage = get_zero_stage()
        local_rank = int(os.environ.get("LOCAL_RANK", -1))

        # 创建 policy 目录（只有 rank 0）
        policy_dir = join(dir, filename)
        if local_rank in [0, -1]:
            os.makedirs(policy_dir, exist_ok=True)

        # ZeRO-3: 同步所有进程，确保目录已创建
        if zero_stage == 3 and dist.is_initialized():
            dist.barrier()

        rank0_print(f"\n[FTRL Save] 💾 保存模型到统一目录: {policy_dir}")
        if zero_stage == 3:
            rank0_print(f"[FTRL Save] 🔧 ZeRO-3 模式：将 gather 分片参数后保存")

        # ZeRO-3: 保存前切换到 eval 模式并清理缓存
        was_training = self.actor.training
        if zero_stage == 3:
            self.actor.eval()
            torch.cuda.empty_cache()

        # 处理 DataParallel：如果模型被包装，需要访问 .module
        actor_model = self.actor.module if isinstance(self.actor, torch.nn.DataParallel) else self.actor
        if isinstance(self.actor, torch.nn.DataParallel):
            rank0_print(f"[FTRL Save] 检测到 DataParallel，解包模型进行保存")

        # ===== 1. 保存 LoRA =====
        from peft import PeftModel
        # 使用更可靠的 PEFT 检测（ZeRO-3 兼容）
        base_model = actor_model.feature_extractor.base_model if hasattr(actor_model.feature_extractor, 'base_model') else None
        is_peft = (
            base_model is not None and
            (hasattr(base_model, 'peft_config') or hasattr(base_model, 'active_adapters') or isinstance(base_model, PeftModel))
        )

        if is_peft:
            rank0_print(f"[FTRL Save] 1️⃣ 保存 LoRA adapters...")
            lora_params_list = [(n, p.numel(), p.requires_grad) for n, p in actor_model.feature_extractor.base_model.named_parameters() if 'lora' in n.lower()]
            total_lora_params = sum(numel for _, numel, _ in lora_params_list)
            trainable_lora = sum(1 for _, _, requires_grad in lora_params_list if requires_grad)

            rank0_print(f"    - LoRA 参数: {total_lora_params:,} ({total_lora_params/1e6:.2f}M)")
            rank0_print(f"    - 可训练层: {trainable_lora}/{len(lora_params_list)}")

            if zero_stage == 3:
                # ZeRO-3 多模型训练：GatheredParameters 可能导致死锁
                # 直接从原始 checkpoint 复制 LoRA（如果 LoRA 冻结）
                # 或者跳过保存并警告（如果 LoRA 可训练）
                rank0_print(f"    [ZeRO-3] Checking LoRA trainability...")

                # 检查 LoRA 是否可训练
                lora_trainable = any(
                    p.requires_grad for n, p in base_model.named_parameters()
                    if 'lora_' in n.lower()
                )

                if lora_trainable:
                    # LoRA 可训练但 ZeRO-3 多模型下 gather 可能死锁
                    # 尝试不使用 GatheredParameters 直接保存（如果参数已经被 gather）
                    rank0_print(f"    [ZeRO-3] LoRA is trainable, trying direct save...")

                    try:
                        # 检查第一个 LoRA 参数是否有完整数据
                        first_lora = None
                        for n, p in base_model.named_parameters():
                            if 'lora_' in n.lower():
                                first_lora = p
                                break

                        if first_lora is not None and first_lora.numel() > 0:
                            # 参数有数据，尝试直接保存
                            if local_rank in [0, -1]:
                                lora_state = {}
                                for n, p in base_model.named_parameters():
                                    if 'lora_' in n.lower():
                                        # 尝试访问数据（如果是分片的可能会失败）
                                        try:
                                            lora_state[n] = p.data.detach().cpu().clone()
                                        except:
                                            pass

                                if lora_state:
                                    from safetensors.torch import save_file
                                    adapter_path = join(policy_dir, "adapter_model.safetensors")
                                    save_file(lora_state, adapter_path)
                                    rank0_print(f"    ✓ LoRA saved directly (no gather)")
                    except Exception as e:
                        rank0_print(f"    ⚠️  Direct LoRA save failed: {e}")

                # 复制 adapter_config.json（从原始 checkpoint）
                if hasattr(actor_model.feature_extractor, 'checkpoint_path') and local_rank in [0, -1]:
                    source_config = join(actor_model.feature_extractor.checkpoint_path, "adapter_config.json")
                    if os.path.exists(source_config):
                        import shutil
                        target_config = join(policy_dir, "adapter_config.json")
                        if not os.path.exists(target_config):
                            shutil.copy2(source_config, target_config)
                            rank0_print(f"    ✓ adapter_config.json copied")

                # 不使用 barrier，避免多模型死锁
                rank0_print(f"    [ZeRO-3] LoRA save complete (skipping barrier)")
            else:
                # ZeRO-2 或普通模式
                # 转换为 bfloat16 节省空间
                params_to_restore = []
                for name, param in actor_model.feature_extractor.base_model.named_parameters():
                    if 'lora' in name.lower() and param.dtype != torch.bfloat16:
                        params_to_restore.append((param, param.dtype))
                        param.data = param.data.to(torch.bfloat16)

                # 保存到 policy/ 目录
                actor_model.feature_extractor.base_model.save_pretrained(policy_dir)

                # 恢复 dtype
                for param, original_dtype in params_to_restore:
                    param.data = param.data.to(original_dtype)

            # 验证
            if local_rank in [0, -1]:
                adapter_model_path = join(policy_dir, "adapter_model.safetensors")
                if os.path.exists(adapter_model_path):
                    file_size = os.path.getsize(adapter_model_path) / 1024 / 1024
                    rank0_print(f"    ✓ adapter_model.safetensors ({file_size:.2f} MB)")
                    rank0_print(f"    ✓ adapter_config.json")
        else:
            # Fallback: 如果 LoRA 被冻结或 merge，从原始 checkpoint 复制
            rank0_print(f"[FTRL Save] ⚠️  LoRA 不是 PeftModel（可能已冻结或merge）")

            # 尝试从原始 checkpoint 复制 LoRA 文件
            if hasattr(actor_model.feature_extractor, 'checkpoint_path') and local_rank in [0, -1]:
                source_checkpoint = actor_model.feature_extractor.checkpoint_path
                source_adapter_config = join(source_checkpoint, "adapter_config.json")
                source_adapter_model = join(source_checkpoint, "adapter_model.safetensors")

                if os.path.exists(source_adapter_config) and os.path.exists(source_adapter_model):
                    rank0_print(f"[FTRL Save] 1️⃣ 从原始 checkpoint 复制 LoRA...")
                    rank0_print(f"    源路径: {source_checkpoint}")

                    shutil.copy2(source_adapter_config, join(policy_dir, "adapter_config.json"))
                    shutil.copy2(source_adapter_model, join(policy_dir, "adapter_model.safetensors"))

                    file_size = os.path.getsize(source_adapter_model) / 1024 / 1024
                    rank0_print(f"    ✓ adapter_model.safetensors ({file_size:.2f} MB) [复制自原始checkpoint]")
                    rank0_print(f"    ✓ adapter_config.json [复制自原始checkpoint]")
                else:
                    rank0_print(f"    ✗ 原始 checkpoint 中未找到 LoRA 文件，跳过")
            elif local_rank in [0, -1]:
                rank0_print(f"    ✗ 无法获取原始 checkpoint 路径，跳过")

        # ===== 2. 保存 Regression Head (DPT) =====
        if hasattr(actor_model.feature_extractor, 'dpt_head'):
            rank0_print(f"[FTRL Save] 2️⃣ 保存 Regression Head (DPT)...")
            regression_path = join(policy_dir, 'regression_head', 'pytorch_model.bin')

            dpt_head = actor_model.feature_extractor.dpt_head

            if zero_stage == 3:
                # ZeRO-3: 使用 GatheredParameters gather 分片参数
                import deepspeed
                rank0_print(f"    [ZeRO-3] Gathering DPT parameters...")

                # 收集所有 DPT 参数
                dpt_params_list = list(dpt_head.parameters())

                # 清除 active_sub_modules 以避免 partition 时的 assert 错误
                for param in dpt_params_list:
                    if hasattr(param, 'ds_active_sub_modules'):
                        param.ds_active_sub_modules.clear()

                with torch.no_grad():
                    with deepspeed.zero.GatheredParameters(dpt_params_list, modifier_rank=None):
                        # 所有 rank 都执行 clone，但只有 rank 0 保存文件
                        dpt_state_dict_clean = {}
                        for name, param in dpt_head.named_parameters():
                            dpt_state_dict_clean[name] = param.data.detach().cpu().clone()
                        for name, buf in dpt_head.named_buffers():
                            dpt_state_dict_clean[name] = buf.data.detach().cpu().clone()

                        if local_rank in [0, -1]:
                            try:
                                os.makedirs(join(policy_dir, 'regression_head'), exist_ok=True)
                                torch.save(dpt_state_dict_clean, regression_path)
                                dpt_params = sum(p.numel() for p in dpt_state_dict_clean.values())
                                rank0_print(f"    ✓ regression_head/pytorch_model.bin ({dpt_params:,} params)")
                            except Exception as e:
                                rank0_print(f"    ⚠️  DPT save failed: {e}")
            else:
                # 非 ZeRO-3: 直接保存
                if local_rank in [0, -1]:
                    try:
                        os.makedirs(join(policy_dir, 'regression_head'), exist_ok=True)
                        dpt_state_dict = dpt_head.state_dict()
                        dpt_state_dict_clean = {k: v.detach().cpu().clone() for k, v in dpt_state_dict.items()}
                        torch.save(dpt_state_dict_clean, regression_path)
                        dpt_params = sum(p.numel() for p in dpt_state_dict_clean.values())
                        rank0_print(f"    ✓ regression_head/pytorch_model.bin ({dpt_params:,} params)")
                    except Exception as e:
                        rank0_print(f"    ⚠️  DPT save failed: {e}")

            # 保存 history_config.json（用于 qwen_server.py 正确创建 DPTHead）
            if local_rank in [0, -1]:
                import json
                use_history = actor_model.feature_extractor.use_history if hasattr(actor_model.feature_extractor, 'use_history') else False
                history_dim = 256  # 默认值
                if hasattr(dpt_head, 'history_dim'):
                    history_dim = dpt_head.history_dim

                history_config = {
                    'use_history': use_history,
                    'history_dim': history_dim,
                    'history_image_size': 224  # 默认值
                }
                with open(join(policy_dir, 'history_config.json'), 'w') as f:
                    json.dump(history_config, f, indent=2)
                rank0_print(f"    ✓ history_config.json (use_history={use_history})")

        # ===== 3. 保存 History Encoder =====
        if hasattr(actor_model.feature_extractor, 'history_encoder') and actor_model.feature_extractor.history_encoder is not None:
            rank0_print(f"[FTRL Save] 3️⃣ 保存 History Encoder...")
            history_path = join(policy_dir, 'history_encoder', 'pytorch_model.bin')

            history_encoder = actor_model.feature_extractor.history_encoder

            if zero_stage == 3:
                # ZeRO-3: 使用 GatheredParameters gather 分片参数
                import deepspeed
                rank0_print(f"    [ZeRO-3] Gathering History Encoder parameters...")

                history_params_list = list(history_encoder.parameters())

                # 清除 active_sub_modules
                for param in history_params_list:
                    if hasattr(param, 'ds_active_sub_modules'):
                        param.ds_active_sub_modules.clear()

                with torch.no_grad():
                    with deepspeed.zero.GatheredParameters(history_params_list, modifier_rank=None):
                        # 所有 rank 都执行 clone，但只有 rank 0 保存文件
                        history_state_dict_clean = {}
                        for name, param in history_encoder.named_parameters():
                            history_state_dict_clean[name] = param.data.detach().cpu().clone()
                        for name, buf in history_encoder.named_buffers():
                            history_state_dict_clean[name] = buf.data.detach().cpu().clone()

                        if local_rank in [0, -1]:
                            try:
                                os.makedirs(join(policy_dir, 'history_encoder'), exist_ok=True)
                                torch.save(history_state_dict_clean, history_path)
                                history_params = sum(p.numel() for p in history_state_dict_clean.values())
                                rank0_print(f"    ✓ history_encoder/pytorch_model.bin ({history_params:,} params)")
                            except Exception as e:
                                rank0_print(f"    ⚠️  History Encoder save failed: {e}")
            else:
                # 非 ZeRO-3: 直接保存
                if local_rank in [0, -1]:
                    try:
                        os.makedirs(join(policy_dir, 'history_encoder'), exist_ok=True)
                        history_state_dict = history_encoder.state_dict()
                        history_state_dict_clean = {k: v.detach().cpu().clone() for k, v in history_state_dict.items()}
                        torch.save(history_state_dict_clean, history_path)
                        history_params = sum(p.numel() for p in history_state_dict_clean.values())
                        rank0_print(f"    ✓ history_encoder/pytorch_model.bin ({history_params:,} params)")
                    except Exception as e:
                        rank0_print(f"    ⚠️  History Encoder save failed: {e}")

        # ===== 4. 保存 FC 层（Actor 的输出层）=====
        if hasattr(actor_model, 'fc'):
            rank0_print(f"[FTRL Save] 4️⃣ 保存 FC 层（Actor 输出层）...")
            fc_path = join(policy_dir, 'fc_layer', 'pytorch_model.bin')

            fc_layer = actor_model.fc

            if zero_stage == 3:
                # ZeRO-3: 使用 GatheredParameters gather 分片参数
                import deepspeed
                rank0_print(f"    [ZeRO-3] Gathering FC parameters...")

                fc_params_list = list(fc_layer.parameters())

                # 清除 active_sub_modules
                for param in fc_params_list:
                    if hasattr(param, 'ds_active_sub_modules'):
                        param.ds_active_sub_modules.clear()

                with torch.no_grad():
                    with deepspeed.zero.GatheredParameters(fc_params_list, modifier_rank=None):
                        # 所有 rank 都执行 clone，但只有 rank 0 保存文件
                        fc_state_dict_clean = {}
                        for name, param in fc_layer.named_parameters():
                            fc_state_dict_clean[name] = param.data.detach().cpu().clone()
                        for name, buf in fc_layer.named_buffers():
                            fc_state_dict_clean[name] = buf.data.detach().cpu().clone()

                        if local_rank in [0, -1]:
                            try:
                                os.makedirs(join(policy_dir, 'fc_layer'), exist_ok=True)
                                torch.save(fc_state_dict_clean, fc_path)
                                fc_params = sum(p.numel() for p in fc_state_dict_clean.values())
                                rank0_print(f"    ✓ fc_layer/pytorch_model.bin ({fc_params:,} params)")
                            except Exception as e:
                                rank0_print(f"    ⚠️  FC save failed: {e}")
            else:
                # 非 ZeRO-3: 直接保存
                if local_rank in [0, -1]:
                    try:
                        os.makedirs(join(policy_dir, 'fc_layer'), exist_ok=True)
                        fc_state_dict = fc_layer.state_dict()
                        fc_state_dict_clean = {k: v.detach().cpu().clone() for k, v in fc_state_dict.items()}
                        torch.save(fc_state_dict_clean, fc_path)
                        fc_params = sum(p.numel() for p in fc_state_dict_clean.values())
                        rank0_print(f"    ✓ fc_layer/pytorch_model.bin ({fc_params:,} params)")
                    except Exception as e:
                        rank0_print(f"    ⚠️  FC save failed: {e}")

        # ===== 5. 保存归一化参数 =====
        if self.param_mean is not None and self.param_std is not None and local_rank in [0, -1]:
            rank0_print(f"[FTRL Save] 5️⃣ 保存归一化参数...")
            norm_dir = join(policy_dir, 'normalization')
            os.makedirs(norm_dir, exist_ok=True)
            np.save(join(norm_dir, 'param_mean.npy'), self.param_mean.cpu().numpy())
            np.save(join(norm_dir, 'param_std.npy'), self.param_std.cpu().numpy())
            rank0_print(f"    ✓ normalization/param_mean.npy")
            rank0_print(f"    ✓ normalization/param_std.npy")

        # ===== 6. 保存探索噪声（TD3 专用，在 policy/ 里面）=====
        if local_rank in [0, -1]:
            rank0_print(f"[FTRL Save] 6️⃣ 保存探索噪声 (TD3)...")
            noise_path = join(policy_dir, "exploration_noise.pkl")
            with open(noise_path, "wb") as f:
                pickle.dump(self.exploration_noise, f)
            rank0_print(f"    ✓ exploration_noise.pkl")

        # ZeRO-3: 同步所有进程（所有进程都参与 save，所以可以 barrier）
        if zero_stage == 3 and dist.is_initialized():
            dist.barrier()
            rank0_print(f"[FTRL Save] 🔄 所有进程已同步")

        # ZeRO-3: 恢复训练模式
        if zero_stage == 3 and was_training:
            self.actor.train()

        rank0_print(f"\n[FTRL Save] ✅ 完成！模型已保存到: {policy_dir}")
        rank0_print(f"[FTRL Save]    可直接用于: qwen_server.py --lora_path {policy_dir}")

    def load(self, dir, filename):
        """
        加载 RLFT checkpoint（统一从 policy/ 目录加载）

        ZeRO-3 支持：
        - 在 ZeRO-3 模式下，需要使用 GatheredParameters 来加载分片参数
        - 所有 rank 都需要执行加载（DeepSpeed 会处理参数分发）
        """
        from rlft.zero_utils import get_zero_stage

        policy_dir = join(dir, filename)
        zero_stage = get_zero_stage()

        # 检查 policy/ 目录是否存在
        if not os.path.exists(policy_dir):
            rank0_print(f"[Load] 无 RLFT checkpoint: {policy_dir}")
            rank0_print(f"[Load] 将使用 Stage 1 初始化")
            return

        rank0_print(f"\n[Load] 📂 从统一目录加载: {policy_dir}")
        if zero_stage == 3:
            rank0_print(f"[Load] 🔧 ZeRO-3 模式：将使用 GatheredParameters 加载分片参数")

        # 处理 DataParallel：如果模型被包装，需要访问 .module
        actor_model = self.actor.module if isinstance(self.actor, torch.nn.DataParallel) else self.actor
        if isinstance(self.actor, torch.nn.DataParallel):
            rank0_print(f"[Load] 检测到 DataParallel，解包模型进行加载")

        # ===== 1. 加载 LoRA =====
        adapter_model_path = join(policy_dir, "adapter_model.safetensors")
        if os.path.exists(adapter_model_path):
            rank0_print(f"[Load] 1️⃣ 加载 LoRA adapters...")
            from peft import PeftModel

            try:
                is_peft = isinstance(actor_model.feature_extractor.base_model, PeftModel)

                if is_peft:
                    # 卸载旧的 LoRA
                    rank0_print(f"    - 检测到已有 PeftModel，卸载旧适配器...")
                    if hasattr(actor_model.feature_extractor.base_model, 'unload'):
                        base_model_clean = actor_model.feature_extractor.base_model.unload()
                    elif hasattr(actor_model.feature_extractor.base_model, 'get_base_model'):
                        base_model_clean = actor_model.feature_extractor.base_model.get_base_model()
                    else:
                        base_model_clean = actor_model.feature_extractor.base_model.model

                    # 加载新 LoRA
                    actor_model.feature_extractor.base_model = PeftModel.from_pretrained(
                        base_model_clean, policy_dir, is_trainable=True
                    )
                else:
                    # 直接加载
                    actor_model.feature_extractor.base_model = PeftModel.from_pretrained(
                        actor_model.feature_extractor.base_model, policy_dir, is_trainable=True
                    )

                file_size = os.path.getsize(adapter_model_path) / 1024 / 1024
                rank0_print(f"    ✓ LoRA adapters ({file_size:.2f} MB)")
            except Exception as e:
                rank0_print(f"    ⚠️  加载 LoRA 失败: {e}")

        # ===== 2. 加载 Regression Head (DPT) =====
        regression_path = join(policy_dir, 'regression_head', 'pytorch_model.bin')
        if os.path.exists(regression_path):
            rank0_print(f"[Load] 2️⃣ 加载 Regression Head (DPT)...")
            dpt_state_dict = torch.load(regression_path, map_location='cpu')

            if zero_stage == 3:
                # ZeRO-3: 需要在 GatheredParameters 中加载
                import deepspeed
                rank0_print(f"    [ZeRO-3] Loading into sharded DPT...")
                with deepspeed.zero.GatheredParameters(
                    list(actor_model.feature_extractor.dpt_head.parameters()),
                    modifier_rank=None  # 所有 rank 都需要加载
                ):
                    actor_model.feature_extractor.dpt_head.load_state_dict(dpt_state_dict, strict=False)
            else:
                actor_model.feature_extractor.dpt_head.load_state_dict(dpt_state_dict, strict=False)
            rank0_print(f"    ✓ regression_head/pytorch_model.bin")

        # ===== 3. 加载 History Encoder =====
        history_path = join(policy_dir, 'history_encoder', 'pytorch_model.bin')
        if os.path.exists(history_path):
            rank0_print(f"[Load] 3️⃣ 加载 History Encoder...")
            history_state_dict = torch.load(history_path, map_location='cpu')

            if hasattr(actor_model.feature_extractor, 'history_encoder') and actor_model.feature_extractor.history_encoder is not None:
                if zero_stage == 3:
                    # ZeRO-3: 需要在 GatheredParameters 中加载
                    import deepspeed
                    rank0_print(f"    [ZeRO-3] Loading into sharded History Encoder...")
                    with deepspeed.zero.GatheredParameters(
                        list(actor_model.feature_extractor.history_encoder.parameters()),
                        modifier_rank=None
                    ):
                        actor_model.feature_extractor.history_encoder.load_state_dict(history_state_dict, strict=False)
                else:
                    actor_model.feature_extractor.history_encoder.load_state_dict(history_state_dict, strict=False)
                rank0_print(f"    ✓ history_encoder/pytorch_model.bin")

        # ===== 4. 加载 FC 层（Actor 输出层）=====
        fc_path = join(policy_dir, 'fc_layer', 'pytorch_model.bin')
        if os.path.exists(fc_path):
            rank0_print(f"[Load] 4️⃣ 加载 FC 层（Actor 输出层）...")
            fc_state_dict = torch.load(fc_path, map_location='cpu')

            if hasattr(actor_model, 'fc'):
                if zero_stage == 3:
                    # ZeRO-3: 需要在 GatheredParameters 中加载
                    import deepspeed
                    rank0_print(f"    [ZeRO-3] Loading into sharded FC...")
                    with deepspeed.zero.GatheredParameters(
                        list(actor_model.fc.parameters()),
                        modifier_rank=None
                    ):
                        actor_model.fc.load_state_dict(fc_state_dict, strict=False)
                else:
                    actor_model.fc.load_state_dict(fc_state_dict, strict=False)
                rank0_print(f"    ✓ fc_layer/pytorch_model.bin")

        # ===== 5. 加载归一化参数 =====
        param_mean_path = join(policy_dir, 'normalization', 'param_mean.npy')
        param_std_path = join(policy_dir, 'normalization', 'param_std.npy')
        if os.path.exists(param_mean_path) and os.path.exists(param_std_path):
            rank0_print(f"[Load] 5️⃣ 加载归一化参数...")
            import numpy as np
            self.param_mean = torch.tensor(np.load(param_mean_path), dtype=torch.float32, device=self.device)
            self.param_std = torch.tensor(np.load(param_std_path), dtype=torch.float32, device=self.device)
            rank0_print(f"    ✓ normalization/")

        # ===== 6. 加载探索噪声（TD3 专用）=====
        noise_path = join(policy_dir, "exploration_noise.pkl")
        # 兼容旧格式
        if not os.path.exists(noise_path):
            noise_path = join(dir, filename + "_noise")
        if os.path.exists(noise_path):
            rank0_print(f"[Load] 6️⃣ 加载探索噪声...")
            with open(noise_path, "rb") as f:
                self.exploration_noise = pickle.load(f)
            rank0_print(f"    ✓ exploration_noise.pkl")

        # ===== 7. 更新 actor_target 组件 =====
        # 重新创建独立的 target 组件（FC/LoRA/DPT/History）
        self._create_actor_target(self._unwrap_model(self.actor))

        # 同步所有进程（确保加载完成后再继续）
        if zero_stage in [2, 3]:
            from rlft.ddp_utils import barrier
            barrier()
            rank0_print(f"[Load] 🔄 所有进程已同步")

        rank0_print(f"[Load] ✅ 加载完成！")


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6), device="cpu"):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.mean, self.std = 0.0, 1.0

        self.state = np.zeros((max_size, *state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, *state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))
        self.task = np.zeros((max_size, 1))

        self.device = device

    def add(self, state, action, next_state, reward, done, task):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward # (reward - 0.02478) / 6.499
        self.not_done[self.ptr] = 1. - done
        self.task[self.ptr] = task

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        if self.ptr == 1000:
            rew = self.reward[:1000]
            self.mean, self.std = rew.mean(), rew.std()
            if np.isclose(self.std, 0, 1e-2):
                self.mean, self.std = 0.0, 1.0

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.FloatTensor(self.task[ind]).to(self.device),
            ind)

    def n_step_return(self, n_step, ind, gamma):
        reward = []
        not_done = []
        next_state = []
        gammas = []
        for i in ind:
            n = 0
            r = 0
            for _ in range(n_step):
                idx = (i + n) % self.size
                r += (self.reward[idx] - self.mean) / self.std * gamma ** n
                if not self.not_done[idx]:
                    break
                n = n + 1
            next_state.append(self.next_state[idx])
            not_done.append(self.not_done[idx])
            reward.append(r)
            gammas.append([gamma ** (n + 1)])
        next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
        not_done = torch.FloatTensor(np.array(not_done)).to(self.device)
        reward = torch.FloatTensor(np.array(reward)).to(self.device)
        gammas = torch.FloatTensor(np.array(gammas)).to(self.device)
        return next_state, reward, not_done, gammas