"""
ZeRO 优化工具函数

提供 ZeRO-2 和 ZeRO-3 兼容的训练工具：
- 自动检测 DeepSpeed/ZeRO 模式
- 兼容 4-bit 量化模型（VLM base 不参与 ZeRO 分片）
- 统一的 backward 和 gradient sync 接口

⚠️ 重要限制：
- ZeRO-3 与 4-bit 量化 VLM 不兼容！
- 如果使用 bitsandbytes 4-bit 量化，只能使用 ZeRO-2
- 原因：NF4 格式不能被 ZeRO-3 分片

使用方法：
    # 方式1：使用 accelerate launch（推荐）
    $ accelerate launch --config_file accelerate_configs/zero2_config.yaml train.py ...

    # 方式2：使用环境变量指定 ZeRO 阶段
    $ ZERO_STAGE=2 torchrun --nproc_per_node=3 train.py ...
"""
import os
from typing import Optional, Tuple, List, Any

import torch
import torch.nn as nn


def is_deepspeed_enabled() -> bool:
    """检测是否在 DeepSpeed 模式下运行"""
    try:
        from accelerate import Accelerator
        from accelerate.state import AcceleratorState
        state = AcceleratorState()
        return state.deepspeed_plugin is not None
    except:
        return False


def get_zero_stage() -> int:
    """
    获取当前 ZeRO 优化阶段

    Returns:
        0: 无 ZeRO
        2: ZeRO-2 (optimizer + gradient 分片)
        3: ZeRO-3 (optimizer + gradient + param 分片)
    """
    # 方式1：从 accelerate 获取
    try:
        from accelerate.state import AcceleratorState
        state = AcceleratorState()
        if state.deepspeed_plugin is not None:
            return state.deepspeed_plugin.zero_stage
    except:
        pass

    # 方式2：从环境变量获取
    zero_stage = os.environ.get("ZERO_STAGE", "0")
    return int(zero_stage)


def is_zero3() -> bool:
    """检测是否使用 ZeRO-3"""
    return get_zero_stage() == 3


def is_zero2() -> bool:
    """检测是否使用 ZeRO-2"""
    return get_zero_stage() == 2


class _AttrDict(dict):
    """
    支持动态属性赋值的 dict 子类

    用于修复 DeepSpeed ZeRO-3 与某些模型（如 PEFT）的兼容性问题。
    DeepSpeed 需要在 module._parameters 上设置 _in_forward 属性，
    但普通 dict 不支持这种操作。

    技术细节：
    - Python 的普通 dict 不支持 `d._in_forward = True` 这种属性赋值
    - DeepSpeed ZeRO-3 的 PartitionedParameterCoordinator 会尝试这样做
    - 继承 dict 并允许任意属性赋值可以解决这个问题

    关键方法：
    - __setattr__: 允许 `d._in_forward = True` 这种赋值
    - __getattr__: 对于不存在的属性返回 None（而不是抛出 AttributeError）
    """

    def __setattr__(self, name, value):
        """允许任意属性赋值"""
        # 使用 object.__setattr__ 确保属性存储在实例的 __dict__ 中
        object.__setattr__(self, name, value)

    def __getattr__(self, name):
        """
        处理不存在的属性访问

        注意：__getattr__ 只在属性不存在时调用（__getattribute__ 失败后）
        返回 None 而不是抛出 AttributeError，与 DeepSpeed 的期望行为一致
        """
        # 直接返回 None，因为 __getattr__ 只在属性真正不存在时才被调用
        return None


_deepspeed_patched = False


def patch_deepspeed_engine():
    """
    Monkey-patch DeepSpeed engine 以处理 _parameters 兼容性问题
    """
    global _deepspeed_patched
    if _deepspeed_patched:
        return
    _deepspeed_patched = True  # 先设置，避免重复调用

    from rlft.ddp_utils import rank0_print
    rank0_print("[DeepSpeed Patch] Installing patch...")

    try:
        import deepspeed.runtime.engine as ds_engine

        # 保存原始的 forward 方法
        _original_forward = ds_engine.DeepSpeedEngine.forward

        def _patched_forward(self, *inputs, **kwargs):
            # 每次 forward 前修复 _parameters（确保所有模块都被修复）
            for module in self.module.modules():
                if hasattr(module, '_parameters'):
                    params = module._parameters
                    if type(params) is dict:  # 只替换普通 dict
                        module._parameters = _AttrDict(params)
            return _original_forward(self, *inputs, **kwargs)

        ds_engine.DeepSpeedEngine.forward = _patched_forward
        rank0_print("[DeepSpeed Patch] ✓ OK")

    except Exception as e:
        rank0_print(f"[DeepSpeed Patch] ⚠️ Failed: {e}")


def ensure_deepspeed_compatible(model):
    """
    确保模型与 DeepSpeed ZeRO-3 兼容
    在 accelerator.prepare() 之后、第一次 forward 之前调用
    """
    from rlft.ddp_utils import rank0_print

    if not hasattr(model, 'module'):
        return

    fixed = 0
    for module in model.module.modules():
        if hasattr(module, '_parameters'):
            params = module._parameters
            if type(params) is dict:
                module._parameters = _AttrDict(params)
                fixed += 1

    if fixed > 0:
        rank0_print(f"[DeepSpeed Fix] Fixed {fixed} modules")


def fix_model_for_deepspeed(model: nn.Module, verbose: bool = False, name: str = "") -> int:
    """
    修复模型以兼容 DeepSpeed ZeRO-3

    问题：DeepSpeed ZeRO-3 在 accelerator.prepare() 和 forward 时会遍历所有模块，
    尝试设置 module._parameters._in_forward = True

    但某些模型（如 PEFT/LoRA）的模块可能有非标准的 _parameters 实现，
    导致 AttributeError: 'dict' object has no attribute '_in_forward'

    解决方案：在 accelerator.prepare() 之前，确保所有模块的 _parameters
    是可以接受属性赋值的 _AttrDict 对象。

    ⚠️ 重要时机：
    - 必须在 accelerator.prepare() 之前调用
    - 必须在模型创建完成后调用（包括 PEFT/LoRA 加载后）

    Args:
        model: PyTorch 模型
        verbose: 是否打印详细信息
        name: 模型名称（用于日志）

    Returns:
        fixed_count: 修复的模块数量
    """
    from rlft.ddp_utils import rank0_print

    fixed_count = 0
    problematic_modules = []

    for mod_name, module in model.named_modules():
        # 检查 _parameters 是否存在
        if hasattr(module, '_parameters'):
            params = module._parameters

            # 检查是否是普通 dict（需要替换）
            # 使用 type() 而不是 isinstance()，因为我们只想替换普通 dict
            if type(params) is dict:
                # 创建支持属性赋值的 _AttrDict 并复制内容
                new_params = _AttrDict(params)
                module._parameters = new_params
                fixed_count += 1
                problematic_modules.append(mod_name)

    if fixed_count > 0:
        prefix = f"[{name}] " if name else ""
        rank0_print(f"{prefix}[DeepSpeed Fix] Fixed {fixed_count} modules with plain dict _parameters")
        if verbose and len(problematic_modules) <= 10:
            for mod_name in problematic_modules:
                rank0_print(f"  - {mod_name}")
        elif verbose:
            rank0_print(f"  (showing first 10 of {len(problematic_modules)})")
            for mod_name in problematic_modules[:10]:
                rank0_print(f"  - {mod_name}")

    return fixed_count


def fix_all_models_for_deepspeed(*models, verbose: bool = False) -> int:
    """
    修复多个模型以兼容 DeepSpeed ZeRO-3

    Args:
        *models: 多个 PyTorch 模型
        verbose: 是否打印详细信息

    Returns:
        total_fixed: 修复的模块总数
    """
    total_fixed = 0
    for i, model in enumerate(models):
        if model is not None:
            fixed = fix_model_for_deepspeed(model, verbose=verbose, name=f"model_{i}")
            total_fixed += fixed
    return total_fixed


class ZeROCompatibleTrainer:
    """
    ZeRO-2/ZeRO-3 兼容的训练器包装

    关键设计：
    1. 4-bit 量化的 VLM base 不参与 ZeRO 分片（保持 frozen）
    2. 可训练部分（LoRA, DPT, FC）使用 accelerator.prepare()
    3. 统一使用 accelerator.backward() 替代 loss.backward()

    使用方法：
        trainer = ZeROCompatibleTrainer(accelerator)

        # 准备模型（只准备可训练部分）
        actor_optim = trainer.prepare_optimizer(actor, actor_optim)
        critic_optim = trainer.prepare_optimizer(critic, critic_optim)

        # 训练循环
        loss = compute_loss()
        trainer.backward(loss)

        # 检查是否需要同步梯度（梯度累积时）
        if trainer.should_sync_gradients(accum_step, total_accum_steps):
            optimizer.step()
    """

    def __init__(self, accelerator: Optional[Any] = None):
        """
        Args:
            accelerator: Accelerate 的 Accelerator 实例
        """
        self.accelerator = accelerator
        self._zero_stage = get_zero_stage()

        # 日志
        from rlft.ddp_utils import rank0_print, is_main_process
        if is_main_process():
            rank0_print(f"[ZeROCompatibleTrainer] Initialized")
            rank0_print(f"    ZeRO Stage: {self._zero_stage}")
            rank0_print(f"    DeepSpeed: {is_deepspeed_enabled()}")
            rank0_print(f"    Accelerator: {'Yes' if accelerator else 'No'}")

    @property
    def zero_stage(self) -> int:
        return self._zero_stage

    def _suppress_deepspeed_logs(self):
        """禁用 DeepSpeed 的冗余日志（在 accelerator.prepare 之前调用）"""
        import logging
        # 设置所有 DeepSpeed 相关 logger 为 WARNING 级别
        deepspeed_loggers = [
            "deepspeed",
            "deepspeed.runtime.config",
            "deepspeed.runtime.zero.config",
            "deepspeed.runtime.zero.stage3",
            "deepspeed.runtime.zero.stage_1_and_2",
            "deepspeed.runtime.utils",
            "deepspeed.utils",
            "deepspeed.runtime.engine",
            "deepspeed.runtime.zero.partition_parameters",
            "deepspeed.runtime.zero.partitioned_param_coordinator",
        ]
        for logger_name in deepspeed_loggers:
            logging.getLogger(logger_name).setLevel(logging.WARNING)

    def prepare_optimizer(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer
    ) -> torch.optim.Optimizer:
        """
        准备优化器（ZeRO 兼容）

        注意：4-bit 量化模型不能用 accelerator.prepare(model)
        只能准备 optimizer

        Args:
            model: PyTorch 模型
            optimizer: 优化器

        Returns:
            准备好的优化器
        """
        if self.accelerator is not None:
            # 禁用 DeepSpeed 冗余日志
            self._suppress_deepspeed_logs()
            # 只准备优化器，不准备模型（4-bit 模型兼容）
            optimizer = self.accelerator.prepare(optimizer)
        return optimizer

    def prepare_model_and_optimizer(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        is_quantized: bool = False
    ) -> Tuple[nn.Module, torch.optim.Optimizer]:
        """
        准备模型和优化器

        Args:
            model: PyTorch 模型
            optimizer: 优化器
            is_quantized: 是否是量化模型（4-bit/8-bit）

        Returns:
            (model, optimizer) - 准备好的模型和优化器
        """
        if self.accelerator is None:
            return model, optimizer

        # 在 accelerator.prepare() 之前禁用 DeepSpeed 冗余日志
        self._suppress_deepspeed_logs()

        if is_quantized:
            # 量化模型：只准备优化器
            optimizer = self.accelerator.prepare(optimizer)
            return model, optimizer
        else:
            # 非量化模型：可以完整准备
            if get_zero_stage() == 3:
                # 🔧 ZeRO-3 关键修复：在 accelerator.prepare() 之前修复 _parameters
                # DeepSpeed 在 prepare() 期间会遍历所有模块设置 _in_forward 属性
                # 必须在此之前将所有普通 dict 替换为 _AttrDict
                from rlft.ddp_utils import rank0_print
                rank0_print("[ZeRO-3] Fixing model _parameters before accelerator.prepare()...")
                fixed = fix_model_for_deepspeed(model, verbose=False, name="pre-prepare")
                rank0_print(f"[ZeRO-3] ✓ Fixed {fixed} modules")

                # 同时安装 forward patch 作为备份
                patch_deepspeed_engine()

            model, optimizer = self.accelerator.prepare(model, optimizer)

            # 🔧 验证 DeepSpeed 包装
            from rlft.ddp_utils import rank0_print
            zero_stage = get_zero_stage()
            if zero_stage in [2, 3]:
                has_backward = hasattr(model, 'backward')
                model_type = type(model).__name__
                rank0_print(f"[ZeRO-{zero_stage}] Model after prepare: {model_type}, has backward: {has_backward}")

                # 检查优化器类型
                opt_type = type(optimizer).__name__
                rank0_print(f"[ZeRO-{zero_stage}] Optimizer after prepare: {opt_type}")

            # 🔧 ZeRO-3: prepare 之后再次检查（以防新模块被创建）
            if get_zero_stage() == 3:
                ensure_deepspeed_compatible(model)

                # 🔧 ZeRO-3: 在 prepare 之后启用 gradient checkpointing
                # 这样 DeepSpeed 的 hooks 已经设置好了
                if hasattr(model, 'feature_extractor') and hasattr(model.feature_extractor, 'enable_gradient_checkpointing'):
                    model.feature_extractor.enable_gradient_checkpointing()

            return model, optimizer

    def backward(self, loss: torch.Tensor, retain_graph: bool = False, model: nn.Module = None, cross_model: bool = False):
        """
        反向传播（ZeRO 兼容）

        重要：对于多模型训练（如 TD3 的 actor/critic），必须传入 model 参数！

        策略：
        - ZeRO-3: 使用 loss.backward()（DeepSpeed hooks 自动处理参数 gather）
        - ZeRO-2:
          - 单模型 loss: 使用 model.backward() 初始化优化器状态
          - 跨模型 loss（cross_model=True）: 使用 loss.backward()

        Args:
            loss: 损失值
            retain_graph: 是否保留计算图
            model: 计算 loss 的模型（多模型训练时必须提供）
            cross_model: 是否是跨模型 loss（如 TD3 的 actor_loss = -critic.Q1(s, actor(s))）
        """
        zero_stage = get_zero_stage()

        if zero_stage == 3:
            # 🔧 ZeRO-3: 直接使用 loss.backward()
            # DeepSpeed 在 accelerator.prepare() 时注册了 backward hook
            # 这些 hook 会自动处理参数 gather/partition
            loss.backward(retain_graph=retain_graph)
        elif zero_stage == 2:
            # 🔧 ZeRO-2 多模型训练策略：
            # - 单模型 loss（如 critic_loss = MSE(critic(s,a), target)）: 使用 model.backward()
            # - 跨模型 loss（如 actor_loss = -critic.Q1(s, actor(s))）: 使用 loss.backward()
            #
            # 原因：跨模型 loss 的计算图涉及多个 DeepSpeedEngine
            # 调用 model.backward() 时，model 的优化器期望立即看到自己的参数
            # 但跨模型 loss 中，其他模型的参数可能先被遇到，导致 ipg_buffer 初始化问题

            if cross_model:
                # 🔧 跨模型 loss: 直接使用 loss.backward()
                # PyTorch autograd 会正确处理跨模型的梯度流动
                loss.backward(retain_graph=retain_graph)
            elif model is not None and hasattr(model, 'backward'):
                # 单模型 loss: 使用 DeepSpeedEngine 的 backward 方法
                model.backward(loss)
            elif model is not None and hasattr(model, 'module') and hasattr(model.module, 'backward'):
                # DataParallel 包装的 DeepSpeedEngine
                model.module.backward(loss)
            elif self.accelerator is not None:
                self.accelerator.backward(loss, retain_graph=retain_graph)
            else:
                loss.backward(retain_graph=retain_graph)
        elif model is not None and hasattr(model, 'backward'):
            # 非 ZeRO 模式的 DeepSpeed：使用模型自己的 backward
            model.backward(loss)
        elif model is not None and hasattr(model, 'module') and hasattr(model.module, 'backward'):
            # DataParallel 包装的 DeepSpeedEngine
            model.module.backward(loss)
        elif self.accelerator is not None:
            # 单模型训练，使用 accelerator.backward()
            self.accelerator.backward(loss, retain_graph=retain_graph)
        else:
            loss.backward(retain_graph=retain_graph)

    def should_sync_gradients(self, accum_step: int, total_accum_steps: int) -> bool:
        """
        检查是否应该同步梯度（梯度累积时使用）

        在梯度累积的最后一步才进行同步和优化器更新

        Args:
            accum_step: 当前累积步骤（0-indexed）
            total_accum_steps: 总累积步数

        Returns:
            是否应该同步
        """
        return (accum_step + 1) == total_accum_steps

    def clip_grad_norm(
        self,
        model: nn.Module,
        max_norm: float = 1.0
    ) -> float:
        """
        梯度裁剪（ZeRO 兼容）

        Args:
            model: 模型
            max_norm: 最大梯度范数

        Returns:
            裁剪前的梯度范数
        """
        if self.accelerator is not None:
            # accelerator 会自动处理 ZeRO-3 的分布式梯度裁剪
            return self.accelerator.clip_grad_norm_(model.parameters(), max_norm)
        else:
            return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

    def unwrap_model(self, model: nn.Module) -> nn.Module:
        """
        解包模型（ZeRO 兼容）

        在 ZeRO-3 中，模型参数被分片，需要用这个方法获取完整模型

        Args:
            model: 包装后的模型

        Returns:
            解包后的原始模型
        """
        if self.accelerator is not None:
            return self.accelerator.unwrap_model(model)
        elif isinstance(model, torch.nn.DataParallel):
            return model.module
        elif hasattr(model, 'module'):
            return model.module
        return model

    def gather_for_metrics(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        收集所有进程的张量用于计算指标（ZeRO 兼容）

        Args:
            tensor: 本地张量

        Returns:
            所有进程的张量（在主进程）
        """
        if self.accelerator is not None:
            return self.accelerator.gather_for_metrics(tensor)
        return tensor

    def wait_for_everyone(self):
        """等待所有进程（同步点）"""
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
        else:
            from rlft.ddp_utils import barrier
            barrier()

    def is_main_process(self) -> bool:
        """检查是否是主进程"""
        if self.accelerator is not None:
            return self.accelerator.is_main_process
        from rlft.ddp_utils import is_main_process
        return is_main_process()

    def print(self, *args, **kwargs):
        """只在主进程打印"""
        if self.accelerator is not None:
            self.accelerator.print(*args, **kwargs)
        else:
            from rlft.ddp_utils import rank0_print
            rank0_print(*args, **kwargs)

    def save_state(self, output_dir: str):
        """
        保存训练状态（ZeRO 兼容）

        ZeRO-3 需要特殊处理：收集所有分片的参数
        """
        if self.accelerator is not None:
            self.accelerator.save_state(output_dir)

    def load_state(self, input_dir: str):
        """
        加载训练状态（ZeRO 兼容）
        """
        if self.accelerator is not None:
            self.accelerator.load_state(input_dir)


# ============================================================
# ZeRO-3 兼容的保存/加载工具
# ============================================================

def maybe_zero_3(param: torch.Tensor) -> torch.Tensor:
    """
    ZeRO-3 兼容的参数提取

    在 ZeRO-3 中，参数被分片到不同 GPU，有 ds_id 属性。
    需要使用 GatheredParameters 收集后才能访问完整参数。

    Args:
        param: 参数张量（可能是 ZeRO-3 分片的）

    Returns:
        完整的参数张量（在 CPU 上）
    """
    if hasattr(param, "ds_id"):
        # ZeRO-3 分片参数
        import deepspeed
        with deepspeed.zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_peft_state_maybe_zero_3(named_params, bias: str = "none") -> dict:
    """
    ZeRO-3 兼容的 LoRA 参数提取

    Args:
        named_params: model.named_parameters() 的迭代器
        bias: LoRA bias 模式 ("none", "all", "lora_only")

    Returns:
        LoRA 参数的 state_dict（在 CPU 上）
    """
    # 首先收集所有需要保存的参数（不触发 gather）
    # 使用 list 确保迭代器被完全消耗，并保持顺序一致
    all_named_params = list(named_params)

    if bias == "none":
        to_return = [(k, t) for k, t in all_named_params if "lora_" in k]
    elif bias == "all":
        to_return = [(k, t) for k, t in all_named_params if "lora_" in k or "bias" in k]
    elif bias == "lora_only":
        lora_params = []
        maybe_lora_bias = []
        lora_bias_names = set()
        for k, t in all_named_params:
            if "lora_" in k:
                lora_params.append((k, t))
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias.append((k, t))
        to_return = lora_params + [(k, t) for k, t in maybe_lora_bias if k in lora_bias_names]
    else:
        raise NotImplementedError(f"Unknown bias mode: {bias}")

    if not to_return:
        return {}

    # 检查是否有 ZeRO-3 分片参数
    has_zero3_params = any(hasattr(t, "ds_id") for _, t in to_return)

    if has_zero3_params:
        # ZeRO-3: 一次性收集所有参数（单次集体操作，避免死锁）
        import deepspeed

        # 确保参数列表顺序一致（按名称排序）
        to_return_sorted = sorted(to_return, key=lambda x: x[0])
        all_params = [t for _, t in to_return_sorted]

        with deepspeed.zero.GatheredParameters(all_params, modifier_rank=None):
            # 在 gather 上下文中，所有参数都可以访问完整数据
            result = {k: t.data.detach().cpu().clone() for k, t in to_return_sorted}
        return result
    else:
        # 非 ZeRO-3：直接 clone
        return {k: t.detach().cpu().clone() for k, t in to_return}


def save_module_zero3(
    module: nn.Module,
    save_path: str,
    local_rank: int = 0
) -> bool:
    """
    ZeRO-3 兼容的模块保存

    在 ZeRO-3 中，参数被分片，需要先 gather 再保存。

    Args:
        module: 要保存的 PyTorch 模块
        save_path: 保存路径
        local_rank: 当前进程的 local rank

    Returns:
        是否成功保存（只有 rank 0 返回 True）
    """
    import os
    from rlft.ddp_utils import rank0_print

    zero_stage = get_zero_stage()

    if zero_stage == 3:
        # ZeRO-3: 需要 gather 参数
        import deepspeed

        rank0_print(f"[ZeRO-3 Save] Gathering parameters from all GPUs...")

        # 收集所有参数到 rank 0
        with deepspeed.zero.GatheredParameters(
            list(module.parameters()),
            modifier_rank=0
        ):
            if local_rank == 0 or local_rank == -1:
                # 只在 rank 0 保存
                state_dict = module.state_dict()

                # 关键：clone 每个 tensor 避免保存整个 buffer
                # ZeRO-3 的 GatheredParameters 创建一个大 buffer
                # state_dict 中的 tensor 是这个 buffer 的 view
                # 直接保存会保存整个 buffer（很大）
                state_dict_clean = {k: v.clone() for k, v in state_dict.items()}

                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(state_dict_clean, save_path)

                # 统计
                num_params = len(state_dict_clean)
                total_values = sum(p.numel() for p in state_dict_clean.values())
                file_size = os.path.getsize(save_path) / 1024 / 1024
                rank0_print(f"[ZeRO-3 Save] ✓ Saved {num_params} tensors, {total_values:,} values, {file_size:.2f} MB")
                return True
        return False
    else:
        # ZeRO-2 或普通模式：直接保存
        if local_rank == 0 or local_rank == -1:
            state_dict = module.state_dict()
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(state_dict, save_path)
            return True
        return False


def save_lora_zero3(
    peft_model,
    save_dir: str,
    local_rank: int = 0
) -> bool:
    """
    ZeRO-3 兼容的 LoRA 保存

    Args:
        peft_model: PEFT 包装的模型
        save_dir: 保存目录
        local_rank: 当前进程的 local rank

    Returns:
        是否成功保存
    """
    import os
    from rlft.ddp_utils import rank0_print

    zero_stage = get_zero_stage()

    if zero_stage == 3:
        rank0_print(f"[ZeRO-3 Save] Gathering LoRA parameters...")

        if local_rank == 0 or local_rank == -1:
            # 使用 get_peft_state_maybe_zero_3 收集 LoRA 参数
            state_dict = get_peft_state_maybe_zero_3(
                peft_model.named_parameters(),
                bias="none"
            )

            # 使用 PEFT 的保存方法，传入收集好的 state_dict
            peft_model.save_pretrained(save_dir, state_dict=state_dict)

            num_params = len(state_dict)
            total_values = sum(p.numel() for p in state_dict.values())
            rank0_print(f"[ZeRO-3 Save] ✓ LoRA: {num_params} tensors, {total_values:,} values")

            if total_values == 0:
                rank0_print(f"[ZeRO-3 Save] ⚠️ WARNING: All LoRA parameters are empty!")

            return True
        return False
    else:
        # ZeRO-2 或普通模式
        if local_rank == 0 or local_rank == -1:
            peft_model.save_pretrained(save_dir)
            return True
        return False


def create_accelerator_for_rlft(
    gradient_accumulation_steps: int = 1,
    mixed_precision: str = "bf16",
    zero_stage: int = 2,
    deepspeed_config_file: Optional[str] = None,
    batch_size: int = 1
) -> Any:
    """
    为 RLFT 创建 Accelerator

    Args:
        gradient_accumulation_steps: 梯度累积步数
        mixed_precision: 混合精度类型 ("no", "fp16", "bf16")
        zero_stage: ZeRO 优化阶段 (0, 2, 3)
        deepspeed_config_file: DeepSpeed 配置文件路径（优先级最高）
        batch_size: 每个 GPU 的 micro batch size

    Returns:
        Accelerator 实例
    """
    # 在 import deepspeed 之前禁用冗余日志
    import logging
    deepspeed_loggers = [
        "deepspeed",
        "deepspeed.runtime.config",
        "deepspeed.runtime.zero.config",
        "deepspeed.runtime.zero.stage3",
        "deepspeed.runtime.zero.stage_1_and_2",
        "deepspeed.runtime.utils",
        "deepspeed.utils",
        "deepspeed.runtime.engine",
        "deepspeed.runtime.zero.partition_parameters",
        "deepspeed.runtime.zero.partitioned_param_coordinator",
    ]
    for logger_name in deepspeed_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    from accelerate import Accelerator
    from accelerate.utils import DeepSpeedPlugin

    # 如果指定了配置文件，直接使用
    if deepspeed_config_file is not None:
        deepspeed_plugin = DeepSpeedPlugin(
            hf_ds_config=deepspeed_config_file,
            gradient_accumulation_steps=gradient_accumulation_steps
        )
    else:
        # 动态创建 DeepSpeed 配置（使用 hf_ds_config 传递完整配置）
        ds_config = {
            "train_micro_batch_size_per_gpu": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "gradient_clipping": 0.5,
            "bf16": {"enabled": mixed_precision == "bf16"},
            "fp16": {"enabled": mixed_precision == "fp16"},
            "zero_optimization": {
                "stage": zero_stage,
                "reduce_bucket_size": 5e7,
            },
            "zero_allow_untested_optimizer": True,
            # 禁用 DeepSpeed 冗余日志
            "steps_per_print": float("inf"),  # 禁用周期性打印
            "wall_clock_breakdown": False,
            "dump_state": False,  # 禁用详细配置打印
        }

        if zero_stage == 2:
            ds_config["zero_optimization"].update({
                "overlap_comm": True,
                "contiguous_gradients": True,
                "allgather_bucket_size": 5e8,
                "reduce_scatter": True,
            })

        if zero_stage == 3:
            # ZeRO-3 多模型训练：禁用可能导致问题的特性
            # 关键：round_robin_gradients 和 reduce_scatter 在多模型时会导致 ds_id 冲突
            ds_config["zero_optimization"].update({
                "overlap_comm": False,  # 禁用通信重叠，避免参数追踪问题
                "contiguous_gradients": False,  # 禁用连续梯度
                "reduce_scatter": False,  # 🔧 禁用 reduce_scatter，使用 allreduce
                "round_robin_gradients": False,  # 🔧 禁用，避免多模型时 ds_id 重复
                "stage3_prefetch_bucket_size": 0,  # 🔧 禁用预取，避免参数追踪冲突
                "stage3_param_persistence_threshold": 1e9,  # 🔧 高阈值，让更多参数持久化
                "stage3_max_live_parameters": 1e9,  # 🔧 增大限制
                "stage3_max_reuse_distance": 1e9,  # 🔧 增大限制
                "stage3_gather_16bit_weights_on_model_save": True,
                "sub_group_size": 1e9,  # 🔧 大值禁用子组分割，避免 IPG 冲突
            })

        # 使用 hf_ds_config 传递完整的 DeepSpeed 配置
        deepspeed_plugin = DeepSpeedPlugin(
            hf_ds_config=ds_config,
            zero3_init_flag=(zero_stage == 3),
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        deepspeed_plugin=deepspeed_plugin,
    )

    # 在 accelerator 创建后，强制禁用 DeepSpeed 的冗余日志
    import logging
    for logger_name in [
        "deepspeed",
        "deepspeed.runtime.config",
        "deepspeed.runtime.zero.config",
        "deepspeed.runtime.zero.stage3",
        "deepspeed.runtime.zero.stage_1_and_2",
        "deepspeed.runtime.utils",
        "deepspeed.utils",
    ]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    return accelerator
