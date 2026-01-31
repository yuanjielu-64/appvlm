"""
DDP (Distributed Data Parallel) 工具函数

提供分布式训练相关的辅助函数，包括：
- 进程检测（主进程/非主进程）
- 只在主进程打印的 rank0_print
- 分布式同步工具
- 手动梯度同步（解决 4-bit 量化模型无法使用 DDP 的问题）
"""
import os


def is_ddp_enabled():
    """检测是否在 DDP 模式下运行"""
    return int(os.environ.get("WORLD_SIZE", 1)) > 1


def get_local_rank():
    """获取当前进程的 LOCAL_RANK"""
    return int(os.environ.get("LOCAL_RANK", 0))


def get_world_size():
    """获取总进程数"""
    return int(os.environ.get("WORLD_SIZE", 1))


def is_main_process():
    """
    判断是否为主进程

    使用多种方式检测，优先级：
    1. torch.distributed.get_rank()（最可靠）
    2. RANK 环境变量
    3. LOCAL_RANK 环境变量
    """
    # 方式1：使用 torch.distributed（最可靠）
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            return dist.get_rank() == 0
    except:
        pass

    # 方式2：检查 RANK 环境变量（torchrun 设置）
    rank = os.environ.get("RANK")
    if rank is not None:
        return int(rank) == 0

    # 方式3：检查 LOCAL_RANK 环境变量
    return get_local_rank() == 0


def rank0_print(*args, **kwargs):
    """只在主进程打印"""
    if is_main_process():
        print(*args, **kwargs)


def barrier():
    """同步所有进程（如果在 DDP 模式下）"""
    if is_ddp_enabled():
        import torch.distributed as dist
        if dist.is_initialized():
            dist.barrier()


def sync_gradients(model, verbose=False):
    """
    手动同步模型梯度（All-Reduce）

    用于 4-bit 量化模型无法使用 accelerator.prepare(model) 的情况。
    在 optimizer.step() 之前调用，确保所有 GPU 上的梯度一致。

    Args:
        model: PyTorch 模型
        verbose: 是否打印同步信息

    Returns:
        int: 同步的参数数量
    """
    if not is_ddp_enabled():
        return 0

    import torch
    import torch.distributed as dist

    if not dist.is_initialized():
        return 0

    world_size = get_world_size()
    synced_count = 0

    for name, param in model.named_parameters():
        if param.grad is not None:
            # All-Reduce: 求所有 GPU 的梯度平均值
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= world_size
            synced_count += 1

    if verbose and is_main_process():
        print(f"[DDP] Synced gradients for {synced_count} parameters")

    return synced_count


def sync_model_params(model, src_rank=0, verbose=False):
    """
    从指定 rank 广播模型参数到所有进程

    用于确保所有 GPU 上的模型参数一致（例如训练开始前）。

    Args:
        model: PyTorch 模型
        src_rank: 源 rank（默认从 rank 0 广播）
        verbose: 是否打印同步信息

    Returns:
        int: 同步的参数数量
    """
    if not is_ddp_enabled():
        return 0

    import torch.distributed as dist

    if not dist.is_initialized():
        return 0

    synced_count = 0

    for name, param in model.named_parameters():
        dist.broadcast(param.data, src=src_rank)
        synced_count += 1

    if verbose and is_main_process():
        print(f"[DDP] Broadcast {synced_count} parameters from rank {src_rank}")

    return synced_count
