"""
FTRL训练脚本 - VLM+DPT + TD3 (Condor离线模式)

优化版本:
1. 自动启动/关闭 tmux VLM 服务
2. 深度配置合并（buffer config + VLM config）
3. 服务健康检查
4. GPU 智能分配
5. 完整的信号处理和清理

纯Python 3.10环境，不依赖gym/ROS
完全模仿td3/train.py的Condor模式，使用InfoEnv + CondorCollector
"""
# ============ GPU选择必须在import torch之前 ============
import sys
import os

# ============ 禁用冗余日志（必须最早设置）============
_local_rank = int(os.environ.get("LOCAL_RANK", 0))

# 所有进程都禁用 DeepSpeed 的冗余 config 打印
# 只保留 WARNING 和 ERROR
os.environ["DEEPSPEED_LOG_LEVEL"] = "WARNING"

if _local_rank != 0:
    # 非主进程：禁用所有日志
    os.environ["DEEPSPEED_LOG_LEVEL"] = "ERROR"
    # 禁用 transformers 日志
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    # 禁用 HuggingFace 进度条
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    # 禁用 accelerate 日志
    os.environ["ACCELERATE_LOG_LEVEL"] = "ERROR"
    # 禁用一般日志
    import logging
    logging.getLogger("deepspeed").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("accelerate").setLevel(logging.ERROR)
else:
    # 主进程：只禁用 DeepSpeed 的冗余 config 打印，保留重要信息
    import logging
    logging.getLogger("deepspeed").setLevel(logging.WARNING)
    # 禁用 DeepSpeed config 详细打印
    logging.getLogger("deepspeed.runtime.config").setLevel(logging.WARNING)
    logging.getLogger("deepspeed.runtime.zero.config").setLevel(logging.WARNING)

# 配置文件简写映射（全局常量）
CONFIG_SHORTCUTS = {
    'ddp': 'ftrl_vlm_ddp',
    'DDP': 'ftrl_vlm_ddp',
    'dwa': 'ftrl_vlm_dwa',
    'DWA': 'ftrl_vlm_dwa',
    'teb': 'ftrl_vlm_teb',
    'TEB': 'ftrl_vlm_teb',
    'mppi': 'ftrl_vlm_mppi',
    'MPPI': 'ftrl_vlm_mppi',
}

def _early_gpu_setup():
    """在import torch之前设置CUDA_VISIBLE_DEVICES

    优先级：命令行 --gpu > 配置文件 gpu_config.cuda_visible_devices
    """
    # DDP 模式：只有主进程打印（LOCAL_RANK 由 torchrun 在脚本启动前设置）
    is_rank0 = int(os.environ.get("LOCAL_RANK", 0)) == 0

    def _print(*args, **kwargs):
        if is_rank0:
            print(*args, **kwargs)  # 使用内置 print，因为此时 rank0_print 还未导入

    # 1. 检查命令行参数（最高优先级）
    for i, arg in enumerate(sys.argv):
        if arg == '--gpu' and i + 1 < len(sys.argv):
            gpu_id = sys.argv[i + 1]
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
            _print(f"[GPU] 设置 CUDA_VISIBLE_DEVICES={gpu_id} (命令行参数)")
            return gpu_id

    # 2. 尝试从配置文件读取
    try:
        import yaml
        # 解析配置文件路径（默认值）
        config_path = "../script/ft_qwen/configs/"
        config_file = "ftrl_vlm_ddp"

        for i, arg in enumerate(sys.argv):
            if arg == '--config_path' and i + 1 < len(sys.argv):
                config_path = sys.argv[i + 1]
            if arg == '--config_file' and i + 1 < len(sys.argv):
                config_file = sys.argv[i + 1]
                # 应用简写映射
                config_file = CONFIG_SHORTCUTS.get(config_file, config_file)

        _print(f"[GPU] Debug: config_path={config_path}, config_file={config_file}")

        if config_path and config_file:
            config_file_path = os.path.join(config_path, config_file + '.yaml')
            _print(f"[GPU] Debug: 尝试读取配置文件: {config_file_path}")
            _print(f"[GPU] Debug: 文件存在? {os.path.exists(config_file_path)}")

            if os.path.exists(config_file_path):
                with open(config_file_path, 'r') as f:
                    config = yaml.safe_load(f)

                # 读取 gpu_config
                gpu_config = config.get('gpu_config', {})
                cuda_visible_devices = gpu_config.get('cuda_visible_devices')

                _print(f"[GPU] Debug: gpu_config={gpu_config}")
                _print(f"[GPU] Debug: cuda_visible_devices={cuda_visible_devices}")

                if cuda_visible_devices:
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
                    _print(f"[GPU] 设置 CUDA_VISIBLE_DEVICES={cuda_visible_devices} (配置文件)")
                    return cuda_visible_devices
                else:
                    _print(f"[GPU] Debug: cuda_visible_devices 为空或 None")
            else:
                _print(f"[GPU] Debug: 配置文件不存在")
        else:
            _print(f"[GPU] Debug: config_path 或 config_file 未提供")
    except Exception as e:
        import traceback
        _print(f"[GPU] 警告：从配置文件读取GPU设置失败: {e}")
        _print(f"[GPU] 详细错误:\n{traceback.format_exc()}")

    _print(f"[GPU] 未设置 CUDA_VISIBLE_DEVICES，使用系统默认")
    return None

_EARLY_GPU = _early_gpu_setup()
# ======================================================

import argparse
import GPUtil
import yaml
import numpy as np
from datetime import datetime, timedelta
from os.path import join, dirname, abspath, exists
import shutil
import logging
import collections
import time
from pprint import pformat
import signal

import torch
import wandb

# 添加路径
sys.path.append(dirname(dirname(abspath(__file__))))

# RLFT模块 (VLM专用)
from rlft.vlm_net import VLM_DPT_FeatureExtractor, VLM_DPT_Actor, VLM_DPT_Critic
from rlft.rl import TD3
from rlft.collector import VLMCondorCollector, VLMReplayBuffer

# Accelerate 支持多卡 DDP 训练
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

# DDP 工具函数
from rlft.ddp_utils import is_ddp_enabled, get_local_rank, get_world_size, is_main_process, rank0_print, sync_model_params

# ZeRO 兼容工具（支持 ZeRO-2 和 ZeRO-3）
from rlft.zero_utils import ZeROCompatibleTrainer, is_deepspeed_enabled, get_zero_stage, create_accelerator_for_rlft

torch.set_num_threads(8)

import psutil
import subprocess
import atexit

# 全局变量：跟踪启动的 tmux 服务
TMUX_SERVICES_STARTED = []
PLANNER = "ddp"  # 默认值，会在 main 中更新


def start_tmux_services(config, policy_checkpoint_path=None):
    """
    根据配置自动启动 tmux VLM 服务

    Args:
        config: 包含 ftrl_config 的配置字典
        policy_checkpoint_path: 直接指定 policy checkpoint 路径（与 rl.py 保持一致）

    Returns:
        list: 启动的 tmux session 名称列表
    """
    global TMUX_SERVICES_STARTED, PLANNER

    ftrl_config = config.get("ftrl_config", {}).copy()
    training_config = config.get("training_config", {})

    # 从 gpu_config 读取 inference_gpu_strategy（如果存在）
    gpu_config = config.get("gpu_config", {})
    if "inference_gpu_strategy" in gpu_config:
        ftrl_config["gpu_strategy"] = gpu_config["inference_gpu_strategy"]
        rank0_print(f"    >>>> Using inference_gpu_strategy from gpu_config: {gpu_config['inference_gpu_strategy']}")

    # 检查是否自动启动
    if not ftrl_config.get("auto_start_services", False):
        rank0_print("    >>>> auto_start_services=False, 跳过自动启动服务")
        return []

    # 获取配置参数
    server_urls = ftrl_config.get("server_urls", [])
    num_services = len(server_urls) if server_urls else ftrl_config.get("num_services", 1)
    start_port = ftrl_config.get("start_port", 7000)
    base_model = ftrl_config.get("base_model", "Qwen/Qwen2.5-VL-3B-Instruct")

    # lora_path：优先使用传入的路径（与 rl.py save/load 保持一致）
    if policy_checkpoint_path:
        lora_path = policy_checkpoint_path
        rank0_print(f"    >>>> 使用传入的 checkpoint 路径: {lora_path}")
    elif 'BUFFER_PATH' in globals() and BUFFER_PATH:
        # 回退：从全局变量构建路径
        lora_path = join(BUFFER_PATH, "policy")
        rank0_print(f"    >>>> [回退] 使用全局 BUFFER_PATH 构建: {lora_path}")
    else:
        rank0_print("[ERROR] 未提供 checkpoint 路径且 BUFFER_PATH 未设置")
        return []

    # 验证 checkpoint 是否存在
    if not exists(lora_path):
        rank0_print(f"[ERROR] Checkpoint 目录不存在: {lora_path}")
        rank0_print(f"[ERROR] 请确保先运行 policy.save() 保存模型")
        return []

    # 检查必需文件并显示详情
    rank0_print(f"    >>>> 验证 checkpoint 内容:")
    checkpoint_files = {
        "LoRA": join(lora_path, "adapter_model.safetensors"),
        "LoRA Config": join(lora_path, "adapter_config.json"),
        "Regression Head": join(lora_path, "regression_head", "pytorch_model.bin"),
        "History Config": join(lora_path, "history_config.json"),
        "History Encoder": join(lora_path, "history_encoder", "pytorch_model.bin"),
        "Normalization": join(lora_path, "normalization", "param_mean.npy"),
    }

    all_ok = True
    for name, filepath in checkpoint_files.items():
        if exists(filepath):
            if name == "LoRA":
                size_mb = os.path.getsize(filepath) / 1024 / 1024
                rank0_print(f"         ✓ {name}: {size_mb:.1f} MB")
            else:
                rank0_print(f"         ✓ {name}")
        else:
            rank0_print(f"         ✗ {name} (缺失)")
            if name in ["LoRA", "Regression Head", "History Config", "Normalization"]:
                all_ok = False

    if not all_ok:
        rank0_print(f"[ERROR] Checkpoint 缺少必需文件，无法启动服务")
        return []

    # 使用全局 PLANNER（已在 main 中设置）
    planner = PLANNER

    rank0_print("\n" + "=" * 60)
    rank0_print("  启动 FTRL VLM 服务 (tmux)")
    rank0_print("=" * 60)
    rank0_print(f"  Planner: {planner}")
    rank0_print(f"  Number of services: {num_services}")
    rank0_print(f"  Start port: {start_port}")
    rank0_print(f"  Base model: {base_model}")
    rank0_print(f"  LoRA path: {lora_path}")

    # 获取脚本目录
    script_dir = dirname(dirname(abspath(__file__)))
    ft_qwen_dir = join(script_dir, "script", "ft_qwen")
    server_script = join(ft_qwen_dir, "qwen_server.py")

    # 检查 server 脚本是否存在
    if not exists(server_script):
        rank0_print(f"[ERROR] qwen_server.py not found: {server_script}")
        return []

    # 检查 LoRA checkpoint 是否存在
    if not exists(lora_path):
        rank0_print(f"[ERROR] LoRA checkpoint not found: {lora_path}")
        return []

    # ===== 新的GPU分配逻辑：支持每个服务单独指定GPU =====
    # 优先使用 gpu_assignments（列表），否则使用 gpu_strategy（全局策略）
    gpu_array = []

    if "gpu_assignments" in ftrl_config:
        # 方式1: 每个服务单独指定GPU（推荐）
        gpu_assignments = ftrl_config["gpu_assignments"]

        if not isinstance(gpu_assignments, list):
            rank0_print(f"[ERROR] gpu_assignments must be a list, got {type(gpu_assignments)}")
            return []

        if len(gpu_assignments) != num_services:
            rank0_print(f"[ERROR] gpu_assignments length ({len(gpu_assignments)}) must match num_services ({num_services})")
            return []

        gpu_array = [int(g) for g in gpu_assignments]
        rank0_print(f"  GPU assignments (per service):")
        for i, gpu in enumerate(gpu_array):
            rank0_print(f"    Service {i} -> GPU {gpu}")

    elif "gpu_strategy" in ftrl_config:
        # 方式2: 使用全局策略（向后兼容）
        gpu_strategy = str(ftrl_config["gpu_strategy"])

        if gpu_strategy == "auto":
            # 自动递增分配
            try:
                available_gpus = GPUtil.getAvailable(order='memory', limit=8, maxLoad=0.5, maxMemory=0.5)
                if len(available_gpus) < num_services:
                    rank0_print(f"  [WARN] Only {len(available_gpus)} GPUs available, requested {num_services}")
                    # 循环使用可用 GPU
                    gpu_array = [available_gpus[i % len(available_gpus)] for i in range(num_services)]
                else:
                    gpu_array = available_gpus[:num_services]
                rank0_print(f"  Auto-detected GPUs: {gpu_array}")
            except Exception as e:
                rank0_print(f"  [ERROR] Failed to auto-detect GPUs: {e}")
                rank0_print(f"  Falling back to GPU 0")
                gpu_array = [0] * num_services
        elif "," in gpu_strategy:
            gpu_array = [int(g.strip()) for g in gpu_strategy.split(",")]
            if len(gpu_array) != num_services:
                rank0_print(f"[ERROR] GPU count ({len(gpu_array)}) must match num_services ({num_services})")
                return []
            rank0_print(f"  GPU strategy: {gpu_array}")
        else:
            # 单个GPU，所有服务共享
            gpu_array = [int(gpu_strategy)] * num_services
            rank0_print(f"  All services will share GPU {gpu_strategy}")
    else:
        # 默认: 所有服务使用GPU 0
        gpu_array = [0] * num_services
        rank0_print(f"  Default: All services will use GPU 0")

    rank0_print("=" * 60)

    # inference_gpu_strategy 直接使用物理 GPU 编号，无需转换
    rank0_print(f"  Inference GPUs (physical): {gpu_array}")

    # Conda Python 路径 (根据环境调整)
    conda_python_candidates = [
        "/common/home/yl2832/miniconda3/envs/lmms-finetune-qwen/bin/python",
        "/home/yuanjielu/miniforge3/envs/lmms-finetune-qwen/bin/python",
        os.path.expanduser("~/miniconda3/envs/lmms-finetune-qwen/bin/python"),
        os.path.expanduser("~/miniforge3/envs/lmms-finetune-qwen/bin/python"),
    ]
    conda_python = None
    for candidate in conda_python_candidates:
        if exists(candidate):
            conda_python = candidate
            break

    if conda_python is None:
        rank0_print("[ERROR] Cannot find lmms-finetune-qwen conda environment")
        rank0_print("  Tried:", conda_python_candidates)
        return []

    rank0_print(f"  Using Python: {conda_python}")

    started_sessions = []

    for i in range(num_services):
        gpu = gpu_array[i]
        port = start_port + i
        session_name = f"ftrl_{planner}_{i}"

        rank0_print(f"\n  Starting service {i+1}/{num_services}...")
        rank0_print(f"    Name: {session_name}")
        rank0_print(f"    GPU: {gpu}")
        rank0_print(f"    Port: {port}")

        # 检查端口是否被占用
        try:
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"],
                capture_output=True, text=True, timeout=5
            )
            if result.stdout.strip():
                rank0_print(f"    [WARN] Port {port} is in use, killing process...")
                subprocess.run(["kill", "-9"] + result.stdout.strip().split(), timeout=5)
                time.sleep(1)
        except Exception as e:
            pass  # lsof 可能不存在或端口未被占用

        # 检查 tmux session 是否已存在
        try:
            result = subprocess.run(
                ["tmux", "has-session", "-t", session_name],
                capture_output=True, timeout=5
            )
            if result.returncode == 0:
                rank0_print(f"    [WARN] tmux session '{session_name}' already exists, killing it...")
                subprocess.run(["tmux", "kill-session", "-t", session_name], timeout=5)
                time.sleep(1)
        except Exception as e:
            pass

        # 构建启动命令
        cmd = f"""export CUDA_VISIBLE_DEVICES={gpu} && \
{conda_python} {server_script} \
    --base_model {base_model} \
    --lora_path {lora_path} \
    --algorithm {planner.upper()} \
    --port {port} \
    --device cuda:0 \
    --load_in_4bit; \
echo 'Service stopped. Press Enter to exit.'; read"""

        # 创建 tmux session
        try:
            subprocess.run(
                ["tmux", "new-session", "-d", "-s", session_name, cmd],
                check=True, timeout=10
            )
            started_sessions.append(session_name)
            rank0_print(f"    [OK] Started in tmux session '{session_name}'")
        except subprocess.CalledProcessError as e:
            rank0_print(f"    [ERROR] Failed to start tmux session: {e}")
        except subprocess.TimeoutExpired:
            rank0_print(f"    [ERROR] Timeout starting tmux session")

        # 等待一小段时间，避免同时启动太多进程
        time.sleep(3)

    TMUX_SERVICES_STARTED = started_sessions

    rank0_print("\n" + "=" * 60)
    rank0_print(f"  Started {len(started_sessions)}/{num_services} service(s)")
    rank0_print("=" * 60)

    # 等待服务启动完成
    if started_sessions:
        rank0_print("\n  等待服务启动...")
        wait_time = 25  # 等待30秒让服务初始化
        for i in range(wait_time):
            rank0_print(f"\r  等待中... {wait_time - i}s ", end="", flush=True)
            time.sleep(1)
        rank0_print("\n  服务应该已经启动完成")

    return started_sessions


def stop_tmux_services(session_names=None, planner=None):
    """
    关闭 tmux VLM 服务

    Args:
        session_names: 要关闭的 session 名称列表，None 则关闭所有 TMUX_SERVICES_STARTED
        planner: 如果指定，关闭所有匹配 ftrl_{planner}_* 的 session
    """
    global TMUX_SERVICES_STARTED, PLANNER

    if session_names is None:
        session_names = TMUX_SERVICES_STARTED.copy()

    if planner is None:
        planner = PLANNER

    if not session_names and not planner:
        rank0_print("    >>>> 没有需要关闭的 tmux 服务")
        return

    rank0_print("\n" + "=" * 60)
    rank0_print("  关闭 FTRL VLM 服务 (tmux)")
    rank0_print("=" * 60)

    closed_count = 0

    # 关闭指定的 session
    for session_name in session_names:
        try:
            result = subprocess.run(
                ["tmux", "has-session", "-t", session_name],
                capture_output=True, timeout=5
            )
            if result.returncode == 0:
                subprocess.run(
                    ["tmux", "kill-session", "-t", session_name],
                    timeout=5
                )
                rank0_print(f"  [OK] Closed tmux session: {session_name}")
                closed_count += 1
            else:
                rank0_print(f"  [SKIP] Session not found: {session_name}")
        except Exception as e:
            rank0_print(f"  [ERROR] Failed to close {session_name}: {e}")

    # 额外检查：关闭所有 ftrl_{planner}_* 的 session
    if planner:
        try:
            result = subprocess.run(
                ["tmux", "ls"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if line and f"ftrl_{planner}_" in line:
                        session = line.split(":")[0]
                        if session not in session_names:
                            try:
                                subprocess.run(
                                    ["tmux", "kill-session", "-t", session],
                                    timeout=5
                                )
                                rank0_print(f"  [OK] Closed extra tmux session: {session}")
                                closed_count += 1
                            except:
                                pass
        except:
            pass

    TMUX_SERVICES_STARTED.clear()

    rank0_print(f"\n  总共关闭了 {closed_count} 个 tmux session")
    rank0_print("=" * 60)


def restart_gazebo():
    rank0_print(">>>>>>>> 正在重启Gazebo...")

    gazebo_processes = ['gazebo', 'gzserver', 'gzclient', 'roslaunch']

    for proc_name in gazebo_processes:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['name'] == proc_name or \
                        (proc.info['cmdline'] and any(proc_name in cmd for cmd in proc.info['cmdline'])):
                    rank0_print(f"    >>>> 正在杀死进程: {proc.info['name']} (PID: {proc.info['pid']})")
                    proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass


def deep_merge(base_dict, override_dict, path="", verbose=True):
    """
    深度合并两个字典，override_dict 中的值会覆盖 base_dict 中的相同键

    Args:
        base_dict: 基础字典（会被修改）
        override_dict: 覆盖字典
        path: 当前路径（用于日志输出）
        verbose: 是否打印详细日志

    Returns:
        合并后的字典（就是修改后的 base_dict）
    """
    for key, value in override_dict.items():
        current_path = f"{path}.{key}" if path else key

        if key in base_dict:
            if isinstance(base_dict[key], dict) and isinstance(value, dict):
                # 两个都是字典，递归合并
                deep_merge(base_dict[key], value, current_path, verbose)
            elif base_dict[key] != value:
                # 值不同，用 override 覆盖
                old_value = base_dict[key]
                base_dict[key] = value
                if verbose:
                    # 只打印重要的配置变更
                    if not isinstance(value, (list, dict)) or len(str(value)) < 100:
                        rank0_print(f"    >>>> Override {current_path}: {old_value} -> {value}")
                    else:
                        rank0_print(f"    >>>> Override {current_path}: [complex value]")
        else:
            # base 中没有这个键，直接添加
            base_dict[key] = value
            if verbose and (not isinstance(value, (list, dict)) or len(str(value)) < 100):
                rank0_print(f"    >>>> Add {current_path}: {value}")

    return base_dict


def initialize_config(config_path, save_path, override_config_path=None):
    """
    加载配置文件，支持用另一个配置覆盖

    Args:
        config_path: 基础配置文件路径
        save_path: 保存路径
        override_config_path: 覆盖配置路径（会覆盖 config_path 中的同名参数）
    """
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if config is None:
        rank0_print(f"[ERROR] 配置文件为空或格式错误: {config_path}")
        sys.exit(1)

    if override_config_path and exists(override_config_path):
        with open(override_config_path, 'r') as f:
            override_config = yaml.load(f, Loader=yaml.FullLoader)
        if override_config:
            deep_merge(config, override_config, verbose=False)

    config["env_config"]["save_path"] = save_path
    config["env_config"]["config_path"] = config_path

    return config


def initialize_logging(config):
    env_config = config["env_config"]
    training_config = config["training_config"]

    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H_%M")
    save_path = join(
        env_config["save_path"],
        env_config["env_id"],
        training_config['algorithm'],
        dt_string
    )
    rank0_print("    >>>> Saving to %s" % save_path)
    if not exists(save_path):
        os.makedirs(save_path)

    # 初始化 wandb（如果已经初始化则跳过）
    if wandb.run is None:
        wandb.init(
            project="rlft-training",
            name=f"{env_config['env_id']}_{training_config['algorithm']}_{dt_string}",
            config={
                # 环境配置
                "env_id": env_config["env_id"],
                "algorithm": training_config["algorithm"],
                "seed": env_config.get("seed", 0),

                # 学习率
                "actor_lr": training_config.get("actor_lr", 1e-4),
                "critic_lr": training_config.get("critic_lr", 3e-4),

                # Buffer 和 Batch
                "buffer_size": training_config.get("buffer_size", 200000),
                "batch_size_per_gpu": training_config["training_args"].get("batch_size", 64),
                "gradient_accumulation_steps": training_config["training_args"].get("gradient_accumulation_steps", 1),
                "effective_batch_size": (
                    training_config["training_args"].get("batch_size", 64) *
                    (get_world_size() if is_ddp_enabled() else 1) *
                    training_config["training_args"].get("gradient_accumulation_steps", 1)
                ),
                "max_step": training_config["training_args"].get("max_step", 500000),
                "collect_per_step": training_config["training_args"].get("collect_per_step", 128),
                "update_per_step": training_config["training_args"].get("update_per_step", 24),

                # 探索噪声
                "exploration_noise_start": training_config.get("exploration_noise_start", 0.3),
                "exploration_noise_end": training_config.get("exploration_noise_end", 0.05),

                # 冻结策略
                "freeze_vlm_actor": training_config.get("freeze_vlm_actor", True),
                "freeze_dpt_actor": training_config.get("freeze_dpt_actor", False),
                "freeze_history_actor": training_config.get("freeze_history_actor", None),
                "freeze_dpt_critic": training_config.get("freeze_dpt_critic", True),

                # TD3 超参数
                "gamma": training_config.get("policy_args", {}).get("gamma", 0.99),
                "tau": training_config.get("policy_args", {}).get("tau", 0.005),
                "n_step": training_config.get("policy_args", {}).get("n_step", 4),
                "update_actor_freq": training_config.get("policy_args", {}).get("update_actor_freq", 2),

                # VLM checkpoint
                "vlm_checkpoint": training_config.get("vlm_checkpoint_path", ""),

                # DDP 模式
                "use_data_parallel": training_config.get("use_data_parallel", False),
                "world_size": get_world_size() if is_ddp_enabled() else 1,

                # ZeRO 配置
                "zero_stage": training_config.get("zero_stage", 0),
                "deepspeed_enabled": is_deepspeed_enabled(),
                "use_4bit": training_config.get("use_4bit", True),
            },
            dir=save_path,
            resume="allow",  # 允许恢复之前的 run
            tags=[training_config['algorithm'], env_config['env_id']],  # 添加标签方便筛选
        )

    shutil.copyfile(
        env_config["config_path"],
        join(save_path, "config.yaml")
    )

    return save_path


def initialize_envs(config):
    """
    初始化环境 - Condor模式使用InfoEnv (不需要真实gym环境)

    完全模仿td3/train.py的逻辑，但移除gym依赖
    """
    env_config = config["env_config"]
    env_id = env_config["env_id"]

    rank0_print("    >>>> Using condor mode (offline training)")

    # 简化的Space类 - 替代gym.spaces.Box
    class SimpleSpace:
        def __init__(self, low, high, shape, dtype):
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype

    # 创建InfoEnv - 只提供space信息，不实际运行仿真
    class InfoEnv:
        def __init__(self, config):
            self.config = config
            env_id = config["env_config"]["env_id"]

            # 根据env_id确定action space (参数 + next_linear_vel + next_angular_vel)
            if "dwa" in env_id.lower():
                action_low = np.array([0.1, 0.314, 3, 10, 0.01, 0.01, 0.1, -0.5, -3.14])
                action_high = np.array([2.0, 3.14, 50, 50, 10.0, 10.0, 0.6, 2.0, 3.14])
            elif "teb" in env_id.lower():
                action_low = np.array([0.1, 0.0, 0.314, 0.1, 0.05, 0.05, 0.1, -0.5, -3.14])
                action_high = np.array([2.0, 1.0, 3.14, 1.0, 0.5, 0.5, 0.6, 2.0, 3.14])
            elif "ddp" in env_id.lower():
                action_low = np.array([0.1, 0.314, 400, 0.01, 0.01, 0.1, -0.5, -3.14])
                action_high = np.array([2.0, 3.14, 800, 0.4, 0.15, 0.6, 2.0, 3.14])
            elif "mppi" in env_id.lower():
                action_low = np.array([0.1, 0.314, 400, 10, 0.01, 0.01, 0.1, 0.1, -0.5, -3.14])
                action_high = np.array([2.0, 3.14, 800, 40, 0.5, 0.5, 2.0, 0.6, 2.0, 3.14])
            else:
                raise ValueError(f"Unknown env_id: {env_id}")

            self.action_space = SimpleSpace(
                low=action_low,
                high=action_high,
                shape=(len(action_low),),
                dtype=np.float32
            )

            # VLM使用图像，但这里不需要实际使用
            self.observation_space = SimpleSpace(
                low=np.array([0]),
                high=np.array([255]),
                shape=(480, 480, 3),  # RGB costmap
                dtype=np.uint8
            )

    return InfoEnv(config)


def seed(config):
    env_config = config["env_config"]
    base_seed = env_config['seed']

    # DDP: 每个进程使用不同的种子，确保采样到不同的数据
    # seed = base_seed + rank，这样每个进程的随机序列不同
    rank = get_local_rank()
    actual_seed = base_seed + rank

    np.random.seed(actual_seed)
    torch.manual_seed(actual_seed)

    rank0_print(f"    >>>> [Seed] base_seed={base_seed}, world_size={get_world_size()}, each rank uses seed=base_seed+rank")


def initialize_policy(config, env, device_override=None, accelerator=None):
    """
    初始化 policy (Actor/Critic/Optimizer)

    DDP 模式：
    - 每个进程加载自己的模型副本
    - 使用 LOCAL_RANK 确定 device
    - 4-bit 模型不使用 DataParallel，而是用 accelerate 同步梯度

    Args:
        config: 配置字典
        env: 环境（用于获取 action space）
        device_override: 手动指定设备
        accelerator: Accelerator 实例（DDP 模式）
    """
    training_config = config["training_config"]

    state_dim = env.observation_space.shape
    action_dim = np.prod(env.action_space.shape)
    action_space_low = env.action_space.low
    action_space_high = env.action_space.high

    # ===== 设备选择 =====
    if is_ddp_enabled():
        # DDP 模式：使用 LOCAL_RANK 确定设备
        local_rank = get_local_rank()
        device = f"cuda:{local_rank}"
        rank0_print(f"    >>>> [DDP] Process {local_rank}/{get_world_size()}, using device: {device}")
    elif device_override is not None:
        device = device_override
        rank0_print(f"    >>>> Using manually specified device: {device}")
    elif "CUDA_VISIBLE_DEVICES" in os.environ:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        rank0_print(f"    >>>> CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}, using device: {device}")
    else:
        devices = GPUtil.getAvailable(order='first', limit=1, maxLoad=0.8, maxMemory=0.8,
                                      includeNan=False, excludeID=[], excludeUUID=[])
        device = "cuda:%d" % (devices[0]) if len(devices) > 0 else "cpu"
        rank0_print(f"    >>>> Auto-selected device: {device}")

    vlm_checkpoint = training_config["vlm_checkpoint_path"]
    rank0_print(f"    >>>> Loading VLM+DPT from {vlm_checkpoint}")

    # 获取算法类型 (DWA/TEB/MPPI/DDP)
    algorithm = PLANNER.upper()
    rank0_print(f"    >>>> Algorithm: {algorithm}")

    # 获取量化配置
    use_4bit = training_config.get("use_4bit", True)
    rank0_print(f"    >>>> use_4bit: {use_4bit}")
    if not use_4bit:
        rank0_print(f"    >>>> 使用 bfloat16（非量化），支持 ZeRO-3")

    # ===== Actor Feature Extractor =====
    rank0_print("\n" + "=" * 70)
    rank0_print("创建 Actor Feature Extractor")
    rank0_print("=" * 70)
    actor_feature_extractor = VLM_DPT_FeatureExtractor(
        checkpoint_path=vlm_checkpoint,
        freeze_vlm=training_config.get("freeze_vlm_actor", True),
        freeze_dpt=training_config.get("freeze_dpt_actor", False),
        freeze_history=training_config.get("freeze_history_actor", None),
        device=device,
        use_4bit=use_4bit,  # 从配置读取
        algorithm=algorithm,
        name="Actor",  # 用于日志标识
        use_gradient_checkpointing=training_config.get("use_gradient_checkpointing", False)
    )

    actor = VLM_DPT_Actor(
        feature_extractor=actor_feature_extractor,
        action_dim=action_dim,
        algorithm=algorithm
    )
    # ZeRO-3 兼容：不手动 .to(device)，让 accelerator.prepare() 处理
    # 如果不是 ZeRO-3，手动移动到 device
    if not (is_ddp_enabled() and training_config.get("zero_stage", 0) == 3 and not use_4bit):
        actor = actor.to(device)

    # ===== 多 GPU 训练策略 =====
    # 4-bit 量化模型不兼容 DataParallel（device_map 冲突）
    # 使用 DDP 模式：每个进程独立加载模型，accelerate 同步梯度
    use_data_parallel = training_config.get("use_data_parallel", False)

    if use_data_parallel and not is_ddp_enabled():
        rank0_print(f"\n" + "=" * 70)
        rank0_print("⚠️  警告：use_data_parallel=True 但未检测到 DDP 环境")
        rank0_print("    4-bit 量化模型不兼容 DataParallel，请使用 DDP 模式：")
        rank0_print("    $ torchrun --nproc_per_node=N python train.py ...")
        rank0_print("    或者设置 use_data_parallel=False 使用单卡训练")
        rank0_print("=" * 70 + "\n")

    if is_ddp_enabled():
        rank0_print(f"\n" + "=" * 70)
        rank0_print(f"🚀 DDP 多 GPU 训练模式")
        rank0_print(f"   World size: {get_world_size()}")
        rank0_print(f"   Local rank: {get_local_rank()}")
        rank0_print(f"   Effective batch_size: {training_config['training_args']['batch_size']} × {get_world_size()}")
        rank0_print("=" * 70 + "\n")

    actor_optim = torch.optim.Adam(
        filter(lambda p: p.requires_grad, actor.parameters()),
        lr=training_config['actor_lr']
    )

    # ===== Critic Feature Extractor (完全独立) =====
    rank0_print("\n" + "=" * 70)
    rank0_print("创建 Critic Feature Extractor (完全独立)")
    rank0_print("=" * 70)

    # 完全独立的 VLM，避免设备冲突
    critic_feature_extractor = VLM_DPT_FeatureExtractor(
        checkpoint_path=vlm_checkpoint,
        freeze_vlm=True,  # Critic 的 VLM 始终冻结
        freeze_dpt=training_config.get("freeze_dpt_critic", True),  # Critic DPT 默认冻结
        freeze_history=training_config.get("freeze_history_critic", True),  # Critic History 默认冻结
        device=device,
        use_4bit=use_4bit,  # 从配置读取（与 Actor 一致）
        algorithm=algorithm,
        name="Critic",  # 用于日志标识
        use_gradient_checkpointing=training_config.get("use_gradient_checkpointing", False)  # Critic 也启用以保持一致
    )

    critic = VLM_DPT_Critic(
        feature_extractor=critic_feature_extractor,  # 独立的 feature_extractor
        action_dim=action_dim,
        detach_features=False  # 不需要 detach，因为 Critic 有自己的 DPT
    )
    # ZeRO-3 兼容：不手动 .to(device)，让 accelerator.prepare() 处理
    if not (is_ddp_enabled() and training_config.get("zero_stage", 0) == 3 and not use_4bit):
        critic = critic.to(device)

    # Critic 不使用 DataParallel（4-bit 模型使用 DDP）

    # 过滤 Critic 参数：
    # - 排除 base_model 参数（共享的 VLM，不应由 Critic 优化器更新）
    # - 包含 dpt_head 和 history_encoder 参数（Critic 独立的，如果可训练）
    # - 包含 Q-heads 参数
    critic_params = []
    excluded_params = []
    for name, param in critic.named_parameters():
        if param.requires_grad:
            # 排除共享的 VLM base_model
            if 'base_model' in name:
                excluded_params.append(name)
            else:
                critic_params.append(param)

    rank0_print(f"    >>>> Critic 优化器参数: {len(critic_params)}")
    if excluded_params:
        rank0_print(f"    >>>> Critic 排除参数（共享的 VLM base_model）: {len(excluded_params)}")

    critic_optim = torch.optim.Adam(critic_params, lr=training_config['critic_lr'])

    # 打印参数统计
    actor_trainable = sum(p.numel() for p in actor.parameters() if p.requires_grad)
    actor_total = sum(p.numel() for p in actor.parameters())
    rank0_print(f"    >>>> Actor: {actor_trainable:,} / {actor_total:,} trainable ({100*actor_trainable/actor_total:.2f}%)")

    critic_trainable = sum(p.numel() for p in critic.parameters() if p.requires_grad)
    critic_total = sum(p.numel() for p in critic.parameters())
    rank0_print(f"    >>>> Critic: {critic_trainable:,} / {critic_total:,} trainable ({100*critic_trainable/critic_total:.2f}%)")

    # 加载参数归一化统计（从VLM checkpoint）
    param_mean, param_std = None, None
    normalization_dir = os.path.join(vlm_checkpoint, "normalization")
    if os.path.exists(normalization_dir):
        mean_path = os.path.join(normalization_dir, "param_mean.npy")
        std_path = os.path.join(normalization_dir, "param_std.npy")

        if os.path.exists(mean_path) and os.path.exists(std_path):
            param_mean = np.load(mean_path)
            param_std = np.load(std_path)
            rank0_print(f"    >>>> Loaded parameter normalization from {normalization_dir}")
            rank0_print(f"         Mean: {param_mean}")
            rank0_print(f"         Std:  {param_std}")

            # 验证维度匹配
            if len(param_mean) != action_dim or len(param_std) != action_dim:
                rank0_print(f"    >>>> WARNING: Normalization dim mismatch!")
                rank0_print(f"         Expected {action_dim}, got mean={len(param_mean)}, std={len(param_std)}")
                rank0_print(f"         Disabling normalization")
                param_mean, param_std = None, None
        else:
            rank0_print(f"    >>>> WARNING: Normalization files not found in {normalization_dir}")
    else:
        rank0_print(f"    >>>> INFO: No normalization directory found, using raw parameter space")

    # ===== DDP/ZeRO 模式：使用 ZeROCompatibleTrainer =====
    zero_trainer = None
    if accelerator is not None and is_ddp_enabled():
        zero_trainer = ZeROCompatibleTrainer(accelerator)
        zero_stage = get_zero_stage()

        rank0_print(f"    >>>> [ZeRO] 使用 ZeROCompatibleTrainer...")
        rank0_print(f"    >>>> [ZeRO] ZeRO Stage: {zero_stage}")
        rank0_print(f"    >>>> [ZeRO] use_4bit: {use_4bit}")

        if use_4bit:
            # 4-bit 模式：只能准备优化器（模型不能被 ZeRO 分片）
            actor_optim = zero_trainer.prepare_optimizer(actor, actor_optim)
            critic_optim = zero_trainer.prepare_optimizer(critic, critic_optim)
            rank0_print(f"    >>>> [ZeRO] ✓ 优化器已准备（4-bit 模式，模型不分片）")
        else:
            # 非 4-bit 模式：可以完整使用 ZeRO-3 分片模型
            actor, actor_optim = zero_trainer.prepare_model_and_optimizer(
                actor, actor_optim, is_quantized=False
            )
            critic, critic_optim = zero_trainer.prepare_model_and_optimizer(
                critic, critic_optim, is_quantized=False
            )
            rank0_print(f"    >>>> [ZeRO] ✓ 模型和优化器已准备（bfloat16 模式，参数分片）")

    # TD3 policy
    policy = TD3(
        actor, actor_optim,
        critic, critic_optim,
        action_range=[action_space_low, action_space_high],
        device=device,
        param_mean=param_mean,
        param_std=param_std,
        **training_config["policy_args"]
    )

    # 保存 accelerator 和 zero_trainer 引用（用于 train 函数）
    if accelerator is not None:
        policy.accelerator = accelerator
        policy.zero_trainer = zero_trainer  # ZeROCompatibleTrainer 实例
    else:
        policy.accelerator = None
        policy.zero_trainer = None

    # VLM专用ReplayBuffer
    buffer = VLMReplayBuffer(
        state_dim, action_dim,
        training_config['buffer_size'],
        device=device,
        image_size=(400, 600)  # (height, width)
    )

    return policy, buffer


def train(env, policy, buffer, config, skip_test=False):
    """训练主循环 - Condor模式（完全模仿td3/train.py）

    Args:
        skip_test: 是否跳过 test 评估（默认 False）
    """
    env_config = config["env_config"]
    training_config = config["training_config"]

    # DDP 模式：只在主进程初始化日志
    if is_main_process():
        save_path = initialize_logging(config)
    else:
        # 非主进程不需要 wandb
        save_path = join(
            config["env_config"]["save_path"],
            config["env_config"]["env_id"],
            training_config['algorithm'],
            datetime.now().strftime("%Y_%m_%d_%H_%M")
        )

    training_args = training_config["training_args"]

    # 传递 ftrl_config 给 collector（合并 gpu_config）
    ftrl_config = config.get("ftrl_config", {}).copy()

    # 从 gpu_config 读取 gpu_strategy（如果存在）
    gpu_config = config.get("gpu_config", {})
    if "inference_gpu_strategy" in gpu_config:
        ftrl_config["gpu_strategy"] = gpu_config["inference_gpu_strategy"]
        rank0_print(f"    >>>> Using gpu_strategy from gpu_config: {gpu_config['inference_gpu_strategy']}")

    # Debug 模式：--no_cleanup（命令行参数 > 配置文件 > 默认值 False）
    no_cleanup = ftrl_config.get("no_cleanup", False)
    if args.no_cleanup:
        no_cleanup = True
    ftrl_config["no_cleanup"] = no_cleanup

    if no_cleanup and is_main_process():
        rank0_print("\n" + "=" * 80)
        rank0_print("🔧 DEBUG 模式 (--no_cleanup)")
        rank0_print("   - 不清理轨迹文件和图片")
        rank0_print("   - World 顺序固定（不随机打乱）")
        rank0_print("=" * 80 + "\n")

    # 清空所有 actor 目录（只在主进程执行，no_cleanup 时跳过）
    if is_main_process() and not no_cleanup:
        rank0_print("\n" + "=" * 80)
        rank0_print("清理 actor 目录...")
        rank0_print("=" * 80)

        import glob
        actor_dirs = glob.glob(join(BUFFER_PATH, "actor_*"))

        if actor_dirs:
            cleaned_count = 0
            file_count = 0

            for actor_dir in actor_dirs:
                # 清理 pickle 文件
                pickle_files = glob.glob(join(actor_dir, "*.pickle"))
                for f in pickle_files:
                    try:
                        os.remove(f)
                        file_count += 1
                    except Exception as e:
                        rank0_print(f"  ⚠️ 无法删除 {f}: {e}")

                # 清理 png 文件
                png_files = glob.glob(join(actor_dir, "*.png"))
                for f in png_files:
                    try:
                        os.remove(f)
                        file_count += 1
                    except Exception as e:
                        rank0_print(f"  ⚠️ 无法删除 {f}: {e}")

                if pickle_files or png_files:
                    cleaned_count += 1

            rank0_print(f"  ✓ 已清理 {cleaned_count} 个 actor 目录，删除 {file_count} 个文件")
        else:
            rank0_print(f"  ℹ️  没有找到 actor 目录")

        rank0_print("=" * 80 + "\n")

    # DDP 同步：等待主进程清理完成
    if is_ddp_enabled():
        import torch.distributed as dist
        if dist.is_initialized():
            dist.barrier()

    collector = VLMCondorCollector(
        policy, env, buffer, BUFFER_PATH,
        ftrl_config=ftrl_config  # 传递合并后的配置
    )

    # 注册信号处理器，确保 Ctrl+C 时清理子进程和 tmux 服务
    def signal_handler(sig, frame):
        rank0_print("\n\n" + "=" * 80)
        rank0_print("🛑 收到退出信号 (Ctrl+C)，正在清理...")
        rank0_print("=" * 80)

        # DDP 模式：先同步所有进程，避免 TCPStore broken pipe
        if is_ddp_enabled():
            try:
                import torch.distributed as dist
                if dist.is_initialized():
                    rank0_print("同步所有 DDP 进程...")
                    # 非主进程等待一下，让主进程先清理
                    if not is_main_process():
                        time.sleep(2)
                    dist.barrier(timeout=timedelta(seconds=5))
            except Exception as e:
                rank0_print(f"⚠️ DDP 同步失败: {e}")

        # 1. 停止数据收集容器（只在主进程）
        if is_main_process():
            try:
                collector.stop_collection_containers()
                rank0_print("✓ 所有数据收集容器已停止")
            except Exception as e:
                rank0_print(f"⚠️ 清理数据收集容器时出现错误: {e}")

        # 2. 停止 tmux VLM 服务（只在主进程，可配置）
        if is_main_process():
            ftrl_config = config.get("ftrl_config", {})
            auto_stop_services = ftrl_config.get("auto_stop_services", True)  # 默认关闭

            try:
                if TMUX_SERVICES_STARTED and auto_stop_services:
                    stop_tmux_services()
                    rank0_print("✓ 所有 tmux VLM 服务已停止")
                elif TMUX_SERVICES_STARTED and not auto_stop_services:
                    rank0_print("ℹ️ tmux VLM 服务保持运行 (auto_stop_services=False)")
                    rank0_print(f"   手动停止: tmux kill-session -t ftrl_*")
                else:
                    rank0_print("ℹ️ 没有需要停止的 tmux 服务")
            except Exception as e:
                rank0_print(f"⚠️ 清理 tmux 服务时出现错误: {e}")

        # 3. 关闭 wandb（只在主进程）
        if is_main_process():
            try:
                wandb.finish()
                rank0_print("✓ wandb 已关闭")
            except Exception as e:
                rank0_print(f"⚠️ 关闭 wandb 时出现错误: {e}")

        rank0_print("=" * 80)
        rank0_print("退出训练")
        rank0_print("=" * 80)

        # DDP 模式：非主进程先退出，主进程等待一下
        if is_ddp_enabled():
            if not is_main_process():
                sys.exit(0)
            else:
                time.sleep(1)  # 主进程等待，确保其他进程先退出

        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # kill 命令

    # 显示信号处理器配置（只在主进程）
    ftrl_config = config.get("ftrl_config", {})
    auto_stop = ftrl_config.get("auto_stop_services", True)
    if is_main_process():
        rank0_print(f"    >>>> 信号处理器已注册 (Ctrl+C 将清理容器，tmux服务={'自动关闭' if auto_stop else '保持运行'})")

    # 启动数据收集容器（后台持续运行）
    # DDP 模式：只在主进程启动容器
    # 优先级：no_cleanup > 命令行参数 > 配置文件 > 默认值(False)
    start_containers = ftrl_config.get("start_containers", False)
    if args.start_containers is not None:  # 命令行覆盖
        start_containers = args.start_containers
    if args.no_cleanup:  # no_cleanup 时强制跳过 tmux 和 containers
        start_containers = False
        rank0_print(f"    >>>> [DEBUG] no_cleanup={args.no_cleanup}, 跳过启动数据收集容器")

    if is_main_process():
        if start_containers:
            rank0_print("\n" + "=" * 80)
            rank0_print("启动数据收集容器...")
            rank0_print("=" * 80)
            collector.start_collection_containers(mode='train')
            rank0_print("=" * 80)
            rank0_print("✓ 数据收集容器已启动，将在后台持续收集数据")
            rank0_print("=" * 80 + "\n")
        else:
            rank0_print("\n" + "=" * 80)
            rank0_print("⚠️  跳过启动数据收集容器 (start_containers=False)")
            rank0_print("    将从已有的 buffer 数据进行训练")
            rank0_print("=" * 80 + "\n")

    # DDP 模式：同步所有进程
    if is_ddp_enabled():
        import torch.distributed as dist
        if dist.is_initialized():
            dist.barrier()

    # ===== 初始测试：训练前先评估一次 =====
    if is_main_process():
        rank0_print("\n" + "=" * 60)
        rank0_print("初始测试 (训练前基线)")
        rank0_print("=" * 60)

    init_test_start = time.time()
    init_test_steps, init_test_epinfo = collector.collect(
        n_steps=training_args['collect_per_step'],
        status='test'
    )
    init_test_time = time.time() - init_test_start

    # 计算初始测试指标
    if init_test_epinfo:
        init_nav_metrics = [ep['nav_metric'] for ep in init_test_epinfo]
        init_ep_times = [ep['ep_time'] for ep in init_test_epinfo]
        init_ep_lens = [ep['ep_len'] for ep in init_test_epinfo]

        init_log = {
            'Test_counts': len(init_test_epinfo),
            'Test_length': np.mean(init_ep_lens) if init_ep_lens else 0,
            'Test_nav_metric': np.mean(init_nav_metrics) if init_nav_metrics else 0,
            'Test_time': np.mean(init_ep_times) if init_ep_times else 0,
        }
        print(f"\n初始测试结果:")
        for k, v in init_log.items():
            print(f"  {k}: {v}")
        print(f"\n测试耗时: {init_test_time:.1f}s")
    else:
        print("⚠️ 初始测试未收集到任何 episode")
        
    print("=" * 60 + "\n")

    if is_main_process():
        rank0_print("    >>>> Pre-collect experience (Condor mode)")
        rank0_print(f"    >>>> Target buffer size: {training_config['pre_collect']} steps")

    # Condor collect：从actor目录加载轨迹
    collector.collect(n_steps=training_config['pre_collect'], status='train')

    rank0_print(f"    >>>> Buffer filled! Current size: {buffer.size} steps")
    rank0_print(f"    >>>> Starting policy training...")
    rank0_print("=" * 80)

    n_steps = 0
    n_iter = 0
    n_ep = 0
    epinfo_buf = collections.deque(maxlen=BUFFER_SIZE)
    world_ep_buf = collections.defaultdict(lambda: collections.deque(maxlen=30))
    t0 = time.time()

    tensorboard_step = 0

    best_episode_length = float('inf')
    best_episode_nav = float('-inf')

    # ===== 计算并打印 Batch Size 详情 =====
    batch_size_per_gpu = training_args['batch_size']
    gradient_accum_steps = training_args.get('gradient_accumulation_steps', 1)
    world_size = get_world_size()
    effective_batch_size = batch_size_per_gpu * world_size * gradient_accum_steps

    rank0_print(f"\n{'='*70}")
    rank0_print(f"  📊 Batch Size 详情")
    rank0_print(f"{'='*70}")
    rank0_print(f"  batch_size (per GPU):          {batch_size_per_gpu}")
    rank0_print(f"  gradient_accumulation_steps:   {gradient_accum_steps}")
    rank0_print(f"  world_size (GPU 数量):         {world_size}")
    rank0_print(f"  ─────────────────────────────────────────")
    rank0_print(f"  🎯 effective_batch_size:       {batch_size_per_gpu} × {world_size} × {gradient_accum_steps} = {effective_batch_size}")
    rank0_print(f"{'='*70}")
    rank0_print(f"  每次 optimizer.step() 实际使用 {effective_batch_size} 个样本的梯度")
    rank0_print(f"{'='*70}\n")

    rank0_print(f"{'='*60}")
    rank0_print(f"开始训练循环 | max_step: {training_args['max_step']}")
    rank0_print(f"update_per_step: {training_args['update_per_step']} | collect_per_step: {training_args['collect_per_step']}")
    rank0_print(f"{'='*60}\n")

    while n_steps < training_args["max_step"]:
        iter_start_time = time.time()
        rank0_print(f"\n[Iter {n_iter}] 开始 | 总步数: {n_steps}/{training_args['max_step']}")

        # Linear decaying exploration noise
        policy.exploration_noise = \
            - (training_config["exploration_noise_start"] - training_config["exploration_noise_end"]) \
            * n_steps / training_args["max_step"] + training_config["exploration_noise_start"]

        rank0_print(f"[Iter {n_iter}] 收集训练数据...")
        collect_start = time.time()
        steps, epinfo = collector.collect(training_args["collect_per_step"], status = 'train')
        collect_time = time.time() - collect_start
        rank0_print(f"[Iter {n_iter}] 收集完成: {steps} steps, {len(epinfo)} episodes, 耗时: {collect_time:.1f}s")

        # 只在主进程更新统计变量
        if is_main_process():
            n_steps += steps
            n_iter += 1
            n_ep += len(epinfo)
            epinfo_buf.extend(epinfo)

            for d in epinfo:
                world = d["world"].split("/")[-1]
                world_ep_buf[world].append(d)
        else:
            # 非主进程也需要更新 n_steps，确保循环条件一致
            # 但不需要 broadcast（避免与梯度同步冲突）
            n_steps += steps
            n_iter += 1

        actor_grad_norms = []
        critic_grad_norms = []
        actor_losses = []
        critic_losses = []

        # 动态调整 update_per_step（根据 buffer 大小）
        # 配合 collect_per_step=1024，保持 UTD ≈ 3-6
        buffer_size = len(buffer)
        if buffer_size < 5000:
            dynamic_update_per_step = 24
        elif buffer_size < 20000:
            dynamic_update_per_step = 48
        else:
            dynamic_update_per_step = 64

        rank0_print(f"[Iter {n_iter}] 开始策略更新 ({dynamic_update_per_step} 次, buffer={buffer_size})...")
        update_start = time.time()

        # 获取梯度累积步数（默认=1，即不累积）
        gradient_accumulation_steps = training_args.get("gradient_accumulation_steps", 1)

        for update_idx in range(dynamic_update_per_step):
            update_iter_start = time.time()
            actor_grad_norm, critic_grad_norm, actor_loss, critic_loss = policy.train(
                buffer,
                batch_size=training_args["batch_size"],
                gradient_accumulation_steps=gradient_accumulation_steps
            )
            update_iter_time = time.time() - update_iter_start

            if actor_loss is not None:
                actor_grad_norms.append(actor_grad_norm)
                actor_losses.append(actor_loss)

            critic_grad_norms.append(critic_grad_norm)
            critic_losses.append(critic_loss)

            # 每4次更新打印一次进度（偶数次能看到actor_loss）
            if (update_idx + 1) % 4 == 0 or update_idx == 0:
                rank0_print(f"  更新 {update_idx+1}/{dynamic_update_per_step} | "
                      f"critic_loss: {critic_loss:.4f} | "
                      f"actor_loss: {actor_loss if actor_loss else 'N/A'} | "
                      f"耗时: {update_iter_time:.2f}s")

        update_time = time.time() - update_start
        rank0_print(f"[Iter {n_iter}] 策略更新完成, 总耗时: {update_time:.1f}s")

        t1 = time.time()

        # 执行 test（或跳过实际运行）
        rank0_print(f"[Iter {n_iter}] 收集测试数据...")
        test_start = time.time()
        # skip_actual_test=True: 执行步骤 1-4（保存、停止、清理），跳过步骤 5-7（运行 test）
        test_steps, test_epinfo = collector.collect(
            n_steps=training_args['collect_per_step'],
            status='test',
            skip_actual_test=skip_test
        )
        test_time = time.time() - test_start
        if skip_test:
            rank0_print(f"[Iter {n_iter}] Test 阶段完成 (跳过实际运行): 耗时 {test_time:.1f}s")
        else:
            rank0_print(f"[Iter {n_iter}] 测试完成: {test_steps} steps, {len(test_epinfo)} episodes, 耗时: {test_time:.1f}s")

        # 只在主进程计算统计和构建日志
        # (非主进程的 epinfo_buf 和 test_epinfo 都是空的)
        # 初始化 should_save_best（所有进程都需要，用于 ZeRO-3 同步）
        should_save_best = False
        if is_main_process():
            status_counts = {"success": 0, "flip": 0, "timeout": 0}
            total_episodes = len(epinfo_buf)

            for epinfo in epinfo_buf:
                status = epinfo.get("ep_status", "unknown")
                if status in status_counts:
                    status_counts[status] += 1

            success_rate = 100.0 * status_counts["success"] / total_episodes if total_episodes > 0 else 0.0
            flip_rate = 100.0 * status_counts["flip"] / total_episodes if total_episodes > 0 else 0.0
            timeout_rate = 100.0 * status_counts["timeout"] / total_episodes if total_episodes > 0 else 0.0

            # 安全地计算均值（避免空列表导致 nan）
            nav_metric_score = np.mean([epinfo["nav_metric"] for epinfo in epinfo_buf]) if epinfo_buf else 0.0

            nav_metrics = [ep['nav_metric'] for ep in test_epinfo]
            avg_nav_metric = sum(nav_metrics) / len(nav_metrics) if nav_metrics else 0.0

            ep_times = [ep['ep_time'] for ep in test_epinfo]
            avg_ep_time = sum(ep_times) / len(ep_times) if ep_times else 0.0

            ep_time_steps = [ep['ep_len'] for ep in test_epinfo]
            avg_ep_len = sum(ep_time_steps) / len(ep_time_steps) if ep_time_steps else 0.0

            log = {
                "Episode_reward": np.mean([epinfo["ep_rew"] for epinfo in epinfo_buf]) if epinfo_buf else 0.0,
                "Episode_length": np.mean([epinfo["ep_len"] for epinfo in epinfo_buf]) if epinfo_buf else 0.0,
                "Episode_nav_metric": nav_metric_score,
                "Test_nav_metric": avg_nav_metric,
                "Test_time" : avg_ep_time,
                "Test_length": avg_ep_len,
                "Test_counts": len(nav_metrics),
                "Success_rate": success_rate,
                "Flip_rate": flip_rate,
                "Timeout_rate": timeout_rate,
                "Status_counts": total_episodes,
                "Time": np.mean([epinfo["ep_time"] for epinfo in epinfo_buf]) if epinfo_buf else 0.0,
                "Collision": np.mean([epinfo["collision"] for epinfo in epinfo_buf]) if epinfo_buf else 0.0,
                "Actor_grad_norm": np.mean(actor_grad_norms) if actor_grad_norms else 0.0,
                "Critic_grad_norm": np.mean(critic_grad_norms) if critic_grad_norms else 0.0,
                "Actor_loss": np.mean(actor_losses) if actor_losses else 0.0,
                "Critic_loss": np.mean(critic_losses) if critic_losses else 0.0,
                "fps": n_steps / (t1 - t0),
                "n_episode": n_ep,
                "Steps": n_steps,
                "Exploration_noise": policy.exploration_noise
            }

            logging.info(pformat(log))

            # 最佳模型保存逻辑
            # 注意：ZeRO-3 的 GatheredParameters 需要所有进程参与，
            # 所以 policy.save() 必须在 is_main_process() 块外面调用
            if len(epinfo_buf) >= 0:
                current_episode_reward = log["Episode_reward"]

                # 根据是否跳过 test 选择评估指标
                if skip_test:
                    # 跳过 test：使用训练数据的指标
                    current_episode_length = log["Episode_length"]
                    current_episode_nav_metric = log["Episode_nav_metric"]
                    rank0_print(f"    >>>> [Skip Test] 使用训练指标评估: Episode_nav_metric={current_episode_nav_metric:.3f}")
                else:
                    # 正常 test：使用测试数据的指标
                    current_episode_length = log["Test_length"]
                    current_episode_nav_metric = log["Test_nav_metric"]

                if (current_episode_nav_metric > best_episode_nav):
                    should_save_best = True

                    metric_name = "Episode_nav_metric (Train)" if skip_test else "Test_nav_metric"
                    rank0_print(f"    >>>> 💾 新的最佳模型! {metric_name}: {current_episode_nav_metric:.3f} (之前: {best_episode_nav:.3f})")

                    best_episode_nav = current_episode_nav_metric
                    best_episode_length = current_episode_length

        # ZeRO-3 集体操作：所有进程都必须参与 policy.save()
        # 使用 dist.broadcast 同步 should_save_best 决策
        import torch.distributed as dist
        if dist.is_initialized():
            should_save_tensor = torch.tensor([1 if should_save_best else 0], dtype=torch.int, device='cuda')
            dist.broadcast(should_save_tensor, src=0)
            should_save_best = should_save_tensor.item() == 1

        if should_save_best:
            # 所有进程参与 gather，但只有 rank 0 写文件
            policy_name = f"policy_step_{tensorboard_step}"
            policy.save(save_path, policy_name)
            policy.save(save_path, "best_model")

            # 只在主进程写入元数据文件
            if is_main_process():
                rank0_print(f"    >>>> 已保存到: {join(save_path, 'best_model')}")

                # 记录最佳性能信息
                with open(join(save_path, f"best_performance_step_{tensorboard_step}.txt"), 'w') as f:
                    f.write(f"Best Performance at TensorBoard Step {tensorboard_step}:\n")
                    f.write(f"Training Step: {n_steps}\n")
                    f.write(f"Episode Reward: {current_episode_reward:.3f}\n")
                    f.write(f"Episode Length: {current_episode_length:.3f}\n")
                    f.write(f"Success Rate: {success_rate:.3f}%\n")
                    if skip_test:
                        f.write(f"Episode_nav_metric (Train): {current_episode_nav_metric:.3f}\n")
                        f.write(f"Note: Evaluated on training data (--skip_test enabled)\n")
                    else:
                        f.write(f"Test_nav_metric: {current_episode_nav_metric:.3f}\n")

                # 更新 best_model 的元信息
                with open(join(save_path, "best_model", "best_info.txt"), 'w') as f:
                    f.write(f"Best Model Info\n")
                    f.write(f"===============\n")
                    f.write(f"TensorBoard Step: {tensorboard_step}\n")
                    f.write(f"Training Step: {n_steps}\n")
                    if skip_test:
                        f.write(f"Episode_nav_metric (Train): {current_episode_nav_metric:.3f}\n")
                        f.write(f"Evaluation Mode: Training data (--skip_test)\n")
                    else:
                        f.write(f"Test_nav_metric: {current_episode_nav_metric:.3f}\n")
                        f.write(f"Evaluation Mode: Test data\n")
                    f.write(f"Episode Reward: {current_episode_reward:.3f}\n")
                    f.write(f"Episode Length: {current_episode_length:.3f}\n")
                    f.write(f"Success Rate: {success_rate:.3f}%\n")
                    f.write(f"Saved at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # 继续主进程的日志逻辑
        if is_main_process():

            # Wandb 日志写入
            if n_iter % training_config["log_intervals"] == 0:
                # 构建 wandb 日志字典
                wandb_log = {}

                # 添加主要训练指标
                for k in log.keys():
                    wandb_log[f'train/{k}'] = log[k]

                # 添加每个 world 的指标
                for k in world_ep_buf.keys():
                    wandb_log[f'{k}/Episode_reward'] = np.mean([epinfo["ep_rew"] for epinfo in world_ep_buf[k]])
                    wandb_log[f'{k}/Episode_length'] = np.mean([epinfo["ep_len"] for epinfo in world_ep_buf[k]])
                    wandb_log[f'{k}/Episode_nav_metric'] = np.mean([epinfo["nav_metric"] for epinfo in world_ep_buf[k]])
                    wandb_log[f'{k}/Success_rate'] = success_rate
                    wandb_log[f'{k}/Flip_rate'] = flip_rate
                    wandb_log[f'{k}/Timeout_rate'] = timeout_rate
                    wandb_log[f'{k}/Status_counts'] = total_episodes
                    wandb_log[f'{k}/Time'] = np.mean([epinfo["ep_time"] for epinfo in world_ep_buf[k]])
                    wandb_log[f'{k}/Collision'] = np.mean([epinfo["collision"] for epinfo in world_ep_buf[k]])

                # 一次性写入所有指标
                wandb.log(wandb_log, step=tensorboard_step)

        # 所有进程更新 tensorboard_step（避免 broadcast 与梯度同步冲突）
        tensorboard_step += steps

    # Condor模式：清理临时文件（可选）
    # shutil.rmtree(BUFFER_PATH, ignore_errors=True)
    rank0_print("    >>>> Training completed!")

    # 关闭 wandb（只在主进程）
    if is_main_process():
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Start training')
    parser.add_argument('--config_path', dest='config_path', default="../script/ft_qwen/configs/")
    parser.add_argument('--config_file', dest='config_file', default="ddp")
    parser.add_argument('--buffer_path', dest='buffer_path', default="../buffer/")
    parser.add_argument('--model_path', dest='model_path', default="../model_rlft/",
                        help='Path to load checkpoints (e.g., ../model_ftrl/)')
    parser.add_argument('--logging_path', dest='logging_path', default="../logging/")
    parser.add_argument('--buffer_size', dest='buffer_size', default= 200)
    parser.add_argument('--device', dest='device', default="cuda:0")
    parser.add_argument('--gpu', dest='gpu', type=int, default=None,
                        help='GPU ID to use (e.g., --gpu 1). Sets CUDA_VISIBLE_DEVICES.')
    parser.add_argument('--policy_name', dest='policy_name', default="ddp_rlft",
                        help='Policy name for buffer directory (e.g., dwa_ftrl)')
    parser.add_argument('--planner', dest='planner', default=None,
                        help='Planner name for checkpoint directory (e.g., DWA, TEB, DDP, MPPI)')
    parser.add_argument('--start_containers', dest='start_containers', default=None, type=lambda x: x.lower() == 'true',
                        help='Start collection containers (default: from config file, see ftrl_config.start_containers)')
    parser.add_argument('--start_services', dest='start_services', default=None, type=lambda x: x.lower() == 'true',
                        help='Auto start tmux VLM services (default: from config file, see ftrl_config.auto_start_services)')
    parser.add_argument('--skip_test', dest='skip_test', default=False, type=lambda x: x.lower() == 'true',
                        help='Skip test evaluation during training (default: False, use --skip_test true to enable)')
    parser.add_argument('--no_cleanup', dest='no_cleanup', default=False, type=lambda x: x.lower() == 'true',
                        help='Debug mode: disable file cleanup and world shuffling (default: False)')

    logging.getLogger().setLevel("INFO")
    args = parser.parse_args()

    # ===== 配置文件简写映射 =====
    if args.config_file in CONFIG_SHORTCUTS:
        original_config = args.config_file
        args.config_file = CONFIG_SHORTCUTS[args.config_file]
        rank0_print(f"[Config] 简写映射: {original_config} → {args.config_file}")

    # ===== GPU 设置 =====
    # 注意: CUDA_VISIBLE_DEVICES 已在文件开头 import torch 之前设置
    if args.gpu is not None:
        args.device = "cuda:0"  # CUDA_VISIBLE_DEVICES 后，cuda:0 就是指定的 GPU
        rank0_print(f"[GPU] 使用 GPU {args.gpu} (实际设备: cuda:0)")

    BUFFER_BASE_PATH = args.buffer_path   # e.g., ../buffer/
    MODEL_PATH = args.model_path           # e.g., ../model_rlft/
    BUFFER_SIZE = args.buffer_size
    SAVE_PATH = args.logging_path          # e.g., ../logging/
    policy_name = args.policy_name
    VLM_CONFIG_PATH = args.config_path
    VLM_CONFIG_NAME = args.config_file

    # ===== 检查参数和加载配置 =====
    if not policy_name:
        rank0_print("[ERROR] --policy_name 是必须参数")
        sys.exit(1)

    BUFFER_PATH = join(BUFFER_BASE_PATH, policy_name)
    vlm_config_file = join(VLM_CONFIG_PATH, VLM_CONFIG_NAME + ".yaml")
    buffer_config_file = join(BUFFER_PATH, "config.yaml")

    if not exists(vlm_config_file):
        rank0_print(f"[ERROR] VLM 配置不存在: {vlm_config_file}")
        sys.exit(1)

    if not exists(BUFFER_PATH):
        os.makedirs(BUFFER_PATH)

    # buffer 必须有 config，用 VLM config 覆盖
    if not exists(buffer_config_file):
        rank0_print(f"[ERROR] Buffer 配置不存在: {buffer_config_file}")
        sys.exit(1)

    config = initialize_config(buffer_config_file, SAVE_PATH, override_config_path=vlm_config_file)

    ACTION_TYPE = config["env_config"].get("action_type", "DDP")
    PLANNER = args.planner.lower() if args.planner else ACTION_TYPE.split('_')[0].lower()

    rank0_print(f"[Config] policy={policy_name}, planner={PLANNER}, buffer={BUFFER_PATH}")

    # ===== Step 2: 初始化环境和策略 =====
    if is_main_process():
        rank0_print("\n" + "=" * 80)
        rank0_print("Step 2: 初始化环境和策略")
        rank0_print("=" * 80)
    seed(config)

    if is_main_process():
        rank0_print("    >>>> Creating the environments (Condor mode)")
    env = initialize_envs(config)

    # ===== 初始化 Accelerator (DDP/ZeRO 模式) =====
    accelerator = None
    if is_ddp_enabled():
        import torch.distributed as dist
        # 初始化分布式后端（如果还没初始化）
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')
            if is_main_process():
                rank0_print(f"    >>>> [DDP] torch.distributed initialized (backend=nccl)")

        # 获取 ZeRO 配置
        training_config = config["training_config"]
        zero_stage = training_config.get("zero_stage", 0)
        zero_config_file = training_config.get("zero_config_file", None)
        gradient_accumulation_steps = training_config["training_args"].get("gradient_accumulation_steps", 1)

        # 🔧 提前设置 ZERO_STAGE 环境变量（vlm_net.py 需要在加载模型时使用）
        os.environ["ZERO_STAGE"] = str(zero_stage)

        if is_main_process():
            rank0_print(f"    >>>> [ZeRO] zero_stage={zero_stage}, config_file={zero_config_file}")

        # ⚠️ 重要检查：ZeRO-3 与 4-bit 量化不兼容
        use_4bit = training_config.get("use_4bit", True)  # 从配置读取
        freeze_vlm = training_config.get("freeze_vlm_actor", True)

        if zero_stage == 3 and use_4bit:
            rank0_print("\n" + "=" * 70)
            rank0_print("❌ 错误：ZeRO-3 与 4-bit 量化 VLM 不兼容！")
            rank0_print("=" * 70)
            rank0_print("原因：")
            rank0_print("  - bitsandbytes 4-bit 使用 NF4 格式")
            rank0_print("  - ZeRO-3 需要分片参数，但 NF4 不能被分片")
            if not freeze_vlm:
                rank0_print("  - freeze_vlm_actor=false，LoRA 与 4-bit base 耦合")
            rank0_print("")
            rank0_print("解决方案：")
            rank0_print("  1. 使用 ZeRO-2（推荐）: zero_stage: 2")
            rank0_print("  2. 使用普通 DDP: zero_stage: 0")
            rank0_print("  3. 关闭量化（显存需求增加）: use_4bit: false")
            rank0_print("=" * 70 + "\n")
            sys.exit(1)

        if zero_stage > 0 or zero_config_file:
            # 使用 ZeRO 优化
            batch_size = training_config["training_args"].get("batch_size", 16)
            accelerator = create_accelerator_for_rlft(
                gradient_accumulation_steps=gradient_accumulation_steps,
                mixed_precision="bf16",
                zero_stage=zero_stage,
                deepspeed_config_file=zero_config_file,
                batch_size=batch_size
            )
            if is_main_process():
                rank0_print(f"    >>>> [ZeRO] Accelerator initialized with ZeRO-{zero_stage}")
                rank0_print(f"         Device: {accelerator.device}")
                rank0_print(f"         Num processes: {accelerator.num_processes}")
                rank0_print(f"         DeepSpeed: {is_deepspeed_enabled()}")
        else:
            # 普通 DDP（无 ZeRO）
            ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
            accelerator = Accelerator(
                kwargs_handlers=[ddp_kwargs],
                gradient_accumulation_steps=gradient_accumulation_steps,
                mixed_precision="bf16"
            )
            if is_main_process():
                rank0_print(f"    >>>> [DDP] Accelerator initialized (no ZeRO)")
                rank0_print(f"         Device: {accelerator.device}")
                rank0_print(f"         Num processes: {accelerator.num_processes}")

        # ZERO_STAGE 已在前面设置

    if is_main_process():
        rank0_print("    >>>> Initializing the policy")
    policy, buffer = initialize_policy(config, env, device_override=args.device, accelerator=accelerator)

    # 加载 RLFT checkpoint（如果有）
    policy.load(BUFFER_PATH, "policy")

    # 🔧 DDP/ZeRO 模式：同步所有 GPU 上的模型参数
    if is_ddp_enabled():
        import torch.distributed as dist
        if dist.is_initialized():
            zero_stage = get_zero_stage()

            rank0_print("\n" + "=" * 70)
            rank0_print(f"🔄 DDP/ZeRO-{zero_stage}: 同步模型参数")
            rank0_print("=" * 70)

            if zero_stage >= 2:
                # ZeRO-2 或 ZeRO-3: DeepSpeed 自动处理参数同步
                # 不需要手动广播参数
                rank0_print(f"    ZeRO-{zero_stage}: 模型已由 DeepSpeed 管理")
                rank0_print("    (梯度/参数自动同步，无需手动广播)")
                if accelerator is not None:
                    accelerator.wait_for_everyone()
            else:
                # 普通 DDP (zero_stage == 0): 需要手动同步参数
                # 同步 Actor 的可训练参数
                actor_synced = sync_model_params(policy.actor, src_rank=0, verbose=True)
                rank0_print(f"    Actor: {actor_synced} parameters synced")

                # 同步 Critic 的可训练参数
                critic_synced = sync_model_params(policy.critic, src_rank=0, verbose=True)
                rank0_print(f"    Critic: {critic_synced} parameters synced")

                # 等待所有进程完成同步
                dist.barrier()

            rank0_print("✓ 参数同步完成，所有 GPU 从相同状态开始训练")
            rank0_print("=" * 70 + "\n")

    # 立即保存到 buffer，确保 VLM 服务器有完整的 checkpoint 可用
    # ZeRO-3 模式：所有进程都必须参与 save（因为 GatheredParameters 是集体操作）
    # 但只有 rank 0 执行实际的文件写入（由 rl.py 内部控制）
    rank0_print("    >>>> Saving policy to buffer (确保 VLM 服务器可访问)")
    policy.save(BUFFER_PATH, "policy")  # 所有进程参与 gather，rank 0 写文件

    # 计算 policy checkpoint 路径（与 rl.py 中的 save/load 逻辑一致）
    policy_checkpoint_path = join(BUFFER_PATH, "policy")
    if is_main_process():
        rank0_print(f"         Checkpoint 已保存到: {policy_checkpoint_path}")

    # ===== Step 4: 启动 tmux VLM 服务 =====
    # DDP 模式：只在主进程启动服务
    # 优先级：no_cleanup > 命令行参数 > 配置文件 > 默认值(True)
    start_services = config.get("ftrl_config", {}).get("auto_start_services", True)
    if args.start_services is not None:  # 命令行覆盖
        start_services = args.start_services
    if args.no_cleanup:  # no_cleanup 时强制跳过 tmux 和 containers
        start_services = False
        rank0_print(f"    >>>> [DEBUG] no_cleanup={args.no_cleanup}, 跳过启动 tmux 服务")

    if is_main_process():
        if start_services:
            rank0_print("\n" + "=" * 80)
            rank0_print("Step 4: 启动 tmux VLM 服务")
            rank0_print("=" * 80)
            # 传递实际的 checkpoint 路径（与 rl.py 保持一致）
            started_services = start_tmux_services(config, policy_checkpoint_path=policy_checkpoint_path)
            if started_services:
                rank0_print(f"    >>>> ✓ Started {len(started_services)} tmux VLM services")
                # 注册 atexit 清理函数，确保程序退出时关闭服务
                atexit.register(stop_tmux_services)
            else:
                rank0_print("    >>>> ✗ No tmux services started (check config or errors above)")
        else:
            rank0_print("\n" + "=" * 80)
            rank0_print("Step 4: 跳过启动 tmux VLM 服务 (--start_services=False)")
            rank0_print("=" * 80)

    # DDP 模式：同步所有进程
    if is_ddp_enabled() and accelerator is not None:
        accelerator.wait_for_everyone()

    # ===== Step 5: 开始训练 =====
    if is_main_process():
        rank0_print("\n" + "=" * 80)
        rank0_print("Step 5: 开始训练")
        rank0_print("=" * 80)
    train(env, policy, buffer, config, skip_test=args.skip_test)
