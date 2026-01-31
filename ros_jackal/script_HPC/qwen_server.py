#!/usr/bin/env python3
"""
Qwen2.5-VL Navigation Parameter Prediction - Switchable Checkpoint Version
基于 qwen_server_flash_attn.py，添加动态切换 checkpoint 功能
"""

import argparse
import os
import sys
import time
import json
import gc
import base64
import io
from typing import List, Dict, Any, Optional

import torch
import numpy as np
from peft import PeftModel
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# FastAPI imports
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# ===========================================================
# Efficiency Parameters
# ===========================================================
USE_FLASH_ATTENTION = True
USE_COMPILE = False

# ============================================================
# 算法参数配置
# ============================================================
ALGORITHM_PARAMS = {
    "DWA": {
        "max_vel_x": {"range": [0.2, 2.0], "type": "float"},
        "max_vel_theta": {"range": [0.314, 3.14], "type": "float"},
        "vx_samples": {"range": [4, 12], "type": "int"},
        "vtheta_samples": {"range": [8, 40], "type": "int"},
        "path_distance_bias": {"range": [0.1, 1.5], "type": "float"},
        "goal_distance_bias": {"range": [0.1, 2.0], "type": "float"},
        "inflation_radius": {"range": [0.1, 0.6], "type": "float"},
        "next_linear_vel": {"range": [-0.5, 2.0], "type": "float"},
        "next_angular_vel": {"range": [-3.14, 3.14], "type": "float"}
    },
    "TEB": {
        "max_vel_x": {"range": [0.5, 2.0], "type": "float"},
        "max_vel_x_backwards": {"range": [0.1, 1.0], "type": "float"},
        "max_vel_theta": {"range": [0.5, 3.0], "type": "float"},
        "dt_ref": {"range": [0.1, 0.5], "type": "float"},
        "min_obstacle_dist": {"range": [0.05, 0.3], "type": "float"},
        "inflation_dist": {"range": [0.1, 0.5], "type": "float"},
        "inflation_radius": {"range": [0.1, 0.4], "type": "float"},
        "next_linear_vel": {"range": [-0.5, 2.0], "type": "float"},
        "next_angular_vel": {"range": [-3.14, 3.14], "type": "float"}
    },
    "MPPI": {
        "max_vel_x": {"range": [0.5, 2.0], "type": "float"},
        "max_vel_theta": {"range": [0.5, 3.0], "type": "float"},
        "nr_pairs_": {"range": [200, 1000], "type": "int"},
        "nr_steps_": {"range": [10, 40], "type": "int"},
        "linear_stddev": {"range": [0.05, 0.3], "type": "float"},
        "angular_stddev": {"range": [0.02, 0.2], "type": "float"},
        "lambda": {"range": [0.5, 2.0], "type": "float"},
        "inflation_radius": {"range": [0.1, 0.4], "type": "float"},
        "next_linear_vel": {"range": [-0.5, 2.0], "type": "float"},
        "next_angular_vel": {"range": [-3.14, 3.14], "type": "float"}
    },
    "DDP": {
        "max_vel_x": {"range": [0.5, 2.0], "type": "float"},
        "max_vel_theta": {"range": [0.5, 3.0], "type": "float"},
        "nr_pairs_": {"range": [200, 1000], "type": "int"},
        "distance": {"range": [0.05, 0.3], "type": "float"},
        "robot_radius": {"range": [0.01, 0.1], "type": "float"},
        "inflation_radius": {"range": [0.1, 0.4], "type": "float"},
        "next_linear_vel": {"range": [-0.5, 2.0], "type": "float"},
        "next_angular_vel": {"range": [-3.14, 3.14], "type": "float"}
    }
}

# ============================================================
# FastAPI 数据模型
# ============================================================
class InferenceRequest(BaseModel):
    image_base64: Optional[str] = Field(None, description="Base64编码的图像（当前帧）")
    image_path: Optional[str] = Field(None, description="图像文件路径（完整路径，兼容旧版本）")
    history_images_base64: Optional[List[str]] = Field(None, description="历史帧的base64编码列表（按时间顺序）")
    # 以下字段保留用于向后兼容，但推荐使用 history_images_base64
    image_dir: Optional[str] = Field(None, description="图像所在目录（用于构建历史帧路径，旧版本）")
    image_filename: Optional[str] = Field(None, description="图像文件名（用于解析帧号和方法名，旧版本）")
    linear_vel: float = Field(default=0.0)
    angular_vel: float = Field(default=0.0)
    algorithm: str = Field(default="DWA")

class InferenceResponse(BaseModel):
    parameters: Dict[str, Any]
    parameters_array: List[float]
    hidden_states_shape: str
    inference_time: float
    success: bool
    checkpoint: str = "unknown"

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    algorithm: str
    policy_name: str
    current_checkpoint: str = "none"

class SwitchCheckpointRequest(BaseModel):
    checkpoint_path: str = Field(..., description="Checkpoint 路径（绝对路径或相对于 model/ 的路径）")
    algorithm: Optional[str] = Field(None, description="算法名称 (DWA/TEB/MPPI/DDP)")
    head_type: Optional[str] = Field(None, description="Head 类型 (dpt/transformer/simple_mlp)")
    num_params: Optional[int] = Field(None, description="参数数量")

class SwitchCheckpointResponse(BaseModel):
    success: bool
    message: str
    old_checkpoint: str
    new_checkpoint: str
    switch_time: float

class ListCheckpointsResponse(BaseModel):
    available_checkpoints: List[Dict[str, str]]
    current_checkpoint: str

# ============================================================
# Prompts
# ============================================================
SYSTEM_PROMPT = """
You are a navigation scene analyzer for Clearpath Jackal robot motion planning.
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

Your navigation reasoning will be used downstream to select parameters for the local motion planner.
"""

USER_PROMPT = """
Current robot state:
- Linear velocity: {linear_vel:.3f} m/s
- Angular velocity: {angular_vel:.3f} rad/s

Target local planner: {algorithm}

Use the scene image and robot state to perform navigation reasoning about obstacle proximity, path curvature, free-space structure, and motion constraints.
"""

# ============================================================
# 全局状态
# ============================================================
class ModelState:
    def __init__(self):
        self.base_model = None
        self.peft_model = None
        self.processor = None
        self.regression_head = None
        self.param_mean = None
        self.param_std = None
        self.current_checkpoint = None
        self.algorithm = None
        self.head_type = "dpt"
        self.num_params = 6
        self.device = None
        self.dtype = None
        self.config = None

state = ModelState()

# ============================================================
# 工具函数
# ============================================================
def clear_gpu_memory():
    """清理 GPU 内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

@torch.no_grad()
def forward_with_hidden_states(model, processor, inputs, num_layers=4, layer_indices=None):
    """
    提取 Qwen2.5-VL 的多层 hidden states（用于 DPT head）
    与 qwen_server_flash_attn.py 完全一致
    """
    model_inputs = {
        "input_ids": inputs["input_ids"],
        "pixel_values": inputs.get("pixel_values"),
        "image_grid_thw": inputs.get("image_grid_thw"),
        "video_grid_thw": inputs.get("video_grid_thw"),
        "attention_mask": inputs.get("attention_mask"),
        "output_hidden_states": True,
    }

    # 关键：调用 model.model() 而不是 model()
    outputs = model.model(**model_inputs)

    all_hidden_states = outputs.hidden_states

    if layer_indices is None:
        layer_indices = list(range(-num_layers, 0))

    multi_layer_hidden_states = [all_hidden_states[idx] for idx in layer_indices]

    return multi_layer_hidden_states

# ============================================================
# 模型加载函数
# ============================================================
def load_base_model(config):
    """加载基础模型（只加载一次）"""
    print("=" * 60)
    print("Loading Qwen2.5-VL Base Model...")
    print("=" * 60)

    start_time = time.time()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if torch.cuda.is_available():
        try:
            major_cc, _ = torch.cuda.get_device_capability(0)
            dtype = torch.bfloat16 if major_cc >= 8 else torch.float16
        except Exception:
            dtype = torch.float16
    else:
        dtype = torch.float32

    print(f"Base model: {config.base_model}")
    print(f"Using dtype: {dtype}")

    # 检查 flash_attn 是否可用
    flash_attn_available = False
    try:
        import flash_attn
        flash_attn_available = True
        print("✓ flash_attn detected")
    except ImportError:
        print("⚠️  flash_attn not installed, using SDPA instead")

    # 选择 attention 实现
    if USE_FLASH_ATTENTION and flash_attn_available:
        attn_impl = "flash_attention_2"
    elif USE_FLASH_ATTENTION and not flash_attn_available:
        print("⚠️  Falling back to SDPA (flash_attn not available)")
        attn_impl = "sdpa"
    else:
        attn_impl = "default"

    resolved_device_map = config.device_map if getattr(config, 'device_map', None) else "auto"

    if getattr(config, 'load_in_4bit', False):
        qdtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
        print(f"[LOAD] Loading model with 4-bit quantization, device_map={resolved_device_map}")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.base_model,
            attn_implementation=attn_impl,
            device_map=resolved_device_map,
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=qdtype,
            bnb_4bit_use_double_quant=True,
        )
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.base_model,
            attn_implementation=attn_impl,
            torch_dtype=dtype,
            device_map=resolved_device_map,
        )

    processor = AutoProcessor.from_pretrained(
        config.base_model,
        min_pixels=256 * 28 * 28,
        max_pixels=384 * 28 * 28
    )

    state.base_model = model
    state.processor = processor
    state.device = model.device
    state.dtype = model.dtype
    state.config = config

    load_time = time.time() - start_time
    print(f"✓ Base model loaded in {load_time:.2f}s")
    print(f"✓ Device: {model.device}, Dtype: {model.dtype}")

def load_checkpoint(checkpoint_path, algorithm=None, head_type="dpt", num_params=6):
    """加载 LoRA checkpoint 和 regression head"""
    print("=" * 60)
    print(f"Loading Checkpoint: {checkpoint_path}")
    print("=" * 60)

    start_time = time.time()

    # 解析路径
    if not os.path.isabs(checkpoint_path):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        checkpoint_path = os.path.join(base_dir, "model", checkpoint_path)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # 1. 加载 LoRA
    print(f"Loading LoRA from: {checkpoint_path}")
    try:
        peft_model = PeftModel.from_pretrained(
            state.base_model,
            checkpoint_path,
            is_trainable=False
        )
    except Exception as e:
        print(f"[WARN] PEFT loading failed: {e}")
        print("[INFO] Trying to load with filtered weights...")

        from peft import get_peft_model, LoraConfig
        from safetensors.torch import load_file

        adapter_config_path = os.path.join(checkpoint_path, 'adapter_config.json')
        with open(adapter_config_path) as f:
            adapter_config = json.load(f)

        lora_config = LoraConfig(
            r=adapter_config['r'],
            lora_alpha=adapter_config['lora_alpha'],
            target_modules=adapter_config['target_modules'],
            lora_dropout=adapter_config.get('lora_dropout', 0.0),
            bias=adapter_config.get('bias', 'none'),
            task_type=adapter_config.get('task_type', 'CAUSAL_LM'),
        )

        peft_model = get_peft_model(state.base_model, lora_config)

        adapter_weights_path = os.path.join(checkpoint_path, 'adapter_model.safetensors')
        lora_dict = load_file(adapter_weights_path)
        filtered_dict = {k: v for k, v in lora_dict.items() if v.numel() > 0}
        print(f"[INFO] Loading {len(filtered_dict)}/{len(lora_dict)} non-empty LoRA weights")
        peft_model.load_state_dict(filtered_dict, strict=False)

    peft_model.eval()
    state.peft_model = peft_model

    # 2. 加载 Regression head
    regression_head_path = os.path.join(checkpoint_path, 'regression_head', 'pytorch_model.bin')
    if os.path.exists(regression_head_path):
        print(f"Loading regression head: {head_type}, num_params={num_params}")

        # 导入 regression head 类
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # script_HPC/ -> ros_jackal/ -> vlm_pipeline/lmms-finetune-qwen
        vlm_pipeline_path = os.path.abspath(os.path.join(script_dir, '../../vlm_pipeline/lmms-finetune-qwen'))
        if vlm_pipeline_path not in sys.path:
            sys.path.insert(0, vlm_pipeline_path)

        from models.qwen2_5_vl_dpt_regression import DPTHead, SimpleMLP, TransformerHead

        hidden_size = peft_model.config.hidden_size
        print(f"✓ Detected hidden_size: {hidden_size}")

        # 读取历史配置（如果存在）
        import json
        history_config_path = os.path.join(os.path.dirname(regression_head_path), '../history_config.json')
        use_history = False
        history_dim = 256
        if os.path.exists(history_config_path):
            with open(history_config_path) as f:
                history_config = json.load(f)
                use_history = history_config.get('use_history', False)
                history_dim = history_config.get('history_dim', 256)
                print(f"✓ Loaded history config: use_history={use_history}, history_dim={history_dim}")

        if head_type == 'dpt':
            regression_head = DPTHead(
                hidden_size=hidden_size,
                num_params=num_params,
                feature_dim=256,
                num_layers=4,
                use_history=use_history,
                history_dim=history_dim,
            )
        elif head_type == 'transformer':
            regression_head = TransformerHead(hidden_size=hidden_size, num_params=num_params)
        else:
            regression_head = SimpleMLP(hidden_size=hidden_size, num_params=num_params)

        # 加载权重
        reg_dict = torch.load(regression_head_path, map_location='cpu', weights_only=False)
        filtered_dict = {k: v for k, v in reg_dict.items() if v.numel() > 0}
        num_empty = len(reg_dict) - len(filtered_dict)

        if num_empty > 0:
            print(f"[WARN] Found {num_empty}/{len(reg_dict)} empty tensors in regression_head (DeepSpeed bug)")
            print(f"[INFO] Loading {len(filtered_dict)}/{len(reg_dict)} non-empty weights with strict=False")
            missing_keys, unexpected_keys = regression_head.load_state_dict(filtered_dict, strict=False)
            if missing_keys:
                print(f"[WARN] Missing keys (will use random init): {missing_keys[:5]}...")
        else:
            regression_head.load_state_dict(reg_dict)

        regression_head.to(device=state.device, dtype=state.dtype)
        regression_head.eval()
        state.regression_head = regression_head
        print(f"✓ Regression head loaded: {head_type}")

    # 3. 加载归一化参数
    normalization_dir = os.path.join(checkpoint_path, 'normalization')
    param_mean_path = os.path.join(normalization_dir, 'param_mean.npy')
    param_std_path = os.path.join(normalization_dir, 'param_std.npy')

    if os.path.exists(param_mean_path) and os.path.exists(param_std_path):
        param_mean = np.load(param_mean_path)
        param_std = np.load(param_std_path)
        state.param_mean = torch.tensor(param_mean, dtype=state.dtype, device=state.device)
        state.param_std = torch.tensor(param_std, dtype=state.dtype, device=state.device)
        print(f"✓ Normalization stats loaded")
        print(f"  mean: {param_mean}")
        print(f"  std:  {param_std}")
    else:
        print(f"⚠️  Normalization stats not found")
        state.param_mean = None
        state.param_std = None

    state.current_checkpoint = checkpoint_path
    state.algorithm = algorithm or "DDP"
    state.head_type = head_type
    state.num_params = num_params

    load_time = time.time() - start_time
    print(f"✓ Checkpoint loaded in {load_time:.2f}s")
    print("=" * 60)

def unload_checkpoint():
    """卸载当前 checkpoint"""
    print("[UNLOAD] Removing current checkpoint...")

    if state.peft_model is not None:
        # 对于 PEFT 模型，需要 unmerge 并删除
        if hasattr(state.peft_model, 'unload'):
            try:
                state.peft_model.unload()
            except:
                pass
        del state.peft_model
        state.peft_model = None

    if state.regression_head is not None:
        del state.regression_head
        state.regression_head = None

    state.param_mean = None
    state.param_std = None
    state.current_checkpoint = None

    clear_gpu_memory()
    print("[UNLOAD] Done")

# ============================================================
# FastAPI 应用
# ============================================================
app = FastAPI(title="Qwen2.5-VL Switchable Checkpoint Service")

@app.on_event("startup")
async def startup():
    """启动时加载基础模型和初始 checkpoint"""
    config = app.state.config

    # 加载基础模型
    load_base_model(config)

    # 加载初始 checkpoint
    if config.lora_path:
        load_checkpoint(
            config.lora_path,
            algorithm=config.algorithm,
            head_type=config.head_type,
            num_params=config.num_params
        )

    print("=" * 60)
    print("✓ Service ready!")
    print("=" * 60)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    alg = state.algorithm or "unknown"
    policy = f"{alg.lower()}_qwen" if alg != "unknown" else "unknown"
    return HealthResponse(
        status="ok" if state.peft_model is not None else "no_checkpoint",
        model_loaded=state.peft_model is not None,
        device=str(state.device) if state.device else "unknown",
        algorithm=alg,
        policy_name=policy,
        current_checkpoint=state.current_checkpoint or "none",
    )

@app.get("/config")
async def get_config():
    """返回模型配置信息，包括历史帧数量"""
    if state.peft_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    config_info = {
        "algorithm": state.algorithm or "DWA",
        "num_params": state.peft_model.regression_head.num_params if hasattr(state.peft_model, 'regression_head') else 7,
        "use_history": hasattr(state.peft_model, 'history_encoder') and state.peft_model.history_encoder is not None,
        "num_history_frames": state.peft_model.num_history_frames if hasattr(state.peft_model, 'num_history_frames') else 0,
        "history_dim": state.peft_model.history_encoder.hidden_dim if hasattr(state.peft_model, 'history_encoder') and state.peft_model.history_encoder else 0,
        "current_checkpoint": state.current_checkpoint or "none",
    }
    return config_info

@app.post("/switch_checkpoint", response_model=SwitchCheckpointResponse)
async def switch_checkpoint(request: SwitchCheckpointRequest):
    """切换到新的 checkpoint"""
    start_time = time.time()
    old_checkpoint = state.current_checkpoint or "none"

    try:
        # 卸载旧的
        unload_checkpoint()

        # 加载新的
        load_checkpoint(
            request.checkpoint_path,
            algorithm=request.algorithm,
            head_type=request.head_type or "dpt",
            num_params=request.num_params or 6
        )

        switch_time = time.time() - start_time

        return SwitchCheckpointResponse(
            success=True,
            message=f"Switched checkpoint in {switch_time:.2f}s",
            old_checkpoint=old_checkpoint,
            new_checkpoint=state.current_checkpoint,
            switch_time=switch_time
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return SwitchCheckpointResponse(
            success=False,
            message=f"Failed: {str(e)}",
            old_checkpoint=old_checkpoint,
            new_checkpoint="none",
            switch_time=time.time() - start_time
        )

@app.get("/list_checkpoints", response_model=ListCheckpointsResponse)
async def list_checkpoints():
    """列出可用的 checkpoints"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, "model")

    checkpoints = []
    for planner in ["dwa", "teb", "mppi", "ddp"]:
        planner_dir = os.path.join(model_dir, planner)
        if os.path.exists(planner_dir):
            for item in os.listdir(planner_dir):
                item_path = os.path.join(planner_dir, item)
                if os.path.isdir(item_path):
                    # 检查是否包含 checkpoint
                    for sub in os.listdir(item_path):
                        if sub.startswith("checkpoint-"):
                            checkpoints.append({
                                "path": f"{planner}/{item}/{sub}",
                                "full_path": os.path.join(item_path, sub),
                                "planner": planner.upper(),
                                "name": sub
                            })

    return ListCheckpointsResponse(
        available_checkpoints=checkpoints,
        current_checkpoint=state.current_checkpoint or "none"
    )

@app.post("/infer", response_model=InferenceResponse)
async def infer(request: InferenceRequest):
    """推理端点 - 与 qwen_server_flash_attn.py 完全一致"""
    if state.peft_model is None or state.regression_head is None:
        raise HTTPException(status_code=503, detail="No checkpoint loaded")

    start_time = time.time()
    print("\n" + "="*60)
    print(f"[INFER] Processing request for {request.algorithm}")
    print("="*60)

    # 1. 加载当前图像和历史图像
    t1_start = time.time()

    try:
        # 优先使用 image_base64（跨 Singularity 容器安全）
        if request.image_base64:
            image_data = base64.b64decode(request.image_base64)
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
        elif request.image_path:
            # Fallback: 直接从路径读取（兼容旧版本）
            image_path_full = os.path.abspath(request.image_path)
            if not os.path.exists(image_path_full):
                raise FileNotFoundError(f"Image not found: {image_path_full}")
            image = Image.open(image_path_full).convert("RGB")
        else:
            raise HTTPException(status_code=400, detail="Must provide image_base64 or image_path")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load image: {str(e)}")

    # 2. 处理历史帧（如果使用 history_encoder）
    history_images_list = []
    if hasattr(state.regression_head, 'use_history') and state.regression_head.use_history:
        # 优先使用客户端传递的 history_images_base64（推荐方式）
        if request.history_images_base64 and len(request.history_images_base64) > 0:
            print(f"[INFER] 📸 Using {len(request.history_images_base64)} history frames from client (base64)")
            for idx, hist_base64 in enumerate(request.history_images_base64):
                try:
                    hist_data = base64.b64decode(hist_base64)
                    hist_img = Image.open(io.BytesIO(hist_data)).convert("RGB")
                    history_images_list.append(hist_img)
                except Exception as e:
                    print(f"[WARN] Failed to decode history frame {idx}: {e}, using current frame")
                    history_images_list.append(image)

        # Fallback: 从服务端文件系统读取（旧版本，需要路径映射）
        elif request.image_filename and request.image_dir:
            import re

            filename = request.image_filename
            dir_path = request.image_dir

            print(f"[INFER] 📂 Reading history frames from server filesystem (legacy mode)")

            # 解析文件名：例如 HB_000025.png -> (HB, 25)
            match = re.match(r'([A-Z]+)_(\d{6})\.png', filename)

            if match:
                method = match.group(1)
                frame_id = int(match.group(2))

                # 读取前 num_history_frames 帧
                num_history = getattr(state, 'num_history_frames', 2)
                for i in range(num_history, 0, -1):  # 从最早的历史帧开始
                    hist_frame_id = frame_id - i
                    hist_filename = f"{method}_{hist_frame_id:06d}.png"
                    hist_path = os.path.join(dir_path, hist_filename)

                    # 如果历史帧存在且帧号>=0，读取历史帧；否则用当前帧填充
                    if hist_frame_id >= 0 and os.path.exists(hist_path):
                        hist_img = Image.open(hist_path).convert("RGB")
                        history_images_list.append(hist_img)
                    else:
                        # 历史帧不存在或帧号<0，使用当前帧填充
                        history_images_list.append(image)

                print(f"[INFER] 📸 Loaded {len(history_images_list)} history frames for {filename}")
            else:
                # 无法解析文件名，用当前帧填充所有历史帧
                print(f"[WARN] Could not parse filename: {filename}, using current frame for all history")
                num_history = getattr(state, 'num_history_frames', 2)
                history_images_list = [image] * num_history
        else:
            # 没有提供任何历史帧信息，用当前帧填充
            print(f"[WARN] No history frames provided, using current frame for all history")
            num_history = getattr(state, 'num_history_frames', 2)
            history_images_list = [image] * num_history

    t1 = time.time() - t1_start
    print(f"[INFER] ⏱️  Image loading: {t1 * 1000:.1f}ms (current: {image.size}, history: {len(history_images_list)} frames)")

    # 2. 构建 prompt
    t2_start = time.time()
    system_prompt = SYSTEM_PROMPT.strip()
    user_prompt = USER_PROMPT.format(
        linear_vel=request.linear_vel,
        angular_vel=request.angular_vel,
        algorithm=request.algorithm
    )
    t2 = time.time() - t2_start

    print(f"[INFER] ⏱️  Prompt building: {t2 * 1000:.1f}ms (length: {len(user_prompt)} chars)")

    # 3. 准备输入
    t3_start = time.time()

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt.strip()},
            ],
        }
    ]

    # 3.1 应用chat template
    t3_template_start = time.time()
    text = state.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    t3_template = time.time() - t3_template_start

    # 3.2 处理器 (vision + text) - 与训练时格式完全一致
    t3_process_start = time.time()
    inputs = state.processor(
        text=[text],
        images=[image],
        videos=None,
        padding=False,
        return_tensors="pt"
    )
    t3_process = time.time() - t3_process_start

    # 3.3 移动到设备
    t3_to_device_start = time.time()
    inputs = inputs.to(state.device)
    t3_to_device = time.time() - t3_to_device_start

    t3 = time.time() - t3_start
    print(f"[INFER] ⏱️  Input preparation: {t3 * 1000:.1f}ms")
    print(f"[INFER]    ├─ Template: {t3_template * 1000:.1f}ms")
    print(f"[INFER]    ├─ Processor (vision+text): {t3_process * 1000:.1f}ms")
    print(f"[INFER]    └─ To device: {t3_to_device * 1000:.1f}ms")

    # 🔍 输入信息
    print(f"[INFER] 📸 Original image size: {image.size}")
    print(f"[INFER] 📸 Processor config: min_pixels={state.processor.image_processor.min_pixels}, max_pixels={state.processor.image_processor.max_pixels}")
    if 'pixel_values' in inputs:
        pix_shape = inputs['pixel_values'].shape
        print(f"[INFER] 📸 Vision shape (pixel_values): {pix_shape}")
    if 'image_grid_thw' in inputs and inputs['image_grid_thw'] is not None:
        grid_thw = inputs['image_grid_thw']
        print(f"[INFER] 📸 Image grid (T×H×W): {grid_thw}")
        if grid_thw.numel() >= 3:
            t, h, w = grid_thw[0, 0].item(), grid_thw[0, 1].item(), grid_thw[0, 2].item()
            visual_tokens_raw = t * h * w
            print(f"[INFER] 📸 Raw visual tokens (before compression): {visual_tokens_raw} ({t}×{h}×{w})")
    if 'input_ids' in inputs:
        total_tokens = inputs['input_ids'].shape[1]
        print(f"[INFER] 📝 Final sequence length (after vision compression): {total_tokens}")

    # 4. 模型推理（提取多层 hidden states）
    t4_start = time.time()
    multi_layer_hidden_states = forward_with_hidden_states(
        state.peft_model,
        state.processor,
        inputs,
        num_layers=4,  # 提取最后 4 层
        layer_indices=[-4, -3, -2, -1]  # 或者指定具体层
    )
    t4 = time.time() - t4_start
    print(f"[INFER] ⏱️  forward_with_hidden_states(): {t4:.3f}s")
    print(f"[INFER] 📦 Extracted {len(multi_layer_hidden_states)} layers")

    # 5. 处理历史帧（如果使用 use_history）
    history_feat = None
    if hasattr(state, 'history_encoder') and state.history_encoder is not None and len(history_images_list) > 0:
        import torchvision.transforms as T

        # 读取 history config 中的 image size
        history_config_path = os.path.join(os.path.dirname(state.current_checkpoint), 'history_config.json')
        if os.path.exists(history_config_path):
            import json
            with open(history_config_path) as f:
                history_config = json.load(f)
                history_img_size = history_config.get('history_image_size', 224)
        else:
            history_img_size = 224

        # 转换历史帧为 tensor: [num_frames, 3, H, W]
        transform = T.Compose([
            T.Resize((history_img_size, history_img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        history_tensors = [transform(img) for img in history_images_list]
        history_images = torch.stack(history_tensors, dim=0).unsqueeze(0)  # [1, num_frames, 3, H, W]
        history_images = history_images.to(state.device, dtype=state.peft_model.dtype)

        # 提取历史特征
        with torch.no_grad():
            history_feat = state.history_encoder(history_images)  # [1, history_dim]

        print(f"[INFER] ✓ Extracted history features: shape={history_feat.shape}, num_frames={len(history_images_list)}")

    # 6. 使用 regression head 预测参数
    t5_start = time.time()
    param_config = ALGORITHM_PARAMS.get(request.algorithm, ALGORITHM_PARAMS["DWA"])
    param_order = list(param_config.keys())

    # ✅ 真正调用 regression head 进行预测
    if hasattr(state, 'regression_head'):
        head_type = getattr(state, 'head_type', 'dpt')

        with torch.no_grad():
            if head_type == 'dpt':
                # DPT head 需要多层 hidden states (List[Tensor]) + 历史特征（可选）
                predicted_params = state.regression_head(multi_layer_hidden_states, history_feat)
            else:
                # SimpleMLP 和 TransformerHead 只需要最后一层
                last_hidden_state = multi_layer_hidden_states[-1]
                predicted_params = state.regression_head(last_hidden_state)

            # 反归一化（恢复到原始参数范围）
            if hasattr(state, 'param_mean') and state.param_mean is not None:
                predicted_params_normalized = predicted_params.squeeze(0)  # [num_params]
                predicted_params_denorm = predicted_params_normalized * state.param_std + state.param_mean
                predicted_params = predicted_params_denorm.cpu().tolist()
                print(f"[INFER] ✓ Denormalized predictions")
            else:
                predicted_params = predicted_params.squeeze(0).cpu().tolist()
                print(f"[INFER] ⚠️  No normalization stats, returning raw predictions")

        print(f"[INFER] ✓ Regression head ({head_type}) predicted {len(predicted_params)} parameters")
    else:
        # 如果没有 regression head，使用占位符
        predicted_params = [1.0] * len(param_order)
        print("[INFER] ⚠️  Regression head not found, using dummy parameters")

    params_dict = {k: v for k, v in zip(param_order, predicted_params)}

    # 格式化 hidden states 信息
    hidden_states_shapes = [str(h.shape) for h in multi_layer_hidden_states]
    hidden_states_shape = f"{len(multi_layer_hidden_states)} layers: {', '.join(hidden_states_shapes)}"
    t5 = time.time() - t5_start
    print(f"[INFER] ⏱️  Regression prediction: {t5 * 1000:.1f}ms")

    inference_time = time.time() - start_time

    # 📊 性能总结
    print("\n" + "="*60)
    print("📊 INFERENCE PERFORMANCE SUMMARY")
    print("="*60)
    print(f"⏱️  Total time:        {inference_time:.3f}s")
    print(f"   ├─ Image load:      {t1*1000:.1f}ms ({t1/inference_time*100:.1f}%)")
    print(f"   ├─ Prompt build:    {t2*1000:.1f}ms ({t2/inference_time*100:.1f}%)")
    print(f"   ├─ Input prep:      {t3*1000:.1f}ms ({t3/inference_time*100:.1f}%)")
    print(f"   ├─ Model forward:   {t4:.3f}s ({t4/inference_time*100:.1f}%)")
    print(f"   └─ Output prep:     {t5*1000:.1f}ms ({t5/inference_time*100:.1f}%)")
    print(f"📦 Hidden states:     {hidden_states_shape}")
    print(params_dict)
    print("="*60 + "\n")

    return InferenceResponse(
        parameters=params_dict,
        parameters_array=predicted_params,
        hidden_states_shape=hidden_states_shape,
        inference_time=inference_time,
        success=True,
        checkpoint=state.current_checkpoint or "none"  # 返回 checkpoint 路径
    )

# ============================================================
# 主函数
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL Switchable Checkpoint Service")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--lora_path", type=str, default=None, help="初始 checkpoint 路径")
    parser.add_argument("--head_type", type=str, default="dpt", choices=["simple_mlp", "transformer", "dpt"])
    parser.add_argument("--num_params", type=int, default=6)
    parser.add_argument("--algorithm", type=str, default="DDP")
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--load_in_4bit", action="store_true", default=True)
    parser.add_argument("--port", type=int, default=5000)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    app.state.config = args

    print("=" * 60)
    print("Starting Qwen2.5-VL Switchable Checkpoint Service")
    print("=" * 60)
    print(f"Base Model:  {args.base_model}")
    print(f"Checkpoint:  {args.lora_path or 'none'}")
    print(f"Algorithm:   {args.algorithm}")
    print(f"Head Type:   {args.head_type}")
    print(f"Num Params:  {args.num_params}")
    print(f"Port:        {args.port}")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
