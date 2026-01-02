#!/usr/bin/env python3
"""
Qwen2.5-VL 动态切换 Checkpoint 服务
支持在运行时切换不同的 LoRA checkpoint 和 regression head
"""

import torch
import gc
import os
import time
import base64
import io
import argparse
from typing import Optional, Dict, Any, List
from pathlib import Path
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import numpy as np

from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

# 导入模型相关
import sys
vlm_pipeline_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../vlm_pipeline/lmms-finetune-qwen'))
if vlm_pipeline_path not in sys.path:
    sys.path.insert(0, vlm_pipeline_path)

from transformers import Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
from models.qwen2_5_vl_dpt_regression import DPTHead, TransformerHead, SimpleMLP

# ============================================================
# 数据模型
# ============================================================
class InferenceRequest(BaseModel):
    image_base64: Optional[str] = Field(None, description="Base64编码的图像")
    image_path: Optional[str] = Field(None, description="图像文件路径")
    linear_vel: float = Field(default=0.0)
    angular_vel: float = Field(default=0.0)
    algorithm: str = Field(default="DWA")

class InferenceResponse(BaseModel):
    parameters: Dict[str, Any]
    parameters_array: List[float]
    checkpoint: str
    inference_time: float
    success: bool

class SwitchCheckpointRequest(BaseModel):
    checkpoint_path: str = Field(..., description="新checkpoint路径（相对于 model/planner/ 或绝对路径）")
    algorithm: Optional[str] = Field(None, description="算法名称（DWA/TEB/MPPI/DDP）")
    head_type: Optional[str] = Field(None, description="Head类型（dpt/transformer/simple_mlp）")
    num_params: Optional[int] = Field(None, description="参数数量")

class SwitchCheckpointResponse(BaseModel):
    success: bool
    message: str
    old_checkpoint: str
    new_checkpoint: str
    switch_time: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    current_checkpoint: str
    device: str
    algorithm: str

class ListCheckpointsResponse(BaseModel):
    available_checkpoints: List[Dict[str, str]]
    current_checkpoint: str

# ============================================================
# 算法参数配置
# ============================================================
ALGORITHM_PARAMS = {
    "DWA": {
        "max_vel_x": 1.0,
        "max_vel_theta": 2.0,
        "vx_samples": 10,
        "vtheta_samples": 20,
        "path_distance_bias": 1.0,
        "goal_distance_bias": 1.0,
    },
    "TEB": {
        "max_vel_x": 0.5,
        "max_vel_x_backwards": 0.2,
        "max_vel_theta": 1.0,
        "dt_ref": 0.3,
        "min_obstacle_dist": 0.5,
        "inflation_dist": 0.6,
        "inflation_radius": 0.3,
    },
    "MPPI": {
        "max_vel_x": 1.0,
        "max_vel_theta": 2.0,
        "nr_pairs_": 500,
        "distance": 0.1,
        "robot_radius": 0.05,
        "inflation_radius": 0.15,
    },
    "DDP": {
        "max_vel_x": 1.8,
        "max_vel_theta": 2.8,
        "nr_pairs_": 767,
        "distance": 0.028,
        "robot_radius": 0.019,
        "inflation_radius": 0.124,
    },
}

SYSTEM_PROMPT = """You are a navigation scene analyzer for autonomous mobile robots..."""
USER_PROMPT = """Current robot state:
- Linear velocity: {linear_vel:.3f} m/s
- Angular velocity: {angular_vel:.3f} rad/s
Target local planner: {algorithm}"""

# ============================================================
# 全局状态
# ============================================================
class ModelState:
    def __init__(self):
        self.base_model = None
        self.processor = None
        self.current_checkpoint = None
        self.algorithm = None
        self.device = None
        self.dtype = None
        self.base_model_path = None

        # LoRA + Regression head
        self.lora_model = None
        self.regression_head = None
        self.param_mean = None
        self.param_std = None

state = ModelState()

# ============================================================
# 辅助函数
# ============================================================
@torch.no_grad()
def forward_with_hidden_states(model, processor, inputs, num_layers=4):
    """提取多层 hidden states"""
    outputs = model(**inputs, output_hidden_states=True)
    all_hidden_states = outputs.hidden_states

    # 提取最后 num_layers 层
    selected_layers = all_hidden_states[-num_layers:]
    return selected_layers

def clear_gpu_memory():
    """清理 GPU 内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def load_base_model_once(base_model_path, device_map="auto", load_in_4bit=True):
    """加载基础模型（只加载一次）"""
    if state.base_model is not None:
        print("[INFO] Base model already loaded, skipping...")
        return

    print(f"[LOAD] Loading base model: {base_model_path}")

    # 量化配置
    from transformers import BitsAndBytesConfig
    quant_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        quantization_config=quant_config if load_in_4bit else None,
        attn_implementation="flash_attention_2"
    )

    state.base_model = model
    state.base_model_path = base_model_path
    state.device = model.device
    state.dtype = model.dtype

    # 加载 processor
    state.processor = AutoProcessor.from_pretrained(
        base_model_path,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28
    )

    print(f"✓ Base model loaded: {base_model_path}")
    print(f"  Device: {state.device}, Dtype: {state.dtype}")

def unload_checkpoint():
    """卸载当前 checkpoint（LoRA + regression head）"""
    if state.lora_model is not None:
        print("[UNLOAD] Removing LoRA weights...")
        # 移除 PEFT adapter
        if hasattr(state.lora_model, 'unload'):
            state.lora_model.unload()
        del state.lora_model
        state.lora_model = None

    if state.regression_head is not None:
        print("[UNLOAD] Removing regression head...")
        del state.regression_head
        state.regression_head = None

    state.param_mean = None
    state.param_std = None
    state.current_checkpoint = None
    state.algorithm = None

    clear_gpu_memory()

def load_checkpoint(checkpoint_path, algorithm=None, head_type="dpt", num_params=6):
    """加载新的 checkpoint"""
    print(f"[LOAD] Loading checkpoint: {checkpoint_path}")

    # 1. 检查路径
    if not os.path.isabs(checkpoint_path):
        # 相对路径，拼接 model/ 目录
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        checkpoint_path = os.path.join(base_dir, "model", checkpoint_path)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # 2. 加载 LoRA
    print(f"[LOAD] Loading LoRA from {checkpoint_path}")
    try:
        lora_model = PeftModel.from_pretrained(
            state.base_model,
            checkpoint_path,
            is_trainable=False
        )
    except Exception as e:
        # Fallback: 过滤空 tensor
        print(f"[WARN] PEFT loading failed: {e}")
        print("[INFO] Trying to load with filtered weights...")

        from safetensors.torch import load_file
        adapter_path = os.path.join(checkpoint_path, 'adapter_model.safetensors')
        lora_dict = load_file(adapter_path)
        filtered_dict = {k: v for k, v in lora_dict.items() if v.numel() > 0}

        print(f"[INFO] Loaded {len(filtered_dict)}/{len(lora_dict)} non-empty LoRA weights")
        state.base_model.load_state_dict(filtered_dict, strict=False)
        lora_model = state.base_model

    state.lora_model = lora_model

    # 3. 加载 regression head
    regression_head_path = os.path.join(checkpoint_path, 'regression_head', 'pytorch_model.bin')

    if os.path.exists(regression_head_path):
        print(f"[LOAD] Loading regression head: {head_type}")

        hidden_size = state.base_model.config.hidden_size

        if head_type == 'dpt':
            regression_head = DPTHead(
                hidden_size=hidden_size,
                num_params=num_params,
                feature_dim=256,
                num_layers=4
            )
        elif head_type == 'transformer':
            regression_head = TransformerHead(hidden_size=hidden_size, num_params=num_params)
        else:
            regression_head = SimpleMLP(hidden_size=hidden_size, num_params=num_params)

        # 加载权重（过滤空 tensor）
        reg_dict = torch.load(regression_head_path, map_location='cpu', weights_only=False)
        filtered_dict = {k: v for k, v in reg_dict.items() if v.numel() > 0}
        regression_head.load_state_dict(filtered_dict, strict=False)

        regression_head.to(device=state.device, dtype=state.dtype)
        regression_head.eval()

        state.regression_head = regression_head
        print(f"✓ Regression head loaded: {head_type}")

    # 4. 加载归一化参数
    norm_dir = os.path.join(checkpoint_path, 'normalization')
    mean_path = os.path.join(norm_dir, 'param_mean.npy')
    std_path = os.path.join(norm_dir, 'param_std.npy')

    if os.path.exists(mean_path) and os.path.exists(std_path):
        param_mean = np.load(mean_path)
        param_std = np.load(std_path)

        state.param_mean = torch.tensor(param_mean, dtype=state.dtype, device=state.device)
        state.param_std = torch.tensor(param_std, dtype=state.dtype, device=state.device)

        print(f"✓ Normalization stats loaded")
        print(f"  mean: {param_mean}")
        print(f"  std: {param_std}")

    state.current_checkpoint = checkpoint_path
    state.algorithm = algorithm or "DWA"

    print(f"✓ Checkpoint loaded: {checkpoint_path}")

# ============================================================
# FastAPI 应用
# ============================================================
app = FastAPI(title="Qwen Dynamic Checkpoint Service")

@app.on_event("startup")
async def startup():
    """启动时加载基础模型和默认 checkpoint"""
    global state

    # 从命令行参数加载配置
    config = app.state.config

    # 加载基础模型（只加载一次）
    load_base_model_once(
        config.base_model,
        device_map=config.device_map,
        load_in_4bit=config.load_in_4bit
    )

    # 加载默认 checkpoint
    if config.checkpoint_path:
        load_checkpoint(
            config.checkpoint_path,
            algorithm=config.algorithm,
            head_type=config.head_type,
            num_params=config.num_params
        )

    print("="*60)
    print("✓ Service ready!")
    print("="*60)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="ok" if state.lora_model is not None else "no_checkpoint",
        model_loaded=state.lora_model is not None,
        current_checkpoint=state.current_checkpoint or "none",
        device=str(state.device) if state.device else "unknown",
        algorithm=state.algorithm or "unknown"
    )

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
                if item.startswith("checkpoint-"):
                    checkpoints.append({
                        "path": f"{planner}/{item}",
                        "full_path": os.path.join(planner_dir, item),
                        "planner": planner.upper(),
                        "name": item
                    })

    return ListCheckpointsResponse(
        available_checkpoints=checkpoints,
        current_checkpoint=state.current_checkpoint or "none"
    )

@app.post("/infer", response_model=InferenceResponse)
async def infer(request: InferenceRequest):
    if state.lora_model is None or state.regression_head is None:
        raise HTTPException(status_code=503, detail="No checkpoint loaded")

    start_time = time.time()

    # 1. 加载图像
    if request.image_base64:
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
    elif request.image_path:
        image = Image.open(request.image_path).convert("RGB")
    else:
        raise HTTPException(status_code=400, detail="No image provided")

    # 2. 构建 prompt
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT.strip()}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": USER_PROMPT.format(
                    linear_vel=request.linear_vel,
                    angular_vel=request.angular_vel,
                    algorithm=request.algorithm
                ).strip()},
            ],
        }
    ]

    text = state.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = state.processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(state.device)

    # 3. 推理
    multi_layer_hidden_states = forward_with_hidden_states(state.lora_model, state.processor, inputs)

    # 4. 回归预测
    with torch.no_grad():
        if hasattr(state.regression_head, 'forward'):
            predicted_params = state.regression_head(multi_layer_hidden_states)
        else:
            predicted_params = state.regression_head(multi_layer_hidden_states[-1])

        # 反归一化
        if state.param_mean is not None:
            predicted_params = predicted_params.squeeze(0) * state.param_std + state.param_mean
        else:
            predicted_params = predicted_params.squeeze(0)

        predicted_params = predicted_params.cpu().tolist()

    # 5. 构建返回结果
    param_config = ALGORITHM_PARAMS.get(request.algorithm, ALGORITHM_PARAMS["DWA"])
    param_order = list(param_config.keys())
    params_dict = {k: v for k, v in zip(param_order, predicted_params)}

    inference_time = time.time() - start_time

    return InferenceResponse(
        parameters=params_dict,
        parameters_array=predicted_params,
        checkpoint=state.current_checkpoint,
        inference_time=inference_time,
        success=True
    )

# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="初始 checkpoint 路径")
    parser.add_argument("--algorithm", type=str, default="DDP")
    parser.add_argument("--head_type", type=str, default="dpt")
    parser.add_argument("--num_params", type=int, default=6)
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--load_in_4bit", action="store_true", default=True)
    parser.add_argument("--port", type=int, default=5000)

    args = parser.parse_args()

    # 保存配置到 app.state
    app.state.config = args

    # 启动服务
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")

if __name__ == "__main__":
    main()
