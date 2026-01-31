#!/usr/bin/env python3
"""
Transformer IL Navigation Parameter Prediction Service
基于FastAPI的HTTP服务，接收导航场景图像，返回规划器参数
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field
import uvicorn
import io
import base64

# 添加 Imitation_Learning 路径
# script/IL/il_server.py -> script/ -> ros_jackal/ -> Imitation_Learning/
IL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                       "Imitation_Learning")
sys.path.insert(0, IL_PATH)

from model import NavigationTransformerIL
from planner_configs import get_param_names, get_num_params
from torchvision import transforms

# ============================================================
# 算法参数配置
# ============================================================

ALGORITHM_PARAMS = {
    "DWA": {
        "num_params": 9,
        "param_names": ["max_vel_x", "max_vel_theta", "vx_samples", "vtheta_samples",
                       "path_distance_bias", "goal_distance_bias", "inflation_radius",
                       "next_linear_vel", "next_angular_vel"]
    },
    "TEB": {
        "num_params": 9,
        "param_names": ["max_vel_x", "max_vel_x_backwards", "max_vel_theta", "dt_ref",
                       "min_obstacle_dist", "inflation_dist", "inflation_radius",
                       "next_linear_vel", "next_angular_vel"]
    },
    "MPPI": {
        "num_params": 10,
        "param_names": ["max_vel_x", "max_vel_theta", "nr_pairs_", "nr_steps_",
                       "linear_stddev", "angular_stddev", "lambda", "inflation_radius",
                       "next_linear_vel", "next_angular_vel"]
    },
    "DDP": {
        "num_params": 8,
        "param_names": ["max_vel_x", "max_vel_theta", "nr_pairs_", "distance",
                       "robot_radius", "inflation_radius", "next_linear_vel", "next_angular_vel"]
    }
}

# ============================================================
# Pydantic 数据模型
# ============================================================

class InferenceRequest(BaseModel):
    """推理请求"""
    image_base64: Optional[str] = Field(None, description="Base64编码的当前帧图像")
    history_images_base64: Optional[List[str]] = Field(None, description="Base64编码的历史帧图像列表")
    linear_vel: float = Field(default=0.0, description="当前线速度")
    angular_vel: float = Field(default=0.0, description="当前角速度")
    algorithm: str = Field(default="DWA", description="规划算法 (DWA/TEB/MPPI/DDP)")


class InferenceResponse(BaseModel):
    """推理响应"""
    parameters: Dict[str, Any] = Field(description="预测的规划器参数")
    parameters_array: List[float] = Field(description="参数数组形式")
    inference_time: float = Field(description="推理耗时 (秒)")
    success: bool = Field(description="是否成功")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    model_loaded: bool
    device: str
    algorithm: str
    policy_name: str
    num_history_frames: int


class ConfigResponse(BaseModel):
    """配置信息响应"""
    algorithm: str
    num_params: int
    param_names: List[str]
    num_history_frames: int


# ============================================================
# FastAPI 应用
# ============================================================

app = FastAPI(
    title="Transformer IL Navigation Service",
    description="基于Transformer的机器人导航参数预测服务",
    version="1.0.0"
)

# 全局变量
model = None
config = None
device = None
image_transform = None
param_mean = None
param_std = None


def load_param_stats(checkpoint_dir: str):
    """加载参数归一化统计"""
    global param_mean, param_std

    # 尝试不同的文件名
    possible_names = ['param_stats.json.npz', 'param_stats.npz', 'param_stats.json']

    for name in possible_names:
        stats_path = os.path.join(checkpoint_dir, name)
        if os.path.exists(stats_path):
            if name.endswith('.npz'):
                data = np.load(stats_path)
                param_mean = data['mean']
                param_std = data['std']
                print(f"Loaded param stats from {stats_path}")
                print(f"  Mean: {param_mean}")
                print(f"  Std: {param_std}")
                return True

    print(f"[WARN] No param stats found in {checkpoint_dir}, using no normalization")
    param_mean = None
    param_std = None
    return False


@app.on_event("startup")
async def load_model():
    """启动时加载模型"""
    global model, device, image_transform

    print("=" * 60)
    print("Loading Transformer IL Model...")
    print("=" * 60)

    start_time = time.time()

    # 设置设备
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 获取参数数量
    num_params = get_num_params(config.algorithm.lower())
    param_names = get_param_names(config.algorithm.lower())
    print(f"Algorithm: {config.algorithm}, Num params: {num_params}")
    print(f"Param names: {param_names}")

    # 尝试从 checkpoint 目录加载训练配置
    checkpoint_path = config.checkpoint_path
    if os.path.isdir(checkpoint_path):
        checkpoint_dir = checkpoint_path
    else:
        checkpoint_dir = os.path.dirname(checkpoint_path)

    train_config_path = os.path.join(checkpoint_dir, "config.json")
    train_config = {}
    if os.path.exists(train_config_path):
        import json
        with open(train_config_path, 'r') as f:
            train_config = json.load(f)
        print(f"Loaded training config from: {train_config_path}")
        print(f"  vision_model: {train_config.get('vision_model')}")
        print(f"  d_model: {train_config.get('d_model')}")
        print(f"  nhead: {train_config.get('nhead')}")
        print(f"  num_transformer_layers: {train_config.get('num_transformer_layers')}")

    # 创建模型 (使用训练时的配置)
    model = NavigationTransformerIL(
        num_params=num_params,
        num_history_frames=train_config.get('num_history_frames', config.num_history_frames),
        vision_model=train_config.get('vision_model', config.vision_model),
        vision_pretrained=False,  # 加载checkpoint时不需要预训练
        d_model=train_config.get('d_model', None),
        nhead=train_config.get('nhead', None),
        num_transformer_layers=train_config.get('num_transformer_layers', config.num_transformer_layers),
        use_velocity=train_config.get('use_velocity', True)
    )

    # 加载checkpoint
    checkpoint_path = config.checkpoint_path
    if os.path.isdir(checkpoint_path):
        # 如果是目录，查找 best_model.pth 或 latest.pth
        for name in ['best_model.pth', 'latest.pth']:
            p = os.path.join(checkpoint_path, name)
            if os.path.exists(p):
                checkpoint_path = p
                break

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # 加载参数统计（用于反归一化）
    checkpoint_dir = os.path.dirname(checkpoint_path)
    load_param_stats(checkpoint_dir)

    # 图像预处理
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f}s")
    print(f"Model parameters: {model.get_num_params() / 1e6:.2f}M")
    print("=" * 60)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    return HealthResponse(
        status="ok" if model is not None else "loading",
        model_loaded=model is not None,
        device=str(device) if device else "unknown",
        algorithm=config.algorithm if config else "unknown",
        policy_name=f"{config.algorithm.lower()}_il" if config else "unknown",
        num_history_frames=config.num_history_frames if config else 0
    )


@app.get("/config", response_model=ConfigResponse)
async def get_config():
    """获取配置信息"""
    alg = config.algorithm.upper()
    alg_config = ALGORITHM_PARAMS.get(alg, ALGORITHM_PARAMS["DWA"])
    return ConfigResponse(
        algorithm=alg,
        num_params=alg_config["num_params"],
        param_names=alg_config["param_names"],
        num_history_frames=config.num_history_frames
    )


@app.post("/infer", response_model=InferenceResponse)
def infer_parameters(request: InferenceRequest):
    """主推理接口"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    start_time = time.time()

    try:
        # 1. 解码当前帧图像
        if request.image_base64:
            image_data = base64.b64decode(request.image_base64)
            current_image = Image.open(io.BytesIO(image_data)).convert("RGB")
        else:
            raise HTTPException(status_code=400, detail="Must provide image_base64")

        # 2. 处理当前帧
        current_tensor = image_transform(current_image).unsqueeze(0).to(device)  # [1, 3, 224, 224]

        # 3. 处理历史帧
        history_tensors = []
        if request.history_images_base64 and len(request.history_images_base64) > 0:
            for hist_base64 in request.history_images_base64[:config.num_history_frames]:
                hist_data = base64.b64decode(hist_base64)
                hist_image = Image.open(io.BytesIO(hist_data)).convert("RGB")
                hist_tensor = image_transform(hist_image)
                history_tensors.append(hist_tensor)

        # 补齐历史帧（如果不足）
        while len(history_tensors) < config.num_history_frames:
            # 用当前帧填充
            history_tensors.append(image_transform(current_image))

        # [1, num_history, 3, 224, 224]
        history_tensor = torch.stack(history_tensors, dim=0).unsqueeze(0).to(device)

        # 4. 准备速度状态
        velocity = torch.tensor([[request.linear_vel, request.angular_vel]],
                               dtype=torch.float32).to(device)

        # 5. 模型推理
        with torch.no_grad():
            predictions = model(current_tensor, history_tensor, velocity)  # [1, num_params]

        # 6. 反归一化（如果有统计信息）
        pred_np = predictions.cpu().numpy()[0]
        if param_mean is not None and param_std is not None:
            pred_np = pred_np * (param_std + 1e-8) + param_mean

        # 7. 构建结果
        alg = request.algorithm.upper()
        param_names = ALGORITHM_PARAMS.get(alg, ALGORITHM_PARAMS["DWA"])["param_names"]

        # 确保参数数量匹配
        pred_list = pred_np.tolist()
        if len(pred_list) < len(param_names):
            pred_list.extend([0.0] * (len(param_names) - len(pred_list)))
        elif len(pred_list) > len(param_names):
            pred_list = pred_list[:len(param_names)]

        params_dict = {name: val for name, val in zip(param_names, pred_list)}

        inference_time = time.time() - start_time

        print(f"[INFER] Algorithm: {alg}, Time: {inference_time:.3f}s")
        print(f"[INFER] Predicted: {pred_list}")

        return InferenceResponse(
            parameters=params_dict,
            parameters_array=pred_list,
            inference_time=inference_time,
            success=True
        )

    except Exception as e:
        print(f"[ERROR] Inference failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.get("/algorithms")
async def list_algorithms():
    """返回支持的算法列表"""
    return {
        "algorithms": list(ALGORITHM_PARAMS.keys()),
        "details": ALGORITHM_PARAMS
    }


# ============================================================
# 命令行参数和启动
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Transformer IL Navigation Service")

    # 模型配置
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to model checkpoint (file or directory)")
    parser.add_argument("--vision_model", type=str, default="vit_large_patch16_224",
                       help="Vision model name")
    parser.add_argument("--num_transformer_layers", type=int, default=2,
                       help="Number of temporal transformer layers")
    parser.add_argument("--num_history_frames", type=int, default=2,
                       help="Number of history frames")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda/cpu)")

    # 算法配置
    parser.add_argument("--algorithm", type=str, default="DWA",
                       choices=["DWA", "TEB", "MPPI", "DDP"],
                       help="Planning algorithm")

    # 服务器配置
    parser.add_argument("--host", type=str, default="0.0.0.0",
                       help="Server host")
    parser.add_argument("--port", type=int, default=6000,
                       help="Server port")

    return parser.parse_args()


if __name__ == "__main__":
    config = parse_args()

    print(f"""
╔═══════════════════════════════════════════════════════════╗
║   Transformer IL Navigation Parameter Prediction Service  ║
╚═══════════════════════════════════════════════════════════╝
    Checkpoint: {config.checkpoint_path}
    Vision:     {config.vision_model}
    Algorithm:  {config.algorithm}
    History:    {config.num_history_frames} frames
    Device:     {config.device}
    Host:       {config.host}:{config.port}
    """)

    uvicorn.run(app, host=config.host, port=config.port)
