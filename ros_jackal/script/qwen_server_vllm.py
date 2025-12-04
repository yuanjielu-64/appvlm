#!/usr/bin/env python3
"""
Qwen2.5-VL Navigation Parameter Prediction Service (vLLM backend)
用于加载 finetune 后的模型进行高速推理
"""

import argparse
import base64
import io
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from PIL import Image

# vLLM imports
try:
    from vllm import LLM, SamplingParams
    from vllm.multimodal.utils import fetch_image
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None

# 算法参数配置（与其他版本保持一致）
ALGORITHM_PARAMS = {
    "DWA": {
        "max_vel_x": {"range": [0.2, 2.0], "type": "float"},
        "max_vel_theta": {"range": [0.314, 3.14], "type": "float"},
        "vx_samples": {"range": [4, 12], "type": "int"},
        "vtheta_samples": {"range": [8, 40], "type": "int"},
        "path_distance_bias": {"range": [0.1, 1.5], "type": "float"},
        "goal_distance_bias": {"range": [0.1, 2.0], "type": "float"},
        "inflation_radius": {"range": [0.1, 0.6], "type": "float"}
    },
    "TEB": {
        "max_vel_x": {"range": [0.2, 2.0], "type": "float"},
        "max_vel_x_backwards": {"range": [0.1, 0.7], "type": "float"},
        "max_vel_theta": {"range": [0.314, 3.14], "type": "float"},
        "dt_ref": {"range": [0.1, 0.35], "type": "float"},
        "min_obstacle_dist": {"range": [0.05, 0.2], "type": "float"},
        "inflation_dist": {"range": [0.01, 0.2], "type": "float"},
        "inflation_radius": {"range": [0.1, 0.6], "type": "float"}
    },
    "MPPI": {
        "max_vel_x": {"range": [-0.5, 2.0], "type": "float"},
        "max_vel_theta": {"range": [0.314, 3.14], "type": "float"},
        "nr_pairs": {"range": [400, 800], "type": "int"},
        "nr_steps": {"range": [20, 40], "type": "int"},
        "linear_stddev": {"range": [0.05, 0.15], "type": "float"},
        "angular_stddev": {"range": [0.02, 0.15], "type": "float"},
        "lambda": {"range": [0.5, 5.0], "type": "float"},
        "inflation_radius": {"range": [0.1, 0.6], "type": "float"}
    },
    "DDP": {
        "max_vel_x": {"range": [0.0, 2.0], "type": "float"},
        "max_vel_theta": {"range": [0.314, 3.14], "type": "float"},
        "nr_pairs": {"range": [400, 800], "type": "int"},
        "distance": {"range": [0.01, 0.2], "type": "float"},
        "robot_radius": {"range": [0.01, 0.05], "type": "float"},
        "inflation_radius": {"range": [0.1, 0.6], "type": "float"}
    }
}

PROMPT_TEMPLATE = (
    "Jackal Robot (0.508m×0.430m). Current velocity: linear={linear_vel}m/s, angular={angular_vel}rad/s. "
    "Predict {number} {algorithm} parameters for the navigation scene.\n"
    "SCENE: Red=obstacles (laser scan), Purple=global path, Yellow=robot, Grid=1m spacing.\n"
    "STRATEGY: Dense obstacles/narrow→reduce max_vel_x; Open space→increase max_vel_x; Sharp turns→increase max_vel_theta;\n"
    "OUTPUT: JSON array format [val1, val2, ...] with {number} values in order:\n{param_order}\n"
    "Ranges: {param_ranges}"
)


# Pydantic 数据模型
class InferenceRequest(BaseModel):
    image_base64: Optional[str] = Field(None)
    image_path: Optional[str] = Field(None)
    linear_vel: float = Field(default=0.0)
    angular_vel: float = Field(default=0.0)
    algorithm: str = Field(default="DWA")


class InferenceResponse(BaseModel):
    parameters: Dict[str, Any]
    parameters_array: List[float]
    raw_output: str
    inference_time: float
    success: bool


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    backend: str
    algorithm: str
    port: Optional[int] = None


# 工具函数
def generate_output_format(param_config: Dict) -> tuple:
    order, ranges = [], []
    for k, v in param_config.items():
        order.append(k)
        vmin, vmax = v["range"]
        ranges.append(f"{k}: [{vmin}, {vmax}]")
    return "\n".join(f"- {k}" for k in order), ", ".join(ranges)


def parse_qwen_output(result: str, param_order: List[str], fallback_params: List[float]) -> tuple:
    try:
        cleaned = result.strip()
        if cleaned.startswith('```'):
            m = re.search(r'```(?:json)?\s*([\[\{].*?[\]\}])\s*```', cleaned, re.DOTALL)
            cleaned = m.group(1) if m else cleaned.replace('```json', '').replace('```', '').strip()

        if not cleaned.lstrip().startswith(('[', '{')):
            i1, i2 = cleaned.find('['), cleaned.find('{')
            i = min(x for x in [i1, i2] if x != -1) if (i1 != -1 or i2 != -1) else -1
            if i != -1:
                open_ch = cleaned[i]
                close_ch = ']' if open_ch == '[' else '}'
                depth, end = 0, -1
                for j in range(i, len(cleaned)):
                    if cleaned[j] == open_ch:
                        depth += 1
                    elif cleaned[j] == close_ch:
                        depth -= 1
                        if depth == 0:
                            end = j + 1
                            break
                if end != -1:
                    cleaned = cleaned[i:end]

        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            arr = parsed[:len(param_order)]
            if len(arr) < len(param_order):
                arr.extend(fallback_params[len(arr):])
            return {k: v for k, v in zip(param_order, arr)}, arr
        elif isinstance(parsed, dict):
            if "inflation_radius" not in parsed and "final_inflation" in parsed:
                parsed["inflation_radius"] = parsed.pop("final_inflation")
            arr = [parsed.get(k, fallback_params[i]) for i, k in enumerate(param_order)]
            return parsed, arr
        else:
            raise ValueError(f"Unexpected JSON type: {type(parsed)}")
    except Exception as e:
        print(f"[ERROR] Parse failed: {e}, using fallback params")
        fb = {k: v for k, v in zip(param_order, fallback_params)}
        return fb, fallback_params


# FastAPI 应用
app = FastAPI(
    title="Qwen2.5-VL Navigation Service (vLLM)",
    description="基于 vLLM 的高速推理服务（支持 finetune 模型）",
    version="1.0.0"
)

llm = None
config = None


@app.on_event("startup")
async def load_model():
    global llm, config

    if not VLLM_AVAILABLE:
        raise RuntimeError(
            "vLLM is not installed. Install it with:\n"
            "pip install vllm"
        )

    print("=" * 60)
    print("Loading Qwen2.5-VL via vLLM...")
    print("=" * 60)
    start = time.time()

    # vLLM 初始化参数
    # 参考：https://docs.vllm.ai/en/latest/models/vlm.html
    llm = LLM(
        model=config.base_model,
        trust_remote_code=True,
        max_model_len=config.max_model_len,
        gpu_memory_utilization=config.gpu_memory_util,
        tensor_parallel_size=config.tp,
        dtype="auto",  # 自动选择 bfloat16/float16
        # 针对 VLM 的优化
        limit_mm_per_prompt={"image": 1},  # 每个 prompt 最多 1 张图
    )

    print(f"✓ vLLM model loaded in {time.time() - start:.2f}s")
    print(f"✓ Algorithm: {config.algorithm}")
    print("=" * 60)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=(llm is not None),
        backend="vllm",
        algorithm=config.algorithm,
        port=getattr(config, 'port', None),
    )


@app.post("/infer", response_model=InferenceResponse)
def infer_parameters(request: InferenceRequest):
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    print(f"\n[INFER] Algorithm: {request.algorithm}, Linear: {request.linear_vel}, Angular: {request.angular_vel}")

    t0 = time.time()

    # 1) 加载图像
    t1s = time.time()
    try:
        if request.image_base64:
            img_data = base64.b64decode(request.image_base64)
            image = Image.open(io.BytesIO(img_data)).convert('RGB')
        elif request.image_path:
            abs_path = os.path.abspath(request.image_path)
            if not os.path.exists(abs_path):
                raise FileNotFoundError(abs_path)
            image = Image.open(abs_path).convert('RGB')
        else:
            raise HTTPException(status_code=400, detail="Must provide either image_base64 or image_path")
    except Exception as e:
        print(f"[ERROR] Failed to load image: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to load image: {str(e)}")
    t1 = time.time() - t1s
    print(f"[INFER] ⏱️  Image loading: {t1*1000:.1f}ms")

    # 2) 算法配置
    t2s = time.time()
    algorithm = request.algorithm.upper()
    if algorithm not in ALGORITHM_PARAMS:
        raise HTTPException(status_code=400, detail=f"Unsupported algorithm: {algorithm}")
    param_cfg = ALGORITHM_PARAMS[algorithm]
    param_order = list(param_cfg.keys())
    order_str, ranges_str = generate_output_format(param_cfg)
    t2 = time.time() - t2s

    # 3) 构建 prompt
    t3s = time.time()
    text_prompt = PROMPT_TEMPLATE.format(
        number=len(param_order),
        algorithm=algorithm,
        linear_vel=round(request.linear_vel, 4),
        angular_vel=round(request.angular_vel, 4),
        param_order=order_str,
        param_ranges=ranges_str,
    )
    t3 = time.time() - t3s

    # 4) vLLM 推理
    try:
        t4s = time.time()

        # vLLM 多模态输入格式
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]

        sampling_params = SamplingParams(
            max_tokens=config.max_new_tokens,
            temperature=0.0,
            top_p=1.0,
        )

        outputs = llm.generate(messages, sampling_params=sampling_params)
        raw_output = outputs[0].outputs[0].text

        t4 = time.time() - t4s
        print(f"[INFER] ⏱️  vLLM generate: {t4:.3f}s")
    except Exception as e:
        print(f"[ERROR] vLLM inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"vLLM inference failed: {str(e)}")

    # 5) 解析
    t5s = time.time()
    fallback = [param_cfg[k]["range"][0] for k in param_order]
    params_dict, params_array = parse_qwen_output(raw_output, param_order, fallback)
    t5 = time.time() - t5s

    total = time.time() - t0
    print("\n" + "="*60)
    print("📊 INFERENCE PERFORMANCE SUMMARY (vLLM)")
    print("="*60)
    print(f"⏱️  Total time:        {total:.3f}s")
    print(f"   ├─ Image load:      {t1*1000:.1f}ms")
    print(f"   ├─ Config setup:    {t2*1000:.1f}ms")
    print(f"   ├─ Prompt build:    {t3*1000:.1f}ms")
    print(f"   ├─ vLLM generate:   {t4:.3f}s ⚡")
    print(f"   └─ Parse output:    {t5*1000:.1f}ms")
    print(f"✅ Parameters:       {params_array}")
    print("="*60 + "\n")

    return InferenceResponse(
        parameters=params_dict,
        parameters_array=params_array,
        raw_output=raw_output,
        inference_time=total,
        success=True,
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct",
                   help="HF model path (支持 finetune 后的本地路径)")
    p.add_argument("--algorithm", type=str, default="DWA", choices=list(ALGORITHM_PARAMS.keys()))
    p.add_argument("--max_new_tokens", type=int, default=80)

    # vLLM 选项
    p.add_argument("--max_model_len", type=int, default=2048,
                   help="最大上下文长度（导航 prompt 较短，2048 足够）")
    p.add_argument("--gpu_memory_util", type=float, default=0.85,
                   help="GPU 显存利用率（0-1，默认 0.85）")
    p.add_argument("--tp", type=int, default=1,
                   help="Tensor parallel size（单卡保持 1）")

    # Server
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=5003)
    return p.parse_args()


if __name__ == "__main__":
    import uvicorn
    config = parse_args()
    print(f"""
╔═══════════════════════════════════════════════════════════╗
║   Qwen2.5-VL Navigation Service (vLLM backend)            ║
╚═══════════════════════════════════════════════════════════╝
    Base Model:      {config.base_model}
    Backend:         vLLM (PagedAttention + Continuous Batching)
    Algorithm:       {config.algorithm}
    Host:            {config.host}:{config.port}
    Max Model Len:   {config.max_model_len}
    GPU Util:        {config.gpu_memory_util}
    Max New Tokens:  {config.max_new_tokens}

💡 使用说明：
   1. Finetune 时：使用 transformers 训练
   2. 推理时：用此服务加载 finetune 后的模型路径
   3. 性能提升：2-5x faster than HF transformers
    """)
    uvicorn.run(app, host=config.host, port=config.port)
