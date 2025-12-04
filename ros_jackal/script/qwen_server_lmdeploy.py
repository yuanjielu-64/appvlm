#!/usr/bin/env python3
"""
Qwen2.5-VL Navigation Parameter Prediction Service (LMDeploy backend)
新增实现：用 LMDeploy pipeline 加速多模态推理；接口与原服务保持一致
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
import uvicorn

# LMDeploy
try:
    from lmdeploy import pipeline as lm_pipeline
    from lmdeploy import GenerationConfig
    try:
        # turbomind 配置（若可用）
        from lmdeploy import TurbomindEngineConfig
    except Exception:
        TurbomindEngineConfig = None
except Exception as e:
    lm_pipeline = None
    GenerationConfig = None
    TurbomindEngineConfig = None

# 延迟导入 transformers（仅在 LMDeploy 不可用时作为最终兜底）
AutoProcessor = None
QwenHF = None


# ============================================================
# 算法参数配置（与 transformers 版本保持一致）
# ============================================================

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


# ============================================================
# Pydantic 数据模型
# ============================================================

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
    run_backend: Optional[str] = None  # 实际运行后端：'lmdeploy' 或 'hf'
    algorithm: str
    port: Optional[int] = None


# ============================================================
# 工具函数
# ============================================================

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


# ============================================================
# FastAPI 应用（LMDeploy 后端）
# ============================================================

app = FastAPI(
    title="Qwen2.5-VL Navigation Service (LMDeploy)",
    description="基于 LMDeploy 的机器人导航参数预测服务",
    version="1.0.0"
)

pipe = None
config = None
run_backend = None  # 'lmdeploy' | 'hf'
hf_model = None
hf_processor = None


@app.on_event("startup")
async def load_model():
    global pipe, config

    if lm_pipeline is None:
        raise RuntimeError("LMDeploy is not installed or failed to import. Please install lmdeploy.")

    print("=" * 60)
    print("Loading Qwen2.5-VL via LMDeploy...")
    print("=" * 60)
    start = time.time()

    # 尝试减少显存碎片
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    engine_cfg = None
    use_backend = (config.backend or 'pytorch').lower()

    def _try_init_pipeline_tm():
        nonlocal engine_cfg
        if TurbomindEngineConfig is None:
            raise RuntimeError("TurbomindEngineConfig not available in this lmdeploy version")
        # 优化显存配置：减小 KV cache，启用 4bit 量化（如果支持）
        engine_cfg = TurbomindEngineConfig(
            session_len=config.session_len,
            tp=config.tp,
            cache_max_entry_count=0.4,  # 减少 KV cache（默认0.8）
            quant_policy=0,              # 0=fp16（更稳定），4=w4a16（更省显存但可能不支持VL）
            max_batch_size=1,            # 单batch推理
        )
        last_err = None
        # 尝试 1: 同时传 backend 与 backend_config
        try:
            return lm_pipeline(config.base_model, backend='turbomind', backend_config=engine_cfg)
        except TypeError as e:
            # 不同版本可能报 multiple values, 改用仅传 backend_config
            last_err = e
        except KeyError as e:
            last_err = e
        # 尝试 2: 仅传 backend_config
        try:
            return lm_pipeline(config.base_model, backend_config=engine_cfg)
        except KeyError as e:
            last_err = e
        # 尝试 3: 仅传 backend
        try:
            return lm_pipeline(config.base_model, backend='turbomind')
        except Exception as e:
            last_err = e
            raise last_err

    def _try_init_pipeline_pt():
        last_err = None
        # 尝试 1: 仅传 backend
        try:
            return lm_pipeline(config.base_model, backend='pytorch')
        except KeyError as e:
            last_err = e
        # 尝试 2: 不传 backend
        try:
            return lm_pipeline(config.base_model)
        except Exception as e:
            last_err = e
            raise last_err

    try:
        if use_backend == 'turbomind':
            pipe = _try_init_pipeline_tm()
            run_backend = 'lmdeploy'
        else:
            pipe = _try_init_pipeline_pt()
            run_backend = 'lmdeploy'
    except Exception as e:
        print(f"[LOAD] Primary backend ({use_backend}) init failed: {e}")
        if getattr(config, 'autofallback', True):
            # 最终兜底：尝试直接使用 transformers 加载（与原服务一致的最小实现）
            print("[LOAD] Fallback to HuggingFace transformers backend...")
            try:
                global AutoProcessor, QwenHF, hf_model, hf_processor
                from transformers import AutoProcessor as _AutoProcessor
                from transformers import Qwen2_5_VLForConditionalGeneration as _QwenHF
                AutoProcessor = _AutoProcessor
                QwenHF = _QwenHF

                import torch
                if torch.cuda.is_available():
                    try:
                        major_cc, _ = torch.cuda.get_device_capability(0)
                        dtype = torch.bfloat16 if major_cc >= 8 else torch.float16
                    except Exception:
                        dtype = torch.float16
                    device_map = 'auto'
                else:
                    dtype = torch.float32
                    device_map = 'cpu'

                hf_model = QwenHF.from_pretrained(config.base_model, torch_dtype=dtype, device_map=device_map)
                hf_model.eval()
                hf_processor = AutoProcessor.from_pretrained(config.base_model)
                run_backend = 'hf'
            except Exception as e2:
                print(f"[LOAD][HF] Failed to initialize transformers backend: {e2}")
                raise
        else:
            raise

    if run_backend == 'lmdeploy':
        print(f"✓ LMDeploy pipeline ready (backend={config.backend}) in {time.time() - start:.2f}s")
    elif run_backend == 'hf':
        print(f"✓ HF transformers backend ready in {time.time() - start:.2f}s")
    print(f"✓ Algorithm: {config.algorithm}")
    print("=" * 60)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=(pipe is not None) or (run_backend == 'hf' and hf_model is not None),
        backend=str(getattr(config, 'backend', 'unknown')),
        run_backend=run_backend,
        algorithm=config.algorithm,
        port=getattr(config, 'port', None),
    )


@app.post("/infer", response_model=InferenceResponse)
def infer_parameters(request: InferenceRequest):
    if pipe is None and not (run_backend == 'hf' and hf_model is not None):
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    print(f"\n[INFER] Received request - Algorithm: {request.algorithm}, Linear: {request.linear_vel}, Angular: {request.angular_vel}")

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
    print(f"[INFER] ⏱️  Image loading: {t1*1000:.1f}ms (size: {image.size})")

    # 2) 算法配置
    t2s = time.time()
    algorithm = request.algorithm.upper()
    if algorithm not in ALGORITHM_PARAMS:
        raise HTTPException(status_code=400, detail=f"Unsupported algorithm: {algorithm}")
    param_cfg = ALGORITHM_PARAMS[algorithm]
    param_order = list(param_cfg.keys())
    order_str, ranges_str = generate_output_format(param_cfg)
    t2 = time.time() - t2s
    print(f"[INFER] ⏱️  Config setup: {t2*1000:.1f}ms")

    # 3) 构建 prompt
    t3s = time.time()
    prompt = PROMPT_TEMPLATE.format(
        number=len(param_order),
        algorithm=algorithm,
        linear_vel=round(request.linear_vel, 4),
        angular_vel=round(request.angular_vel, 4),
        param_order=order_str,
        param_ranges=ranges_str,
    )
    t3 = time.time() - t3s
    print(f"[INFER] ⏱️  Prompt building: {t3*1000:.1f}ms (length: {len(prompt)} chars)")

    # 4) 推理（LMDeploy pipeline 支持多模态：传入 [image, text]）
    try:
        t4s = time.time()
        if run_backend == 'lmdeploy':
            gen_cfg = GenerationConfig(
                max_new_tokens=config.max_new_tokens,
                temperature=0.0,
                top_p=1.0,
                do_sample=False,
            )
            outputs = pipe([image, prompt], gen_config=gen_cfg)
            if isinstance(outputs, (list, tuple)) and len(outputs) > 0:
                text = getattr(outputs[0], 'text', None) or str(outputs[0])
            else:
                text = getattr(outputs, 'text', None) or str(outputs)
            t4 = time.time() - t4s
            print(f"[INFER] ⏱️  model.generate(): {t4:.3f}s")
            raw_output = text
        else:
            # HF transformers 推理路径
            import torch
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text_prompt = hf_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = hf_processor(images=[image], text=text_prompt, return_tensors="pt").to(hf_model.device)
            with torch.inference_mode():
                generated_ids = hf_model.generate(
                    **inputs,
                    max_new_tokens=config.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    num_beams=1,
                    pad_token_id=hf_processor.tokenizer.pad_token_id,
                    eos_token_id=hf_processor.tokenizer.eos_token_id,
                )
            input_len = inputs["input_ids"].shape[1]
            new_ids = generated_ids[0, input_len:]
            raw_output = hf_processor.batch_decode(new_ids.unsqueeze(0), skip_special_tokens=True)[0]
            t4 = time.time() - t4s
            print(f"[INFER] ⏱️  model.generate() [HF]: {t4:.3f}s")
    except Exception as e:
        print(f"[ERROR] Model inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")

    # 5) 解析
    t5s = time.time()
    fallback = [param_cfg[k]["range"][0] for k in param_order]
    params_dict, params_array = parse_qwen_output(raw_output, param_order, fallback)
    t5 = time.time() - t5s

    total = time.time() - t0
    print("\n" + "="*60)
    print("📊 INFERENCE PERFORMANCE SUMMARY (LMDeploy)")
    print("="*60)
    print(f"⏱️  Total time:        {total:.3f}s")
    print(f"   ├─ Image load:      {t1*1000:.1f}ms")
    print(f"   ├─ Config setup:    {t2*1000:.1f}ms")
    print(f"   ├─ Prompt build:    {t3*1000:.1f}ms")
    print(f"   ├─ Model generate:  {t4:.3f}s ⚡")
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


@app.post("/infer_file")
async def infer_from_file(
    file: UploadFile = File(...),
    linear_vel: float = Form(0.0),
    angular_vel: float = Form(0.0),
    algorithm: str = Form("DWA"),
):
    data = await file.read()
    b64 = base64.b64encode(data).decode('utf-8')
    req = InferenceRequest(
        image_base64=b64,
        linear_vel=linear_vel,
        angular_vel=angular_vel,
        algorithm=algorithm,
    )
    return infer_parameters(req)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--algorithm", type=str, default="DWA", choices=list(ALGORITHM_PARAMS.keys()))
    p.add_argument("--max_new_tokens", type=int, default=80)

    # LMDeploy 选项
    p.add_argument("--backend", type=str, default="turbomind", choices=["turbomind", "pytorch"]) 
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--session_len", type=int, default=4096)
    p.add_argument("--autofallback", action="store_true", default=True,
                   help="Auto fallback to PyTorch backend when TurboMind init fails")

    # Server
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=5001)
    return p.parse_args()


if __name__ == "__main__":
    config = parse_args()
    print(f"""
╔═══════════════════════════════════════════════════════════╗
║   Qwen2.5-VL Navigation Service (LMDeploy backend)        ║
╚═══════════════════════════════════════════════════════════╝
    Base Model: {config.base_model}
    Backend:    {config.backend} (tp={config.tp}, session_len={config.session_len})
    Algorithm:  {config.algorithm}
    Host:       {config.host}:{config.port}
    MaxTokens:  {config.max_new_tokens}
    """)
    uvicorn.run(app, host=config.host, port=config.port)
