#!/home/yuanjielu/miniforge3/envs/lmms-finetune-qwen/bin/python
"""
Qwen2.5-VL Navigation Parameter Prediction Service
基于FastAPI的HTTP服务，接收导航场景图像，返回规划器参数
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from peft import PeftModel
from PIL import Image
from pydantic import BaseModel, Field
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from transformers import StoppingCriteria, StoppingCriteriaList
import uvicorn
import io
import base64
import asyncio

# ============================================================
# 算法参数配置 (与 chatgpt.py 保持一致)
# ============================================================

ALGORITHM_PARAMS = {
    "DWA": {
        "max_vel_x": {"range": [0.2, 2.0], "type": "float", "description": "Forward velocity (m/s)"},
        "max_vel_theta": {"range": [0.314, 3.14], "type": "float", "description": "Angular velocity (rad/s)"},
        "vx_samples": {"range": [4, 12], "type": "int", "description": "Number of linear velocity samples"},
        "vtheta_samples": {"range": [8, 40], "type": "int", "description": "Number of angular velocity samples"},
        "path_distance_bias": {"range": [0.1, 1.5], "type": "float", "description": "Path following weight"},
        "goal_distance_bias": {"range": [0.1, 2.0], "type": "float", "description": "Goal seeking weight"},
        "inflation_radius": {"range": [0.1, 0.6], "type": "float", "description": "Inflation radius (m)"}
    },
    "TEB": {
        "max_vel_x": {"range": [0.2, 2.0], "type": "float", "description": "Forward velocity (m/s)"},
        "max_vel_x_backwards": {"range": [0.1, 0.7], "type": "float", "description": "Backward velocity (m/s)"},
        "max_vel_theta": {"range": [0.314, 3.14], "type": "float", "description": "Angular velocity (rad/s)"},
        "dt_ref": {"range": [0.1, 0.35], "type": "float", "description": "Desired temporal resolution (s)"},
        "min_obstacle_dist": {"range": [0.05, 0.2], "type": "float",
                              "description": "Minimum distance to obstacles (m)"},
        "inflation_dist": {"range": [0.01, 0.2], "type": "float", "description": "Inflation distance (m)"},
        "inflation_radius": {"range": [0.1, 0.6], "type": "float", "description": "Inflation radius (m)"}
    },
    "MPPI": {
        "max_vel_x": {"range": [-0.5, 2.0], "type": "float", "description": "Forward velocity (m/s)"},
        "max_vel_theta": {"range": [0.314, 3.14], "type": "float", "description": "Angular velocity (rad/s)"},
        "nr_pairs": {"range": [400, 800], "type": "int", "description": "Number of rollout pairs"},
        "nr_steps": {"range": [20, 40], "type": "int", "description": "Number of prediction steps"},
        "linear_stddev": {"range": [0.05, 0.15], "type": "float", "description": "Linear velocity standard deviation"},
        "angular_stddev": {"range": [0.02, 0.15], "type": "float",
                           "description": "Angular velocity standard deviation"},
        "lambda": {"range": [0.5, 5.0], "type": "float", "description": "Softmax temperature"},
        "inflation_radius": {"range": [0.1, 0.6], "type": "float", "description": "Inflation radius (m)"}
    },
    "DDP": {
        "max_vel_x": {"range": [0.0, 2.0], "type": "float", "description": "Forward velocity (m/s)"},
        "max_vel_theta": {"range": [0.314, 3.14], "type": "float", "description": "Angular velocity (rad/s)"},
        "nr_pairs": {"range": [400, 800], "type": "int", "description": "Number of rollout pairs"},
        "distance": {"range": [0.01, 0.2], "type": "float", "description": "Distance threshold (m)"},
        "robot_radius": {"range": [0.01, 0.05], "type": "float", "description": "Robot radius (m)"},
        "inflation_radius": {"range": [0.1, 0.6], "type": "float", "description": "Inflation radius (m)"}
    }
}

PROMPT_TEMPLATE = (
    "You are a Clearpath Jackal Robot, the length is 0.508 m, and the width is 0.430 m. "
    "The robot primarily moves along the purple global path. Your task is to predict {number} {algorithm} planner parameters "
    "based on the given navigation scene image. The predicted parameters should help traditional planners "
    "achieve faster, safer robot navigation by improving path-following and obstacle-avoidance. "
    "Your current linear velocity is {linear_vel} (linear_vel), and your angular velocity is {angular_vel} (angular_vel)\n"
    "PARAMETER TUNING STRATEGY:\n"
    "- Dense obstacles or narrow passages → Reduce max_vel_x, increase samples\n"
    "- Wide open space → Increase max_vel_x, reduce obstacle weights\n"
    "- Sharp turns ahead → Increase max_vel_theta, increase path_distance_bias\n"
    "- Close to goal → Increase goal_distance_bias\n"
    "SCENE UNDERSTANDING: "
    "- The green line on the robot represents its current direction of movement (x-axis). "
    "- The blue line on the robot represents the y-axis. "
    "- Grid spacing: 1 meter. "
    "- Red points: Hokuyo laser scan data (obstacles). "
    "- Purple line: Global path to follow. "
    "- Yellow rectangle: Robot's current position and footprint\n"
    "- Task: Navigate safely along the path while avoiding obstacles. "
    "OUTPUT FORMAT: The output must be in strict JSON format with exactly the following fields, compact (no spaces), stop immediately after the closing brace:\n"
    "{output_format}"
)


# ============================================================
# Pydantic 数据模型
# ============================================================

class InferenceRequest(BaseModel):
    """推理请求 - 支持两种输入方式"""
    image_base64: Optional[str] = Field(None, description="Base64编码的图像")
    image_path: Optional[str] = Field(None, description="图像文件路径")
    linear_vel: float = Field(default=0.0, description="当前线速度")
    angular_vel: float = Field(default=0.0, description="当前角速度")
    algorithm: str = Field(default="DWA", description="规划算法 (DWA/TEB/MPPI/DDP)")


class InferenceResponse(BaseModel):
    """推理响应"""
    parameters: Dict[str, Any] = Field(description="预测的规划器参数")
    parameters_array: List[float] = Field(description="参数数组形式")
    raw_output: str = Field(description="模型原始输出")
    inference_time: float = Field(description="推理耗时 (秒)")
    success: bool = Field(description="是否成功")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    model_loaded: bool
    device: str
    algorithm: str


# ============================================================
# 工具函数
# ============================================================

def generate_output_format(param_config: Dict) -> str:
    """生成JSON输出格式说明"""
    lines = ["{"]
    for param_name, param_info in param_config.items():
        param_type = "<int>" if param_info["type"] == "int" else "<float>"
        range_str = f"{param_info['range'][0]}–{param_info['range'][1]}"
        line = f'  "{param_name}": {param_type},  // {param_info["description"]}, range: {range_str}'
        lines.append(line)
    lines[-1] = lines[-1].rstrip(',')
    lines.append("}")
    return "\n".join(lines)


def parse_qwen_output(result: str, param_order: List[str], fallback_params: List[float]) -> tuple:
    """
    解析Qwen输出的JSON，提取参数 (与qwen_7b_lora.py保持一致)
    返回: (参数字典, 参数数组)
    """
    try:
        # 去除 markdown 标记 (与qwen_7b_lora.py相同的处理逻辑) + 兼容非JSON前缀
        cleaned = result.strip()
        if cleaned.startswith('```'):
            match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', cleaned, re.DOTALL)
            if match:
                cleaned = match.group(1)
            else:
                cleaned = cleaned.replace("```json", "").replace("```", "").strip()

        # 如果不是以 { 开头，尝试提取第一段平衡的大括号内容
        if not cleaned.lstrip().startswith('{'):
            brace_start = cleaned.find('{')
            brace_end = -1
            if brace_start != -1:
                depth = 0
                for i, ch in enumerate(cleaned[brace_start:], start=brace_start):
                    if ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            brace_end = i + 1
                            break
            if brace_start != -1 and brace_end != -1:
                cleaned = cleaned[brace_start:brace_end]

        # 解析 JSON
        params_dict = json.loads(cleaned)

        # 兼容别名键名（模型可能输出不同命名）
        if "inflation_radius" not in params_dict and "final_inflation" in params_dict:
            params_dict["inflation_radius"] = params_dict.pop("final_inflation")

        # 按顺序提取值
        # 缺失字段用fallback补齐
        param_array = [params_dict.get(key, fallback_params[i]) for i, key in enumerate(param_order)]

        return params_dict, param_array

    except Exception as e:
        print(f"[ERROR] Parse failed: {e}, using fallback params")
        # 返回fallback参数的字典形式
        fallback_dict = {key: val for key, val in zip(param_order, fallback_params)}
        return fallback_dict, fallback_params


# ============================================================
# FastAPI 应用
# ============================================================

app = FastAPI(
    title="Qwen2.5-VL Navigation Service",
    description="基于Qwen2.5-VL的机器人导航参数预测服务",
    version="1.0.0"
)

# 全局变量
model = None
processor = None
config = None


@app.on_event("startup")
async def load_model():
    """启动时加载模型"""
    global model, processor, config

    print("=" * 60)
    print("Loading Qwen2.5-VL Model...")
    print("=" * 60)

    start_time = time.time()
    # 尝试改善显存碎片问题
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # 加载基础模型 (根据设备选择更稳妥的dtype)
    # GPU: 优先 bfloat16（若支持），否则 float16；CPU: 使用 float32
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

    def _load_base_model(**kwargs):
        return Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.base_model,
            **kwargs,
        )

    # 首选：按用户指定 device/device_map 与量化选项加载
    try:
        # 处理 device_map 覆盖
        resolved_device_map = config.device_map if getattr(config, 'device_map', None) else config.device

        # 可选：预设4/8bit量化
        if getattr(config, 'load_in_4bit', False) or getattr(config, 'load_in_8bit', False):
            use_4bit = getattr(config, 'load_in_4bit', False)
            use_8bit = getattr(config, 'load_in_8bit', False)
            if use_4bit and use_8bit:
                print("[LOAD] Both --load_in_4bit and --load_in_8bit set; preferring 4-bit.")
                use_8bit = False

            quant_kwargs = {"device_map": resolved_device_map}
            if use_4bit:
                qdtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
                quant_kwargs.update({
                    "load_in_4bit": True,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_compute_dtype": qdtype,
                    "bnb_4bit_use_double_quant": True,
                })
            elif use_8bit:
                quant_kwargs.update({"load_in_8bit": True})

            print(
                f"[LOAD] Loading model with quantization (4bit={use_4bit}, 8bit={use_8bit}), device_map={resolved_device_map}")
            model = _load_base_model(**quant_kwargs)
        else:
            # 标准全精度/半精度加载
            model_kwargs = {
                "dtype": dtype,
                "torch_dtype": dtype,
                "device_map": resolved_device_map,
            }
            model = _load_base_model(**model_kwargs)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("[LOAD] CUDA OOM during standard load. Retrying with 4-bit quantization on GPU (device_map=auto)...")
            # 尝试4bit量化 + 自动分配/卸载
            try:
                qdtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
                model = _load_base_model(
                    load_in_4bit=True,
                    device_map="auto",
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=qdtype,
                    bnb_4bit_use_double_quant=True,
                )
            except Exception as e2:
                print(f"[LOAD] 4-bit load failed: {e2}")
                print("[LOAD] Falling back to CPU offload (device_map=auto, max_memory cap)")
                # 最后尝试：自动分配到CPU/GPU，限制GPU占用
                max_gpu_gb = getattr(config, 'max_gpu_memory_gb', None)
                if max_gpu_gb is None:
                    # 默认留出1GiB余量
                    max_gpu_gb = 8
                max_memory = {"cuda:0": f"{int(max_gpu_gb)}GiB", "cpu": "48GiB"}
                model = _load_base_model(
                    torch_dtype=dtype,
                    device_map="auto",
                    max_memory=max_memory,
                )
        else:
            raise

    # 加载LoRA权重
    if config.lora_path:
        print(f"Loading LoRA from: {config.lora_path}")
        model = PeftModel.from_pretrained(model, config.lora_path)

    model.eval()

    # 加载处理器
    processor = AutoProcessor.from_pretrained(
        config.base_model,
        min_pixels=256 * 28 * 28,
        max_pixels=384 * 28 * 28
    )

    load_time = time.time() - start_time
    print(f"✓ Model loaded in {load_time:.2f}s")
    print(f"✓ Device: {model.device}")
    print(f"✓ Algorithm: {config.algorithm}")
    print("=" * 60)

    # 可选：后台暖机（不阻塞服务启动）
    if getattr(config, 'startup_warmup', False):
        asyncio.create_task(_run_startup_warmup(model, processor, config))

async def _run_startup_warmup(model, processor, config):
    print("\n[WARMUP] Running non-blocking warmup inference...")
    test_start = time.time()
    try:
        test_image_path = "/home/yuanjielu/robot_navigation/noetic/appvlm_ws/src/ros_jackal/buffer/dwa_qwen/actor_0/VLM_000000.png"
        if os.path.exists(test_image_path):
            test_image = Image.new('RGB', (384, 384), color=(128, 128, 128))
            print(f"[WARMUP] Loaded test image: {test_image_path}, size: {test_image.size}")

            param_config = ALGORITHM_PARAMS.get(config.algorithm, ALGORITHM_PARAMS["DWA"])
            param_order = list(param_config.keys())
            output_format = generate_output_format(param_config)

            test_prompt = PROMPT_TEMPLATE.format(
                number=len(param_config),
                algorithm=config.algorithm,
                linear_vel=0.0,
                angular_vel=0.0,
                output_format=output_format
            )

            test_messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": test_image},
                    {"type": "text", "text": test_prompt},
                ],
            }]
            test_text = processor.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)
            test_inputs = processor(images=[test_image], text=test_text, return_tensors="pt").to(model.device)

            tokens = max(1, int(getattr(config, 'startup_tokens', 16)))
            print(f"[WARMUP] Starting generation with max_new_tokens={tokens}...")
            with torch.inference_mode():
                gen_ids = model.generate(**test_inputs, max_new_tokens=tokens, do_sample=False)

            # 仅解码新增tokens，避免把prompt/system内容一起解码
            input_len = test_inputs["input_ids"].shape[1]
            new_tokens = gen_ids[0, input_len:]
            test_output = processor.batch_decode(new_tokens.unsqueeze(0), skip_special_tokens=True)[0]

            elapsed = time.time() - test_start
            print(f"[WARMUP] ✓ Warmup successful in {elapsed:.2f}s")
            print(f"[WARMUP] Raw output: {test_output[:200]}...")

            # 尝试解析（容错为主）
            fallback_params = [param_config[k]["range"][0] for k in param_order]
            params_dict, params_array = parse_qwen_output(test_output, param_order, fallback_params)
            print(f"[WARMUP] Parsed parameters: {params_array}")
        else:
            print(f"[WARMUP] Test image not found at {test_image_path}, skipping warmup")
    except Exception as e:
        print(f"[WARMUP] ✗ Warmup failed: {e}")
        import traceback
        traceback.print_exc()
    print("=" * 60)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    return HealthResponse(
        status="ok" if model is not None else "loading",
        model_loaded=model is not None,
        device=str(model.device) if model else "unknown",
        algorithm=config.algorithm
    )


@app.post("/infer", response_model=InferenceResponse)
def infer_parameters(request: InferenceRequest):
    """
    主推理接口：接收图像，返回规划器参数
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    print(
        f"\n[INFER] Received request - Algorithm: {request.algorithm}, Linear: {request.linear_vel}, Angular: {request.angular_vel}")

    start_time = time.time()

    # 1. 加载图像
    t1_start = time.time()
    try:
        if request.image_base64:
            # Base64解码
            image_data = base64.b64decode(request.image_base64)
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
        elif request.image_path:
            # 文件路径 - 转换为绝对路径
            abs_path = os.path.abspath(request.image_path)
            if not os.path.exists(abs_path):
                raise FileNotFoundError(f"Image file not found: {abs_path}")
            image = Image.open(abs_path).convert("RGB")
        else:
            raise HTTPException(status_code=400, detail="Must provide either image_base64 or image_path")
    except Exception as e:
        print(f"[ERROR] Failed to load image: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to load image: {str(e)}")
    t1 = time.time() - t1_start
    print(f"[INFER] ⏱️  Image loading: {t1 * 1000:.1f}ms (size: {image.size})")

    # 2. 获取算法配置
    t2_start = time.time()
    algorithm = request.algorithm.upper()
    if algorithm not in ALGORITHM_PARAMS:
        raise HTTPException(status_code=400, detail=f"Unsupported algorithm: {algorithm}")
    param_config = ALGORITHM_PARAMS[algorithm]
    param_order = list(param_config.keys())
    output_format = generate_output_format(param_config)
    t2 = time.time() - t2_start
    print(f"[INFER] ⏱️  Config setup: {t2 * 1000:.1f}ms")

    # 3. 构建prompt
    t3_start = time.time()
    prompt = PROMPT_TEMPLATE.format(
        number=len(param_config),
        algorithm=algorithm,
        linear_vel=round(request.linear_vel, 4),
        angular_vel=round(request.angular_vel, 4),
        output_format=output_format
    )
    t3 = time.time() - t3_start
    print(f"[INFER] ⏱️  Prompt building: {t3 * 1000:.1f}ms (length: {len(prompt)} chars)")

    # 4. 准备输入（图像编码 + tokenization）
    t4_start = time.time()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    t4_template = time.time()
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    t4_template = time.time() - t4_template

    t4_process = time.time()
    inputs = processor(
        images=[image],
        text=text,
        return_tensors="pt",
    )
    t4_process = time.time() - t4_process

    t4_to_device = time.time()
    inputs = inputs.to(model.device)
    t4_to_device = time.time() - t4_to_device

    t4 = time.time() - t4_start
    print(f"[INFER] ⏱️  Input preparation: {t4 * 1000:.1f}ms")
    print(f"[INFER]    ├─ Template: {t4_template * 1000:.1f}ms")
    print(f"[INFER]    ├─ Processor (image encode): {t4_process * 1000:.1f}ms")
    print(f"[INFER]    └─ To device: {t4_to_device * 1000:.1f}ms")

    # 5. 模型推理 - 详细时间分析
    try:
        # 5.1 准备 stopping criteria
        t_stopping_start = time.time()
        stopping = None
        if getattr(config, 'stop_on_json_complete', True):
            class JsonStopper(StoppingCriteria):
                def __init__(self, tokenizer, input_len, required_groups, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.tok = tokenizer
                    self.input_len = input_len
                    self.required_groups = required_groups
                    self._last_check_len = input_len
                    self._cached_text = ""

                def __call__(self, input_ids, scores, **kwargs):
                    try:
                        current_len = input_ids.shape[1]

                        # ✅ 只decode新增的tokens (增量解码)
                        if current_len > self._last_check_len:
                            new_tokens = input_ids[0, self._last_check_len:]
                            new_text = self.tok.decode(new_tokens, skip_special_tokens=True)
                            self._cached_text += new_text
                            self._last_check_len = current_len

                        text = self._cached_text

                        if '}' not in text:
                            return False

                        for group in self.required_groups:
                            if not any((f'"{syn}"' in text) for syn in group):
                                return False

                        depth = 0
                        started = False
                        for ch in text:
                            if ch == '{':
                                started = True
                                depth += 1
                            elif ch == '}':
                                depth -= 1
                                if started and depth == 0:
                                    return True  # ✅ 找到完整的JSON

                        return False  # 括号未平衡

                    except Exception as e:
                        # 出错时不停止，让模型继续生成
                        print(f"[WARNING] JsonStopper error: {e}")
                        return False

            # 构建同义键分组（例如 inflation_radius | final_inflation）
            synonyms = {"inflation_radius": ["inflation_radius", "final_inflation"]}
            required_groups = [synonyms.get(k, [k]) for k in param_order]

            print(f"[DEBUG] param_order: {param_order}")
            print(f"[DEBUG] required_groups: {required_groups}")

            stopping = StoppingCriteriaList([
                JsonStopper(processor.tokenizer, inputs["input_ids"].shape[1], required_groups)
            ])
        t_stopping = time.time() - t_stopping_start
        print(f"[INFER] ⏱️  Stopping criteria setup: {t_stopping * 1000:.1f}ms")

        # 5.2 模型生成
        generate_start = time.time()
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                do_sample=False,
                stopping_criteria=stopping,
                use_cache=True,  # ✅ 启用KV cache
                num_beams=1,  # ✅ 明确指定
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )

        generate_time = time.time() - generate_start
        print(f"[INFER] ⏱️  model.generate(): {generate_time:.3f}s")

        # 5.3 解码输出
        decode_start = time.time()
        input_len = inputs["input_ids"].shape[1]
        new_ids = generated_ids[0, input_len:]
        raw_output = processor.batch_decode(new_ids.unsqueeze(0), skip_special_tokens=True)[0]
        decode_time = time.time() - decode_start
        print(f"[INFER] ⏱️  Decoding: {decode_time * 1000:.1f}ms")
        print(f"[INFER] Generated {new_ids.shape[0]} tokens")
        print(f"[INFER] Raw output: {raw_output[:200]}...")

    except Exception as e:
        print(f"[ERROR] Model inference failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")

    # 6. 解析结果
    print(f"[INFER] Parsing output...")
    fallback_params = config.default_params or [param_config[k]["range"][0] for k in param_order]
    params_dict, params_array = parse_qwen_output(raw_output, param_order, fallback_params)

    inference_time = time.time() - start_time
    print(f"[INFER] ✓ Inference complete! Total time: {inference_time:.2f}s")
    print(f"[INFER] Parameters: {params_array}")

    return InferenceResponse(
        parameters=params_dict,
        parameters_array=params_array,
        raw_output=raw_output,
        inference_time=inference_time,
        success=True
    )


@app.post("/infer_file")
async def infer_from_file(
        file: UploadFile = File(...),
        linear_vel: float = Form(0.0),
        angular_vel: float = Form(0.0),
        algorithm: str = Form("DWA")
):
    """
    接收文件上传的推理接口
    """
    # 读取上传的图像
    image_data = await file.read()
    image_base64 = base64.b64encode(image_data).decode('utf-8')

    # 调用主推理接口
    request = InferenceRequest(
        image_base64=image_base64,
        linear_vel=linear_vel,
        angular_vel=angular_vel,
        algorithm=algorithm
    )

    return infer_parameters(request)


@app.get("/algorithms")
async def list_algorithms():
    """返回支持的算法列表"""
    return {
        "algorithms": list(ALGORITHM_PARAMS.keys()),
        "details": {
            alg: {
                "num_params": len(params),
                "parameters": list(params.keys())
            }
            for alg, params in ALGORITHM_PARAMS.items()
        }
    }


# ============================================================
# 命令行参数和启动
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL Navigation Parameter Prediction Service")

    # 模型配置
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Base model ID or path"
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Path to LoRA checkpoint"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device map (auto/cuda:0/cpu)"
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default=None,
        help="Override device map (e.g., auto). If set, supersedes --device"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Max tokens to generate"
    )
    parser.add_argument(
        "--stop_on_json_complete",
        action="store_true",
        default=True,
        help="Stop generation as soon as complete JSON with required keys is produced"
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Force 4-bit quantized loading on GPU (requires bitsandbytes)"
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Force 8-bit quantized loading on GPU (requires bitsandbytes)"
    )
    parser.add_argument(
        "--max_gpu_memory_gb",
        type=int,
        default=None,
        help="Max GPU memory cap for auto device_map fallback"
    )
    parser.add_argument(
        "--startup_warmup",
        action="store_true",
        help="Run a non-blocking warmup inference after model load"
    )
    parser.add_argument(
        "--startup_tokens",
        type=int,
        default=16,
        help="Max new tokens for warmup generation"
    )

    # 算法配置
    parser.add_argument(
        "--algorithm",
        type=str,
        default="DWA",
        choices=list(ALGORITHM_PARAMS.keys()),
        help="Default algorithm"
    )
    parser.add_argument(
        "--default_params",
        type=float,
        nargs='+',
        default=None,
        help="Fallback parameters (space-separated)"
    )

    # 服务器配置
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Server port"
    )

    return parser.parse_args()


if __name__ == "__main__":
    config = parse_args()

    print(f"""
╔═══════════════════════════════════════════════════════════╗
║   Qwen2.5-VL Navigation Parameter Prediction Service     ║
╚═══════════════════════════════════════════════════════════╝
    Base Model: {config.base_model}
    LoRA Path:  {config.lora_path or 'None'}
    Device:     {config.device}
    Algorithm:  {config.algorithm}
    Host:       {config.host}:{config.port}
    """)

    uvicorn.run(app, host=config.host, port=config.port)
