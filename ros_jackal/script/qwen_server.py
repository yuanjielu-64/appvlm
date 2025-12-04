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
    "Jackal Robot (0.508m×0.430m). Current velocity: linear={linear_vel}m/s, angular={angular_vel}rad/s. "
    "Predict {number} {algorithm} parameters for the navigation scene.\n"
    "SCENE: Red=obstacles (laser scan), Purple=global path, Yellow=robot, Grid=1m spacing.\n"
    "STRATEGY: Dense obstacles/narrow→reduce max_vel_x; Open space→increase max_vel_x; "
    "Sharp turns→increase max_vel_theta.\n"
    "PARAM ORDER: {param_order}\n"
    "RANGES: {param_ranges}\n"
    "OUTPUT FORMAT (VERY IMPORTANT):\n"
    "- Return ONLY one line.\n"
    "- The line MUST contain exactly {number} numbers.\n"
    "- Use space as the separator, no commas, no brackets, no extra text.\n"
    "- Example: 0.5 0.314 8 40 0.5 1.5 0.2\n"
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

def generate_output_format(param_config: Dict) -> tuple:
    """生成简洁的参数顺序和范围说明 (返回: param_order_str, param_ranges_str)"""
    param_order = []
    param_ranges = []

    for param_name, param_info in param_config.items():
        param_order.append(param_name)
        range_str = f"{param_info['range'][0]}-{param_info['range'][1]}"
        param_ranges.append(f"{param_name}:{range_str}")

    param_order_str = ", ".join(param_order)
    param_ranges_str = ", ".join(param_ranges)

    return param_order_str, param_ranges_str


def parse_qwen_output(result: str, param_order: List[str], fallback_params: List[float]) -> tuple:
    """
    解析Qwen输出，优先使用“空格分隔数字一行”的格式；
    若失败，再回退到原来的 JSON 解析逻辑。
    """
    try:
        cleaned = result.strip()

        # 1️⃣ 优先：直接提取所有数字 (int/float)
        number_strs = re.findall(r'[-+]?\d*\.\d+|[-+]?\d+', cleaned)
        if len(number_strs) >= len(param_order):
            values = [float(x) for x in number_strs[:len(param_order)]]
            params_dict = {k: v for k, v in zip(param_order, values)}
            return params_dict, values

        # 2️⃣ 不够再走你原来的 JSON 逻辑
        # —— 下面是你原来的代码，可以基本原样保留，只把 cleaned 复用
        # 去除 markdown 标记
        if cleaned.startswith('```'):
            match = re.search(r'```(?:json)?\s*([\[\{].*?[\]\}])\s*```', cleaned, re.DOTALL)
            if match:
                cleaned = match.group(1)
            else:
                cleaned = cleaned.replace("```json", "").replace("```", "").strip()

        # 后面你的 JSON 提取逻辑保持不变……
        # ...
        parsed = json.loads(cleaned)

        if isinstance(parsed, list):
            param_array = parsed[:len(param_order)]
            if len(param_array) < len(param_order):
                param_array.extend(fallback_params[len(param_array):])
            params_dict = {key: val for key, val in zip(param_order, param_array)}
            return params_dict, param_array

        elif isinstance(parsed, dict):
            if "inflation_radius" not in parsed and "final_inflation" in parsed:
                parsed["inflation_radius"] = parsed.pop("final_inflation")
            param_array = [parsed.get(key, fallback_params[i]) for i, key in enumerate(param_order)]
            return parsed, param_array

        else:
            raise ValueError(f"Unexpected JSON type: {type(parsed)}")

    except Exception as e:
        print(f"[ERROR] Parse failed: {e}, using fallback params")
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

    # 加载处理器 - 🚀 优化: 大幅降低视觉分辨率以加速推理
    processor = AutoProcessor.from_pretrained(
        config.base_model,
        min_pixels=64 * 28 * 28,    # 降低4倍 (128->64)
        max_pixels=128 * 28 * 28     # 保持上限但会因min_pixels自动降低
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
            param_order_str, param_ranges_str = generate_output_format(param_config)

            test_prompt = PROMPT_TEMPLATE.format(
                number=len(param_config),
                algorithm=config.algorithm,
                linear_vel=0.0,
                angular_vel=0.0,
                param_order=param_order_str,
                param_ranges=param_ranges_str
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
    param_order_str, param_ranges_str = generate_output_format(param_config)
    t2 = time.time() - t2_start
    print(f"[INFER] ⏱️  Config setup: {t2 * 1000:.1f}ms")

    # 3. 构建prompt
    t3_start = time.time()
    prompt = PROMPT_TEMPLATE.format(
        number=len(param_config),
        algorithm=algorithm,
        linear_vel=round(request.linear_vel, 4),
        angular_vel=round(request.angular_vel, 4),
        param_order=param_order_str,
        param_ranges=param_ranges_str
    )
    t3 = time.time() - t3_start
    print(f"[INFER] ⏱️  Prompt building: {t3 * 1000:.1f}ms (length: {len(prompt)} chars)")

    # 4. 准备输入（图像编码 + tokenization）
    t4_start = time.time()

    # 4.1 构建消息
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # 4.2 应用chat template
    t4_template_start = time.time()
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    t4_template = time.time() - t4_template_start

    # 4.3 完整处理 (包含vision encoder + tokenization)
    t4_process_start = time.time()

    # 🔥 使用CUDA事件计时图像编码
    use_cuda_timing = torch.cuda.is_available() and getattr(config, 'cuda_timing', True)
    if use_cuda_timing:
        torch.cuda.synchronize()
        cuda_img_start = torch.cuda.Event(enable_timing=True)
        cuda_img_end = torch.cuda.Event(enable_timing=True)
        cuda_img_start.record()

    inputs = processor(
        images=[image],
        text=text,
        return_tensors="pt",
    )

    if use_cuda_timing and 'pixel_values' in inputs:
        cuda_img_end.record()
        torch.cuda.synchronize()
        cuda_img_time = cuda_img_start.elapsed_time(cuda_img_end)
        print(f"[INFER]    ├─ 🎯 Image encoding [CUDA]: {cuda_img_time:.1f}ms")

    t4_process = time.time() - t4_process_start

    # 4.4 移动到GPU
    t4_to_device_start = time.time()
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    t4_to_device = time.time() - t4_to_device_start

    t4 = time.time() - t4_start

    # 📊 详细输出
    print(f"[INFER] ⏱️  Input preparation: {t4 * 1000:.1f}ms")
    print(f"[INFER]    ├─ Template: {t4_template * 1000:.1f}ms")
    print(f"[INFER]    ├─ Processor (vision+text): {t4_process * 1000:.1f}ms")
    print(f"[INFER]    └─ To device: {t4_to_device * 1000:.1f}ms")

    # 🔍 额外信息
    if 'pixel_values' in inputs:
        pix_shape = inputs['pixel_values'].shape
        print(f"[INFER] 📸 Vision shape (pixel_values): {pix_shape}")
    if 'input_ids' in inputs:
        print(f"[INFER] 📝 Input tokens: {inputs['input_ids'].shape[1]}")

    # 5. 模型推理 - 详细时间分析
    try:
        # 5.1 准备 stopping criteria
        t_stopping_start = time.time()
        stopping = None
        expected_count = len(param_order)
        max_new = min(config.max_new_tokens, expected_count * 3)
        if getattr(config, 'stop_on_json_complete', True):

            class JsonStopper(StoppingCriteria):
                def __init__(self, tokenizer, input_len, expected_length, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.tok = tokenizer
                    self.input_len = input_len
                    self.expected_count = expected_length
                    self._last_check_len = input_len
                    self._cached_text = ""

                def __call__(self, input_ids, scores, **kwargs):
                    try:
                        cur_len = input_ids.shape[1]
                        if cur_len > self._last_check_len:
                            new_tokens = input_ids[0, self._last_check_len:]
                            new_text = self.tok.decode(new_tokens, skip_special_tokens=True)
                            self._cached_text += new_text
                            self._last_check_len = cur_len

                        text = self._cached_text
                        nums = re.findall(r'[-+]?\d*\.\d+|[-+]?\d+', text)
                        if len(nums) >= self.expected_count:
                            return True

                        return False
                    except Exception as e:
                        print(f"[WARNING] JsonStopper error: {e}")
                        return False

            print(f"[DEBUG] Expected array length: {expected_count}")
            stopping = StoppingCriteriaList([
                JsonStopper(processor.tokenizer, inputs["input_ids"].shape[1], expected_count)
            ])
        else:
            stopping = None

        t_stopping = time.time() - t_stopping_start
        print(f"[INFER] ⏱️  Stopping criteria setup: {t_stopping * 1000:.1f}ms")

        # 5.2 模型生成 (详细计时)
        generate_start = time.time()

        # 🔥 使用CUDA事件进行精确计时
        use_cuda_timing = torch.cuda.is_available() and getattr(config, 'cuda_timing', True)
        if use_cuda_timing:
            torch.cuda.synchronize()
            cuda_start = torch.cuda.Event(enable_timing=True)
            cuda_end = torch.cuda.Event(enable_timing=True)
            cuda_start.record()

        # 🔍 可选: 启用profiler (只在debug模式)
        enable_profiler = getattr(config, 'enable_profiler', False)

        with torch.inference_mode():
            if enable_profiler:
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    record_shapes=True,
                    with_stack=True,
                ) as prof:
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_new,
                        do_sample=False,
                        stopping_criteria=stopping,
                        use_cache=True,
                        num_beams=1,
                        pad_token_id=processor.tokenizer.pad_token_id,
                        eos_token_id=processor.tokenizer.eos_token_id,
                    )

                # 输出profiler统计
                print("\n" + "="*60)
                print("🔍 PROFILER REPORT (Top 10 operations)")
                print("="*60)
                print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
                print("="*60 + "\n")
            else:
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=config.max_new_tokens,
                    do_sample=False,
                    stopping_criteria=stopping,
                    use_cache=True,
                    num_beams=1,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                )

        if use_cuda_timing:
            cuda_end.record()
            torch.cuda.synchronize()
            cuda_time = cuda_start.elapsed_time(cuda_end) / 1000.0  # ms -> s
            print(f"[INFER] ⏱️  model.generate() [CUDA precise]: {cuda_time:.3f}s")

        generate_time = time.time() - generate_start
        print(f"[INFER] ⏱️  model.generate() [Wall time]: {generate_time:.3f}s")

        # 📊 显存统计
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            mem_reserved = torch.cuda.memory_reserved() / 1024**3
            mem_max = torch.cuda.max_memory_allocated() / 1024**3
            print(f"[INFER] 💾 GPU Memory: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved, {mem_max:.2f}GB peak")

        # 5.3 解码输出
        decode_start = time.time()
        input_len = inputs["input_ids"].shape[1]
        new_ids = generated_ids[0, input_len:]
        raw_output = processor.batch_decode(new_ids.unsqueeze(0), skip_special_tokens=True)[0]
        decode_time = time.time() - decode_start

        num_tokens = new_ids.shape[0]
        tokens_per_sec = num_tokens / generate_time if generate_time > 0 else 0
        ms_per_token = (generate_time * 1000) / num_tokens if num_tokens > 0 else 0

        print(f"[INFER] ⏱️  Decoding: {decode_time * 1000:.1f}ms")
        print(f"[INFER] 📝 Generated {num_tokens} tokens")
        print(f"[INFER] 🚀 Speed: {tokens_per_sec:.1f} tokens/s ({ms_per_token:.1f} ms/token)")
        print(f"[INFER] 📄 Raw output: {raw_output[:200]}...")

    except Exception as e:
        print(f"[ERROR] Model inference failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")

    # 6. 解析结果
    t6_start = time.time()
    print(f"[INFER] Parsing output...")
    fallback_params = config.default_params or [param_config[k]["range"][0] for k in param_order]
    params_dict, params_array = parse_qwen_output(raw_output, param_order, fallback_params)
    t6 = time.time() - t6_start

    inference_time = time.time() - start_time

    # 📊 总结性能报告
    print("\n" + "="*60)
    print("📊 INFERENCE PERFORMANCE SUMMARY")
    print("="*60)
    print(f"⏱️  Total time:        {inference_time:.3f}s")
    print(f"   ├─ Image load:      {t1*1000:.1f}ms ({t1/inference_time*100:.1f}%)")
    print(f"   ├─ Config setup:    {t2*1000:.1f}ms ({t2/inference_time*100:.1f}%)")
    print(f"   ├─ Prompt build:    {t3*1000:.1f}ms ({t3/inference_time*100:.1f}%)")
    print(f"   ├─ Input prep:      {t4*1000:.1f}ms ({t4/inference_time*100:.1f}%)")
    print(f"   ├─ Model generate:  {generate_time:.3f}s ({generate_time/inference_time*100:.1f}%) ⚡")
    print(f"   ├─ Decode:          {decode_time*1000:.1f}ms ({decode_time/inference_time*100:.1f}%)")
    print(f"   └─ Parse output:    {t6*1000:.1f}ms ({t6/inference_time*100:.1f}%)")
    print(f"🚀 Throughput:       {tokens_per_sec:.1f} tokens/s")
    print(f"📝 Output:           {num_tokens} tokens")
    print(f"✅ Parameters:       {params_array}")
    print("="*60 + "\n")

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
        default=30,
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

    # 🔍 性能分析配置
    parser.add_argument(
        "--enable_profiler",
        action="store_true",
        help="Enable PyTorch profiler for detailed performance analysis"
    )
    parser.add_argument(
        "--cuda_timing",
        action="store_true",
        default=True,
        help="Use CUDA events for precise GPU timing (default: True)"
    )
    parser.add_argument(
        "--no_cuda_timing",
        action="store_false",
        dest="cuda_timing",
        help="Disable CUDA event timing"
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
