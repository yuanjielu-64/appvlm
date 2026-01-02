#!/home/yuanjielu/miniforge3/envs/lmms-finetune-qwen/bin/python
"""
Qwen2.5-VL Navigation Parameter Prediction - Flash Attention Version
支持 Benchmark 模式和 HTTP 服务模式
"""

import argparse
import os
import time
import json
import re
import base64
import io
from typing import List, Dict, Any, Optional

import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# FastAPI imports (仅在服务模式需要)
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("[WARN] FastAPI not available, HTTP service mode disabled")

# ===========================================================
# Efficiency Parameters
# ===========================================================
USE_FLASH_ATTENTION = True
USE_COMPILE = False  # ✅ 禁用 torch.compile 避免 CUDA Graphs 内存错误


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
        "inflation_radius": {"range": [0.1, 0.6], "type": "float"}
    }
}

# FastAPI 数据模型
if FASTAPI_AVAILABLE:
    class InferenceRequest(BaseModel):
        image_base64: Optional[str] = Field(None, description="Base64编码的图像")
        image_path: Optional[str] = Field(None, description="图像文件路径")
        linear_vel: float = Field(default=0.0)
        angular_vel: float = Field(default=0.0)
        algorithm: str = Field(default="DWA")

    class InferenceResponse(BaseModel):
        parameters: Dict[str, Any]
        parameters_array: List[float]
        hidden_states_shape: str
        inference_time: float
        success: bool

    class HealthResponse(BaseModel):
        status: str
        model_loaded: bool
        device: str
        algorithm: str

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
# 工具函数
# ============================================================

@torch.no_grad()
def forward_with_hidden_states(model, processor, inputs, num_layers=4, layer_indices=None):
    """
    提取 Qwen2.5-VL 的多层 hidden states（用于 DPT head）

    Args:
        model: Qwen2.5-VL model
        processor: Qwen processor
        inputs: 包含 input_ids, pixel_values 等的字典
        num_layers: 提取多少层（默认 4 层）
        layer_indices: 指定提取哪些层（例如 [-4, -3, -2, -1]）

    Returns:
        multi_layer_hidden_states: List[Tensor], 每个 [B, seq_len, hidden_size]
    """
    # 只做一次前向传播，不生成新 token
    model_inputs = {
        "input_ids": inputs["input_ids"],
        "pixel_values": inputs.get("pixel_values"),
        "image_grid_thw": inputs.get("image_grid_thw"),
        "video_grid_thw": inputs.get("video_grid_thw"),
        "attention_mask": inputs.get("attention_mask"),
        "output_hidden_states": True,  # 关键：输出所有层的 hidden states
    }

    # 前向传播
    outputs = model.model(**model_inputs)

    # outputs.hidden_states 是一个 tuple，包含所有层的 hidden states
    # 每个元素的形状: [B, seq_len, hidden_size]
    all_hidden_states = outputs.hidden_states

    # 提取指定的层
    if layer_indices is None:
        # 默认提取最后 num_layers 层
        layer_indices = list(range(-num_layers, 0))

    multi_layer_hidden_states = [all_hidden_states[idx] for idx in layer_indices]

    return multi_layer_hidden_states

# ============================================================
# 核心逻辑
# ============================================================

def load_model(config):
    print("=" * 60)
    print("Loading Qwen2.5-VL Model...")
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

    def _load_base_model(**kwargs):
        # 自动选择 attention 实现
        if USE_FLASH_ATTENTION and flash_attn_available:
            attn_impl = "flash_attention_2"
        elif USE_FLASH_ATTENTION and not flash_attn_available:
            print("⚠️  Falling back to SDPA (flash_attn not available)")
            attn_impl = "sdpa"
        else:
            attn_impl = "default"

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.base_model,
            attn_implementation=attn_impl,
            **kwargs,
        )
        if USE_COMPILE and hasattr(torch, 'compile'):
            print("[LOAD] Compiling model with torch.compile()...")
            model = torch.compile(model, mode="reduce-overhead")
        return model

    try:
        resolved_device_map = config.device_map if getattr(config, 'device_map', None) else config.device

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

            print(f"[LOAD] Loading model with quantization (4bit={use_4bit}, 8bit={use_8bit}), device_map={resolved_device_map}")
            model = _load_base_model(**quant_kwargs)
        else:
            model_kwargs = {
                "dtype": dtype,
                "torch_dtype": dtype,
                "device_map": resolved_device_map,
            }
            model = _load_base_model(**model_kwargs)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("[LOAD] CUDA OOM during standard load. Retrying with 4-bit quantization on GPU (device_map=auto)...")
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
                max_gpu_gb = getattr(config, 'max_gpu_memory_gb', None)
                if max_gpu_gb is None:
                    max_gpu_gb = 8
                max_memory = {"cuda:0": f"{int(max_gpu_gb)}GiB", "cpu": "48GiB"}
                model = _load_base_model(
                    torch_dtype=dtype,
                    device_map="auto",
                    max_memory=max_memory,
                )
        else:
            raise

    if config.lora_path:
        print(f"Loading LoRA from: {config.lora_path}")

        # 1. 加载 PEFT adapter (LoRA weights)
        # 注意：is_trainable=True 允许跳过未训练的层（空tensor）
        try:
            model = PeftModel.from_pretrained(
                model,
                config.lora_path,
                is_trainable=False  # 推理模式，允许跳过未初始化的LoRA层
            )
        except Exception as e:
            print(f"[WARN] Failed to load LoRA with strict mode, trying with is_trainable=True...")
            print(f"[WARN] Error: {e}")
            # Fallback: 尝试创建adapter config然后手动加载非空的权重
            from peft import get_peft_model, LoraConfig
            import json

            adapter_config_path = os.path.join(config.lora_path, 'adapter_config.json')
            with open(adapter_config_path) as f:
                adapter_config = json.load(f)

            # 只保留实际训练的模块
            lora_config = LoraConfig(
                r=adapter_config['r'],
                lora_alpha=adapter_config['lora_alpha'],
                target_modules=adapter_config['target_modules'],
                lora_dropout=adapter_config.get('lora_dropout', 0.0),
                bias=adapter_config.get('bias', 'none'),
                task_type=adapter_config.get('task_type', 'CAUSAL_LM'),
            )

            model = get_peft_model(model, lora_config)

            # 手动加载非空的权重
            from safetensors.torch import load_file
            adapter_weights_path = os.path.join(config.lora_path, 'adapter_model.safetensors')
            state_dict = load_file(adapter_weights_path)

            # 只加载非空的权重
            filtered_state_dict = {k: v for k, v in state_dict.items() if v.numel() > 0}
            print(f"[INFO] Loading {len(filtered_state_dict)}/{len(state_dict)} non-empty LoRA weights")

            model.load_state_dict(filtered_state_dict, strict=False)
            print("[INFO] LoRA weights loaded successfully (filtered mode)")

        # 2. 加载 Regression head (如果存在)
        regression_head_path = os.path.join(config.lora_path, 'regression_head', 'pytorch_model.bin')
        if os.path.exists(regression_head_path):
            print(f"✓ Loading regression head from: {regression_head_path}")

            # 导入 regression head 类（需要确保在 path 中）
            import sys
            script_dir = os.path.dirname(os.path.abspath(__file__))
            vlm_pipeline_path = os.path.abspath(os.path.join(script_dir, '../../vlm_pipeline/lmms-finetune-qwen'))

            if vlm_pipeline_path not in sys.path:
                sys.path.insert(0, vlm_pipeline_path)

            from models.qwen2_5_vl_dpt_regression import DPTHead, SimpleMLP, TransformerHead

            # 自动从 base_model 获取 hidden_size
            hidden_size = model.config.hidden_size
            print(f"✓ Detected hidden_size: {hidden_size}")

            # 创建 regression head 实例（根据你的配置）
            head_type = getattr(config, 'head_type', 'dpt')  # 默认 dpt
            num_params = getattr(config, 'num_params', 6)    # 默认 6 (DDP)

            if head_type == 'dpt':
                regression_head = DPTHead(
                    hidden_size=hidden_size,  # 自动从模型读取
                    num_params=num_params,
                    feature_dim=256,
                    num_layers=4,
                )
            elif head_type == 'transformer':
                regression_head = TransformerHead(
                    hidden_size=hidden_size,
                    num_params=num_params,
                )
            else:  # simple_mlp
                regression_head = SimpleMLP(
                    hidden_size=hidden_size,
                    num_params=num_params,
                )

            # 加载权重（处理 DeepSpeed ZeRO-3 bug 导致的空 tensor）
            state_dict = torch.load(regression_head_path, map_location='cpu')
            filtered_state_dict = {k: v for k, v in state_dict.items() if v.numel() > 0}
            num_empty = len(state_dict) - len(filtered_state_dict)

            if num_empty > 0:
                print(f"[WARN] Found {num_empty}/{len(state_dict)} empty tensors in regression_head (DeepSpeed bug)")
                print(f"[INFO] Loading {len(filtered_state_dict)}/{len(state_dict)} non-empty weights with strict=False")
                missing_keys, unexpected_keys = regression_head.load_state_dict(filtered_state_dict, strict=False)
                if missing_keys:
                    print(f"[WARN] Missing keys (will use random init): {missing_keys[:5]}...")
            else:
                regression_head.load_state_dict(state_dict)

            # 移动到正确的设备和 dtype（与模型保持一致）
            regression_head.to(device=model.device, dtype=model.dtype)
            regression_head.eval()

            # 将 regression head 附加到 model
            model.regression_head = regression_head
            print(f"✓ Regression head loaded: {head_type}, num_params={num_params}, dtype={model.dtype}")

            # 加载归一化参数（用于反归一化）
            normalization_dir = os.path.join(config.lora_path, 'normalization')
            param_mean_path = os.path.join(normalization_dir, 'param_mean.npy')
            param_std_path = os.path.join(normalization_dir, 'param_std.npy')

            if os.path.exists(param_mean_path) and os.path.exists(param_std_path):
                import numpy as np
                param_mean = np.load(param_mean_path)
                param_std = np.load(param_std_path)

                # 转换为 tensor 并存储在 model 中
                model.param_mean = torch.tensor(param_mean, dtype=model.dtype, device=model.device)
                model.param_std = torch.tensor(param_std, dtype=model.dtype, device=model.device)
                print(f"✓ Loaded normalization stats from {normalization_dir}")
                print(f"  param_mean: {param_mean}")
                print(f"  param_std:  {param_std}")
            else:
                print(f"⚠️  Normalization stats not found at {normalization_dir}")
                print("   Predictions will NOT be denormalized!")
                model.param_mean = None
                model.param_std = None
        else:
            print(f"⚠️  Regression head not found at {regression_head_path}")

    model.eval()

    processor = AutoProcessor.from_pretrained(
        config.base_model,
        min_pixels=256 * 28 * 28,
        max_pixels=384 * 28 * 28
    )

    load_time = time.time() - start_time
    print(f"✓ Model loaded in {load_time:.2f}s")
    print(f"✓ Device: {model.device}")
    
    return model, processor

def infer_once(model, processor, config, image_path, linear_vel=0.0, angular_vel=0.0):
    print(f"\n[INFER] Processing: {image_path}")
    
    start_time = time.time()

    # 1. Load Image
    t1_start = time.time()
    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        return None
    image = Image.open(image_path).convert("RGB")
    t1 = time.time() - t1_start
    print(f"[INFER] ⏱️  Image loading: {t1 * 1000:.1f}ms")

    # 2. Prompt
    system_prompt = SYSTEM_PROMPT.strip()
    user_prompt = USER_PROMPT.format(
        linear_vel=linear_vel,
        angular_vel=angular_vel,
        algorithm="DWA" # Default or placeholder
    )

    # 3. Inputs
    t4_start = time.time()
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
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(images=[image], text=text, return_tensors="pt").to(model.device)
    t4 = time.time() - t4_start
    print(f"[INFER] ⏱️  Input preparation: {t4 * 1000:.1f}ms")

    # 4. Generate (Forward with Hidden States)
    t_gen_start = time.time()
    
    # Call the custom forward function
    hidden_states = forward_with_hidden_states(
        model, 
        processor, 
        inputs, 
        max_new_tokens=config.max_new_tokens
    )
    
    t_gen = time.time() - t_gen_start
    print(f"[INFER] ⏱️  forward_with_hidden_states(): {t_gen:.3f}s")
    
    total_time = time.time() - start_time
    print(f"[INFER] ✓ Total time: {total_time:.2f}s")
    print(f"[INFER] Generated {len(hidden_states)} hidden states.")
    print(f"[INFER] Total Outputs: {torch.concat(hidden_states).shape} Tensor")

    return total_time

# ============================================================
# FastAPI 服务
# ============================================================
if FASTAPI_AVAILABLE:
    app = FastAPI(title="Qwen2.5-VL Flash Attention Service")

    # 全局变量
    _model = None
    _processor = None
    _config = None

    @app.on_event("startup")
    async def startup():
        global _model, _processor, _config
        print("=" * 60)
        print("Loading model for HTTP service...")
        print("=" * 60)
        _model, _processor = load_model(_config)
        print(f"✓ Model loaded on device: {_model.device}")
        print("=" * 60)

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        return HealthResponse(
            status="ok" if _model is not None else "loading",
            model_loaded=_model is not None,
            device=str(_model.device) if _model else "unknown",
            algorithm=_config.algorithm if _config else "DWA"
        )

    @app.post("/infer", response_model=InferenceResponse)
    async def infer(request: InferenceRequest):
        if _model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        start_time = time.time()
        print("\n" + "="*60)
        print(f"[INFER] Processing request for {request.algorithm}")
        print("="*60)

        # 1. 加载图像
        t1_start = time.time()
        try:
            if request.image_base64:
                image_data = base64.b64decode(request.image_base64)
                image = Image.open(io.BytesIO(image_data)).convert("RGB")
            elif request.image_path:
                abs_path = os.path.abspath(request.image_path)
                if not os.path.exists(abs_path):
                    raise FileNotFoundError(f"Image not found: {abs_path}")
                image = Image.open(abs_path).convert("RGB")
            else:
                raise HTTPException(status_code=400, detail="Must provide image_base64 or image_path")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to load image: {str(e)}")
        t1 = time.time() - t1_start
        print(f"[INFER] ⏱️  Image loading: {t1 * 1000:.1f}ms (size: {image.size})")

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
        text = _processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        t3_template = time.time() - t3_template_start

        # 3.2 处理器 (vision + text)
        t3_process_start = time.time()
        inputs = _processor(images=[image], text=text, return_tensors="pt")
        t3_process = time.time() - t3_process_start

        # 3.3 移动到设备
        t3_to_device_start = time.time()
        inputs = inputs.to(_model.device)
        t3_to_device = time.time() - t3_to_device_start

        t3 = time.time() - t3_start
        print(f"[INFER] ⏱️  Input preparation: {t3 * 1000:.1f}ms")
        print(f"[INFER]    ├─ Template: {t3_template * 1000:.1f}ms")
        print(f"[INFER]    ├─ Processor (vision+text): {t3_process * 1000:.1f}ms")
        print(f"[INFER]    └─ To device: {t3_to_device * 1000:.1f}ms")

        # 🔍 输入信息
        if 'pixel_values' in inputs:
            pix_shape = inputs['pixel_values'].shape
            print(f"[INFER] 📸 Vision shape (pixel_values): {pix_shape}")
        if 'input_ids' in inputs:
            print(f"[INFER] 📝 Input tokens: {inputs['input_ids'].shape[1]}")

        # 4. 模型推理（提取多层 hidden states）
        t4_start = time.time()
        multi_layer_hidden_states = forward_with_hidden_states(
            _model,
            _processor,
            inputs,
            num_layers=4,  # 提取最后 4 层
            layer_indices=[-4, -3, -2, -1]  # 或者指定具体层
        )
        t4 = time.time() - t4_start
        print(f"[INFER] ⏱️  forward_with_hidden_states(): {t4:.3f}s")
        print(f"[INFER] 📦 Extracted {len(multi_layer_hidden_states)} layers")

        # 5. 使用 regression head 预测参数
        t5_start = time.time()
        param_config = ALGORITHM_PARAMS.get(request.algorithm, ALGORITHM_PARAMS["DWA"])
        param_order = list(param_config.keys())

        # ✅ 真正调用 regression head 进行预测
        if hasattr(_model, 'regression_head'):
            head_type = getattr(_config, 'head_type', 'dpt')

            with torch.no_grad():
                if head_type == 'dpt':
                    # DPT head 需要多层 hidden states (List[Tensor])
                    predicted_params = _model.regression_head(multi_layer_hidden_states)
                else:
                    # SimpleMLP 和 TransformerHead 只需要最后一层
                    last_hidden_state = multi_layer_hidden_states[-1]
                    predicted_params = _model.regression_head(last_hidden_state)

                # 反归一化（恢复到原始参数范围）
                if hasattr(_model, 'param_mean') and _model.param_mean is not None:
                    predicted_params_normalized = predicted_params.squeeze(0)  # [num_params]
                    predicted_params_denorm = predicted_params_normalized * _model.param_std + _model.param_mean
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
        print("="*60 + "\n")

        return InferenceResponse(
            parameters=params_dict,
            parameters_array=predicted_params,
            hidden_states_shape=hidden_states_shape,
            inference_time=inference_time,
            success=True
        )

def parse_args():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL Benchmark or HTTP Service")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--head_type", type=str, default="dpt", choices=["simple_mlp", "transformer", "dpt"])
    parser.add_argument("--num_params", type=int, default=6, help="Number of parameters to predict")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--device_map", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=1)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--max_gpu_memory_gb", type=int, default=None)

    # HTTP服务参数
    parser.add_argument("--algorithm", type=str, default="DWA", help="Planning algorithm (DWA/TEB/MPPI/DDP)")
    parser.add_argument("--port", type=int, default=5000, help="HTTP service port")
    parser.add_argument("--startup_warmup", action="store_true", help="Warmup on startup")
    parser.add_argument("--startup_tokens", type=int, default=16, help="Warmup token count")
    parser.add_argument("--use_flash_attention", action="store_true", help="Enable FlashAttention")
    parser.add_argument("--optimize_memory", action="store_true", help="Enable memory optimization")
    parser.add_argument("--no_optimizations", action="store_true", help="Disable all optimizations")
    parser.add_argument("--no_cuda_timing", action="store_true", help="Disable CUDA timing")
    parser.add_argument("--enable_profiler", action="store_true", help="Enable detailed profiler")

    # Benchmark模式参数
    parser.add_argument("--test_image", type=str, default="/data/local/yl2832/appvlm_ws/src/ros_jackal/buffer/dwa_qwen/actor_0/VLM_000000.png")
    parser.add_argument("--loops", type=int, default=5, help="Number of inference loops (benchmark mode)")
    parser.add_argument("--benchmark", action="store_true", help="Run in benchmark mode instead of HTTP service")
    return parser.parse_args()

if __name__ == "__main__":
    config = parse_args()

    if config.benchmark:
        # Benchmark模式
        model, processor = load_model(config)

        print("\n" + "=" * 60)
        print(f"{USE_FLASH_ATTENTION=}, {USE_COMPILE=}")

        print("=" * 60)
        print(f"Starting Benchmark Loop ({config.loops} iterations)...")
        print("=" * 60)

        times = []
        for i in range(config.loops):
            print(f"--- Loop {i+1}/{config.loops} ---")
            t = infer_once(model, processor, config, config.test_image)
            if t: times.append(t)

        if times:
            avg_time = sum(times) / len(times)
            print("=" * 60)
            print(f"Average Inference Time: {avg_time:.3f}s")
            print("=" * 60)
    else:
        # HTTP服务模式
        if not FASTAPI_AVAILABLE:
            print("=" * 60)
            print("⚠️  FastAPI not installed!")
            print("=" * 60)
            print("Install with: pip install fastapi uvicorn")
            exit(1)

        # 设置全局配置（供startup使用）
        _config = config

        print("=" * 60)
        print(f"Starting HTTP Service on port {config.port}")
        print(f"Algorithm: {config.algorithm}")
        print("=" * 60)

        # 启动uvicorn服务
        uvicorn.run(app, host="0.0.0.0", port=config.port, log_level="info")
