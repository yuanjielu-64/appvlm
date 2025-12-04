# 🚀 Qwen2.5-VL + evaluate_qwen_all.py 使用指南

## 📝 总结

你的 **`evaluate_qwen_all.py`** 是"服务员"，用来运行 Gazebo 仿真和评估。

现在它支持**两种VLM后端**：
1. **ChatGPT (GPT-4o)** - 默认，需要OpenAI API
2. **Qwen2.5-VL** - 本地模型，通过 `qwen_server.py` 提供服务

## 🎯 架构图

```
evaluate_qwen_all.py (Python 3.8 - ROS环境)
├─ 运行 Gazebo
├─ 遍历 300 worlds
└─ 每步调用 VLM 获取DWA参数
      ↓
   选择后端：
   ├─ ChatGPT: 直接HTTP调用OpenAI API
   └─ Qwen: HTTP调用本地qwen_server.py
               ↓
          qwen_server.py (Python 3.10 - Conda环境)
          └─ 加载 Qwen2.5-VL + LoRA 模型
          └─ 推理并返回参数
```

## 🏃 使用方式

### 方式1: 使用 ChatGPT (默认)

```bash
cd /home/yuanjielu/robot_navigation/noetic/appvlm_ws/src/ros_jackal/script

python3 evaluate_qwen_all.py \
  --id 0 \
  --policy_name dwa_test \
  --buffer_path ../buffer/ \
  --world_path ../jackal_helper/worlds/BARN1/ \
  --total_worlds 300
```

- ✅ 不需要额外启动服务
- ✅ 使用 `$OPENAI_API_KEY` 环境变量
- ❌ 需要付费API，有网络延迟

---

### 方式2: 使用 Qwen2.5-VL (本地)

#### 步骤1: 启动 Qwen 服务 (终端1)

```bash
cd /home/yuanjielu/robot_navigation/noetic/appvlm_ws/src/ros_jackal/script

# 修改 start_qwen_service.sh 中的 LORA_PATH，然后启动
./start_qwen_service.sh
```

**或者手动启动**:
```bash
/home/yuanjielu/miniforge3/envs/lmms-finetune-qwen/bin/python qwen_server.py \
  --base_model "Qwen/Qwen2.5-VL-7B-Instruct" \
  --lora_path "/path/to/your/lora/checkpoint" \
  --device auto \
  --algorithm DWA \
  --port 5000
```

**等待看到**:
```
✓ Model loaded in 15.23s
✓ Device: cuda:0
✓ Algorithm: DWA
Uvicorn running on http://0.0.0.0:5000
```

#### 步骤2: 运行评估 (终端2)

```bash
cd /home/yuanjielu/robot_navigation/noetic/appvlm_ws/src/ros_jackal/script

python3 evaluate_qwen_all.py \
  --id 0 \
  --policy_name dwa_test \
  --buffer_path ../buffer/ \
  --world_path ../jackal_helper/worlds/BARN1/ \
  --total_worlds 300 \
  --use_qwen \
  --qwen_url http://localhost:5000
```

**注意新增的参数**:
- `--use_qwen`: 启用Qwen后端
- `--qwen_url`: Qwen服务地址 (默认 `http://localhost:5000`)

#### 你会看到:
```
>>>>>>>>>> Using Qwen2.5-VL service at http://localhost:5000 <<<<<<<<<<
✓ Qwen service ready!
>>>>>>>>>> Starting to run 300 worlds <<<<<<<<<<

========== World 2/300: world_2.world ==========
--- World 2 - Run 1/2 ---
...
```

---

## 📊 两种方式对比

| 特性 | ChatGPT (GPT-4o) | Qwen2.5-VL (本地) |
|------|------------------|-------------------|
| **启动** | 无需额外步骤 | 需要先启动qwen_server.py |
| **成本** | 付费API (约$0.01/image) | 免费 (本地GPU) |
| **速度** | ~1-2秒/推理 | ~1-2秒/推理 |
| **显存** | 无 | ~14GB (7B模型) |
| **网络** | 需要外网 | 本地运行 |
| **可定制** | API固定 | 可微调LoRA |

---

## 🔍 代码变化

### 修改的文件

1. **`evaluate_qwen_all.py`**
   - 新增 `--use_qwen` 和 `--qwen_url` 参数
   - 自动选择 ChatGPT 或 Qwen 后端
   - 统一接口，透明切换

2. **新增文件**:
   - `qwen_server.py` - Qwen推理服务器
   - `qwen_client.py` - Python客户端库
   - `start_qwen_service.sh` - 启动脚本

### 核心逻辑 (evaluate_qwen_all.py:318-340)

```python
if use_qwen:
    # Qwen: 通过HTTP调用本地服务
    image_name = f"VLM_{vlm_client.img_id:06d}.png"
    image_path = os.path.join(file_sync.actor_dir, image_name)
    result = vlm_client.infer_from_path(image_path, linear_vel, angular_vel)
    act = result['parameters_array']
else:
    # ChatGPT: 直接调用OpenAI API
    act = vlm_client.evaluate_single(linear_vel, angular_vel)
```

---

## ⚙️ 配置

### Qwen服务配置 (start_qwen_service.sh)

```bash
# 修改这些参数
BASE_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
LORA_PATH="/path/to/your/lora/checkpoint"  # ← 改成你的路径
DEVICE="auto"  # auto / cuda:0 / cuda:1
ALGORITHM="DWA"  # DWA / TEB / MPPI / DDP
PORT=5000
```

### 超时设置 (evaluate_qwen_all.py:260)

```python
vlm_client = QwenClient(
    qwen_url=qwen_url,
    algorithm=algorithm,
    timeout=10.0  # ← 如果Qwen推理慢，增加这个值
)
```

---

## 🐛 故障排查

### 问题1: Qwen服务无法启动

**错误**: `Qwen service not available!`

**解决**:
```bash
# 检查服务是否运行
curl http://localhost:5000/health

# 检查端口占用
lsof -i :5000

# 查看Qwen服务日志
./start_qwen_service.sh  # 查看控制台输出
```

### 问题2: 图像找不到

**错误**: `Image not found: /path/to/VLM_000001.png`

**原因**: 环境没有生成VLM图像

**解决**: 确保环境配置中 `use_vlm=True`
```python
env_config["kwargs"]["use_vlm"] = True  # evaluate_qwen_all.py:249
```

### 问题3: 推理超时

**错误**: Qwen调用超时

**解决**: 增加timeout参数
```python
vlm_client = QwenClient(..., timeout=20.0)  # 从10秒改为20秒
```

### 问题4: CUDA OOM

**错误**: `CUDA out of memory`

**解决**:
- 使用更小的模型 (如 Qwen2.5-VL-2B)
- 减少batch size (当前是1，已经最小)
- 使用CPU: `--device cpu` (会很慢)

---

## 📈 性能对比

**预期推理时间** (每步):
- ChatGPT: 1-3秒 (网络延迟 + API处理)
- Qwen (GPU): 1-2秒 (纯推理时间)
- Qwen (CPU): 10-30秒 (不推荐)

**300 worlds 预计总时间**:
- 假设每个world平均100步
- 每步2秒
- 总计: 300 × 100 × 2 = **60000秒 ≈ 16.7小时**

---

## 💡 提示

1. **并行评估**: 可以启动多个进程，每个负责不同的worlds
   ```bash
   # 终端1: worlds 0-99
   python3 evaluate_qwen_all.py --id 0 --use_qwen --total_worlds 100

   # 终端2: worlds 100-199
   python3 evaluate_qwen_all.py --id 1 --use_qwen --total_worlds 100
   ```

2. **监控Qwen服务**: 访问 `http://localhost:5000/docs` 查看API文档和测试接口

3. **切换算法**: 修改 `--policy_name` 和 qwen_server.py 的 `--algorithm` 保持一致

---

## 📞 联系

遇到问题查看:
- `qwen_server.py` 的日志输出
- `QWEN_SERVICE_README.md` 详细文档
- `qwen_client.py` 中的示例代码

---

**Happy Navigation! 🤖**