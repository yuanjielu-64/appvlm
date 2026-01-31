# Qwen 动态 Checkpoint 切换系统

这个目录包含在 Hopper HPC 集群上运行 Qwen2.5-VL 动态 checkpoint 切换服务的所有脚本。

## 📁 文件说明

### 核心服务
- **`qwen_server_dynamic.py`** - FastAPI 推理服务，支持动态切换 checkpoint
  - `/health` - 健康检查和当前状态
  - `/list_checkpoints` - 列出所有可用 checkpoints
  - `/switch_checkpoint` - 切换到指定 checkpoint
  - `/infer` - 执行推理

### 启动脚本
- **`hopper_qwen_dynamic.slurm`** - 在 Hopper GPU 节点上启动服务的 SLURM 脚本
- **`start_qwen_dynamic.sh`** - 本地启动服务（测试用）

### 测试脚本
- **`run_test_on_hopper.sh`** - 主测试入口（推荐使用）
- **`test_checkpoint_inference.sh`** - 详细的 checkpoint 切换和推理测试
- **`switch_checkpoint.sh`** - 快速切换 checkpoint 的便捷脚本

### 恢复工具
- **`recover_checkpoints.py`** - 批量恢复被 DeepSpeed bug 损坏的 checkpoints

## 🚀 快速开始

### 1. 启动 Qwen 服务

在 Hopper 上提交 SLURM job:

```bash
cd /home/yuanjielu/robot_navigation/noetic/appvlm_ws/src/ros_jackal/script_HPC
sbatch hopper_qwen_dynamic.slurm
```

查看 job 状态和节点信息:
```bash
squeue -u $USER
# 输出示例:
#   JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
#  123456       gpu qwen_dyn    ylu22  R       0:42      1 gpu017
```

记录节点名（例如 `gpu017`），后续测试需要用到。

查看服务日志:
```bash
tail -f qwen_dynamic_*.out
```

### 2. 测试服务（推荐方法）

**方法 1: 使用自动化测试脚本**

```bash
# 设置服务节点
export QWEN_HOST=gpu017  # 👈 改为你的实际节点

# 使用默认测试图片
bash run_test_on_hopper.sh

# 或指定自定义图片
export TEST_IMAGE=/path/to/your/costmap.png
bash run_test_on_hopper.sh
```

这会自动:
1. 检查服务健康状态
2. 列出所有可用 checkpoints
3. 切换到 DDP checkpoint-12500 并推理
4. 切换到 DDP checkpoint-10000 并推理
5. 显示预测的参数和推理时间

**方法 2: 手动切换 checkpoint**

```bash
export QWEN_HOST=gpu017

# 切换到 DDP checkpoint-10000
bash switch_checkpoint.sh ddp 10000

# 切换到 DWA checkpoint-12500 (7个参数)
bash switch_checkpoint.sh dwa 12500 7

# 切换到 TEB checkpoint-5000
bash switch_checkpoint.sh teb 5000

# 切换到 MPPI checkpoint-12500
bash switch_checkpoint.sh mppi 12500
```

**方法 3: 直接使用 API**

```bash
export QWEN_HOST=gpu017
export QWEN_URL="http://${QWEN_HOST}:5000"

# 健康检查
curl -s ${QWEN_URL}/health | python3 -m json.tool

# 列出所有 checkpoints
curl -s ${QWEN_URL}/list_checkpoints | python3 -m json.tool

# 切换 checkpoint
curl -X POST ${QWEN_URL}/switch_checkpoint \
  -H "Content-Type: application/json" \
  -d '{
    "checkpoint_path": "ddp/qwen2.5-vl-regression_lora-True_ddp_regression/checkpoint-12500",
    "algorithm": "DDP",
    "head_type": "dpt",
    "num_params": 6
  }' | python3 -m json.tool

# 推理 (需要 base64 编码的图片)
image_base64=$(base64 -w 0 /path/to/costmap.png)
curl -X POST ${QWEN_URL}/infer \
  -H "Content-Type: application/json" \
  -d "{
    \"image_base64\": \"${image_base64}\",
    \"linear_vel\": 0.5,
    \"angular_vel\": 0.0,
    \"algorithm\": \"DDP\"
  }" | python3 -m json.tool
```

### 3. 在 ROS 中使用（待集成）

```python
import requests
import base64
from cv_bridge import CvBridge
import rospy

class QwenParameterPredictor:
    def __init__(self, qwen_host='gpu017', qwen_port=5000):
        self.url = f"http://{qwen_host}:{qwen_port}"
        self.bridge = CvBridge()

    def predict_parameters(self, costmap_image, linear_vel, angular_vel, algorithm='DDP'):
        """从 costmap 图像预测导航参数"""
        # 转换为 base64
        _, buffer = cv2.imencode('.png', costmap_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        # 调用推理 API
        response = requests.post(
            f"{self.url}/infer",
            json={
                'image_base64': image_base64,
                'linear_vel': linear_vel,
                'angular_vel': angular_vel,
                'algorithm': algorithm
            }
        )

        result = response.json()
        if result['success']:
            return result['parameters']
        else:
            rospy.logerr(f"Qwen inference failed: {result.get('error')}")
            return None
```

## 📊 Checkpoint 结构

### 目录布局
```
/scratch/ylu22/appvlm_ws/src/ros_jackal/model/
├── ddp/
│   └── qwen2.5-vl-regression_lora-True_ddp_regression/
│       ├── checkpoint-2500/
│       ├── checkpoint-5000/
│       ├── checkpoint-7500/
│       ├── checkpoint-10000/
│       ├── checkpoint-12500/
│       └── checkpoint-15000/
├── dwa/
│   └── qwen2.5-vl-regression_lora-True_dwa_regression/
│       └── checkpoint-{2500,5000,7500,10000,12500,15000}/
├── teb/
│   └── qwen2.5-vl-regression_lora-True_teb_regression/
│       └── checkpoint-{2500,5000,7500,10000,12500,15000}/
└── mppi/
    └── qwen2.5-vl-regression_lora-True_mppi_regression/
        └── checkpoint-{2500,5000,7500,10000,12500,15000}/
```

### 每个 checkpoint 包含
```
checkpoint-12500/
├── adapter_model.safetensors     # LoRA 权重
├── adapter_config.json           # LoRA 配置
├── regression_head/
│   ├── pytorch_model.bin         # DPT head 权重
│   └── config.json               # DPT head 配置
├── normalization/
│   ├── param_mean.npy            # 参数归一化均值
│   └── param_std.npy             # 参数归一化标准差
├── zero_to_fp32.py               # DeepSpeed 恢复脚本
└── global_step12500/             # DeepSpeed 分布式权重（恢复用）
```

## 🔧 参数配置

### 各规划器的参数数量
- **DDP**: 6 个参数
- **DWA**: 7 个参数
- **TEB**: 7 个参数
- **MPPI**: 6 个参数

### 推理输出示例
```json
{
  "success": true,
  "parameters": {
    "max_vel_x": 2.15,
    "max_vel_theta": 3.42,
    "gamma": 850.67,
    "lambda": 0.025,
    "v_angular_weight": 0.015,
    "tracking_weight": 0.11
  },
  "checkpoint": "ddp/qwen2.5-vl-regression_lora-True_ddp_regression/checkpoint-12500",
  "algorithm": "DDP",
  "inference_time": 0.234
}
```

## 🛠️ 故障排除

### 服务无法启动
```bash
# 检查 SLURM 日志
cat qwen_dynamic_*.out

# 常见问题:
# 1. GPU 内存不足 - 检查其他进程: nvidia-smi
# 2. 模型路径错误 - 检查 hopper_qwen_dynamic.slurm 中的路径
# 3. Conda 环境问题 - 确保 lmms-finetune-qwen 环境存在
```

### 服务启动但无法连接
```bash
# 检查服务是否真的在运行
ssh gpu017  # 或你的节点
curl localhost:5000/health

# 如果本地可以访问但远程不行，检查防火墙设置
```

### Checkpoint 加载失败
```bash
# 检查 checkpoint 是否有空 tensor
cd /home/yuanjielu/robot_navigation/noetic/appvlm_ws/src/ros_jackal/script_HPC
python3 recover_checkpoints.py --checkpoint /path/to/checkpoint --dry_run

# 如果需要恢复
python3 recover_checkpoints.py --checkpoint /path/to/checkpoint
```

## 📝 性能指标

### 典型延迟
- Checkpoint 切换: ~0.9 秒
- 单次推理: ~0.2-0.3 秒 (取决于图片大小)
- 总切换+推理: ~1.2 秒

### 内存占用
- 基础模型 (Qwen2.5-VL-7B 4-bit): ~5GB
- LoRA 权重: ~150MB
- DPT head: ~50MB
- 总计: ~5.2GB GPU 内存

## 🔍 调试技巧

### 查看详细日志
```bash
# 服务启动时添加 debug 模式
# 修改 qwen_server_dynamic.py:
#   uvicorn.run(app, host="0.0.0.0", port=5000, log_level="debug")

# 或在推理时打印 hidden states
curl -X POST ${QWEN_URL}/infer \
  -H "Content-Type: application/json" \
  -d '{"image_base64": "...", "debug": true}' | python3 -m json.tool
```

### 验证预测是否合理
```bash
# 比较不同 checkpoint 的输出
bash test_checkpoint_inference.sh

# 预期结果:
# - checkpoint-2500: 参数可能还不够优化
# - checkpoint-12500: 应该接近最优参数
# - 不同 checkpoint 应该有差异但不会太大
```

## 📚 相关文档
- [CLAUDE.md](../CLAUDE.md) - 项目总览
- [AGENTS.md](../AGENTS.md) - 开发规范
- [qwen_server_dynamic.py](qwen_server_dynamic.py) - 服务实现
- [recover_checkpoints.py](recover_checkpoints.py) - 恢复工具

## ⚠️ 重要注意事项
1. 不要在主节点运行推理 - 必须在 GPU 节点上运行
2. 每次推理前确认服务节点名（可能会变）
3. 测试图片需要是 costmap 格式（RGB，包含机器人、障碍物、路径）
4. 如果遇到空 tensor 错误，使用 `recover_checkpoints.py` 恢复
5. 服务重启会重置到初始 checkpoint，需要重新切换
