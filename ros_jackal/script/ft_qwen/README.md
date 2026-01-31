# FTRL - VLM+DPT强化学习微调

基于预训练VLM+DPT进行TD3强化学习微调的完整实现

## 📂 项目结构

```
ros_jackal/
├── rlft/                           # FTRL核心代码
│   ├── vlm_net.py                  # VLM+DPT网络定义
│   ├── rl.py                       # TD3算法实现
│   ├── train.py                    # 训练脚本
│   └── README.md                   # 技术文档
│
├── script/ft_qwen/                 # 配置和启动脚本
│   ├── configs/
│   │   └── ftrl_vlm_dwa.yaml      # DWA配置文件
│   ├── run_ftrl.sh                # 启动脚本
│   └── README.md                  # 本文档
│
├── buffer/ftrl_vlm/               # Replay buffer存储
└── logging/ftrl_vlm/              # TensorBoard日志
```

## 🚀 快速开始

### 前置条件

**监督学习checkpoint已完成**：
- 路径示例: `/path/to/lmms-finetune-qwen/output/checkpoint-2500`
- 包含: LoRA adapters + DPT Head + (可选) History Encoder

### 阶段1: 数据收集

**Step 1: 启动 Qwen 推理服务**

```bash
# 终端1: 启动 qwen_server.py
cd /path/to/ros_jackal
python script/qwen/qwen_server.py \
  --base_model /path/to/Qwen2.5-VL-7B \
  --lora_path /path/to/checkpoint-2500 \
  --algorithm DWA \
  --port 5000
```

**Step 2: 启动数据收集脚本**

```bash
# 终端2: 启动数据收集
cd /path/to/ros_jackal
python script/ft_qwen/evaluate_ftrl_single.py \
  --id 0 \
  --server_url http://localhost:5000 \
  --policy_name dwa_ftrl \
  --buffer_path ./buffer/
```

**数据保存位置**: `buffer/dwa_ftrl/actor_0/`
- `traj_*.pickle`: 轨迹数据 (obs, action, reward, done)
- `*.png`: Costmap 图像
- `trajectory_results.txt`: 性能统计

### 阶段2: RL 训练

**修改配置文件** `configs/ftrl_vlm_dwa.yaml`:

```yaml
training_config:
  vlm_checkpoint_path: "/path/to/checkpoint-2500"  # 修改为你的路径
```

**启动训练**:

```bash
cd /path/to/ros_jackal
./script/ft_qwen/run_ftrl.sh
```

### 阶段3: 监控进度

```bash
tensorboard --logdir logging/ftrl_vlm
# 浏览器访问: http://localhost:6006
```

## 📊 监控指标

### 关键指标（TensorBoard）

- `train/Test_nav_metric`: 测试导航性能（↑越高越好）
- `train/Success_rate`: 成功率（%）
- `train/Test_length`: 平均轨迹长度（↓越短越好）
- `train/Actor_loss`: Actor损失
- `train/Critic_loss`: Critic损失

### 终端输出

训练过程会在终端输出实时统计：

```
Episode_reward: 0.85
Episode_nav_metric: 0.92
Success_rate: 78.5%
Actor_loss: -0.34
Critic_loss: 0.012
fps: 245.3
```

## ⚙️ 配置参数详解

### 必须修改的参数

```yaml
training_config:
  # VLM+DPT checkpoint路径（必须修改！）
  vlm_checkpoint_path: "/path/to/checkpoint"
```

### 推荐的超参数

```yaml
training_config:
  # 冻结策略（推荐配置）
  freeze_vlm_actor: true     # VLM太大，必须冻结
  freeze_dpt_actor: false    # DPT微调（FTRL核心）
  freeze_dpt_critic: true    # Critic DPT冻结省显存

  # 学习率（VLM微调需要小lr）
  actor_lr: 1.0e-5           # 不要超过1e-4
  critic_lr: 3.0e-4

  # 训练步数
  training_args:
    max_step: 1000000        # 总步数
    collect_per_step: 1000   # 每次收集
    update_per_step: 50      # 每次更新
```

### 性能调优参数

**显存优化**:
```yaml
training_args:
  batch_size: 128           # 默认256，显存不足改为128
```

**训练稳定性**:
```yaml
training_config:
  exploration_noise_start: 0.02  # 减小探索噪声
  actor_lr: 5.0e-6               # 减小学习率
  pre_collect: 50000             # 增加预收集
```

## 🎯 实验指南

### 基础实验（验证可行性）

1. **先跑1小时测试**
   ```yaml
   training_args:
     max_step: 50000  # 约1小时
   ```

2. **检查指标**
   - `Test_nav_metric` 是否提升？
   - `Success_rate` 是否>监督学习？

3. **如果指标下降**
   - 减小 `actor_lr: 5e-6`
   - 减小 `exploration_noise_start: 0.02`

### 完整实验（发论文）

1. **长时间训练**
   ```yaml
   training_args:
     max_step: 1000000  # ~24小时
   ```

2. **保存checkpoint**
   - 最佳模型自动保存在 `logging/ftrl_vlm/.../policy_step_XXX_actor`

3. **评估对比**
   ```bash
   # 评估FTRL模型
   python eval_ftrl.py --checkpoint logging/.../policy_step_XXX_actor

   # 对比监督学习
   python eval_supervised.py --checkpoint /path/to/supervised/checkpoint
   ```

### 消融实验（分析贡献）

**实验1: VLM冻结 vs 微调**
```yaml
# Exp A: VLM冻结（baseline）
freeze_vlm_actor: true

# Exp B: VLM微调
freeze_vlm_actor: false
actor_lr: 5.0e-6  # 更小的lr
```

**实验2: DPT冻结 vs 微调**
```yaml
# Exp A: DPT冻结（不做FTRL）
freeze_dpt_actor: true

# Exp B: DPT微调（FTRL）
freeze_dpt_actor: false
```

**实验3: 不同学习率**
```yaml
actor_lr: [1e-6, 5e-6, 1e-5, 5e-5]
```

## 🐛 常见问题

### Q1: 训练启动失败

**错误**: `ModuleNotFoundError: No module named 'qwen2_5_vl_dpt_regression'`

**解决**:
```bash
# 检查路径
export PYTHONPATH="/path/to/qwen_dpt/lmms-finetune-qwen/models:$PYTHONPATH"
```

---

**错误**: `FileNotFoundError: checkpoint-2500 not found`

**解决**:
```yaml
# 修改配置文件中的路径为绝对路径
vlm_checkpoint_path: "/absolute/path/to/checkpoint-2500"
```

### Q2: 显存溢出

**错误**: `CUDA out of memory`

**解决**:
```yaml
# 方案1: 减小batch size
training_args:
  batch_size: 64  # 默认256

# 方案2: 冻结更多参数
freeze_dpt_actor: true
freeze_dpt_critic: true

# 方案3: 使用4-bit量化（修改vlm_net.py）
```

### Q3: 性能不提升

**现象**: `Test_nav_metric` 不增长或下降

**诊断**:
1. 检查 `Actor_loss`: 如果是NaN → 学习率太大
2. 检查 `Success_rate`: 如果<50% → 探索不够
3. 检查 `Exploration_noise`: 如果太小 → 无法探索

**解决**:
```yaml
# 如果loss是NaN
actor_lr: 5.0e-6  # 减小10倍

# 如果成功率太低
exploration_noise_start: 0.1  # 增加探索
pre_collect: 50000            # 多收集经验

# 如果完全不学习
freeze_dpt_actor: false  # 确保DPT可训练
```

### Q4: 训练太慢

**现象**: fps < 50

**原因**: VLM推理慢

**解决**:
```yaml
# 方案1: 减小更新频率
training_args:
  update_per_step: 20  # 默认50

# 方案2: 增加收集频率
training_args:
  collect_per_step: 2000  # 默认1000
```

## 📈 预期结果

### 监督学习baseline
- MAE: 0.05-0.1
- 成功率: 70-80%
- 推理速度: 100-500ms

### FTRL目标
- 成功率: >80%（超过监督学习）
- 样本效率: <1M steps
- 训练时间: ~24小时

### APPLR对比
- APPLR样本: 5M steps
- APPLR时间: 6小时（500 CPU并行）
- FTRL优势: 预训练加持，样本效率高

## 📝 发论文Checklist

- [ ] 对比监督学习性能
- [ ] 对比APPLR样本效率
- [ ] 消融实验（VLM冻结/微调，DPT冻结/微调）
- [ ] 泛化实验（测试环境 vs 训练环境）
- [ ] 可视化（轨迹、attention map）
- [ ] 性能分析（推理速度、显存占用）

## 🧠 核心原理：RL如何更新VLM+DPT

### 问题：没有ground truth，如何反向传播？

**监督学习** (已完成):
```python
predicted_params = VLM_DPT(image)  # [7个参数]
loss = MSE(predicted_params, ground_truth)  # 有明确目标
loss.backward()  # 梯度清晰
```

**强化学习** (FTRL):
```python
predicted_params = VLM_DPT(image)  # [7个参数]
reward = env.step(predicted_params)  # 只知道好坏，无ground truth
# 问题：如何计算梯度？
```

### 答案：通过Critic作为"learned ground truth"

#### 完整梯度流

```python
# ========== 前向传播 ==========
# 1. Actor (VLM+DPT) 生成action
image → VLM → DPT → FC → action [7个参数]

# 2. Critic评估action的价值
Q = Critic(image, action)  # Q值 = "这个action的长期价值"

# ========== 反向传播 ==========
# 3. Actor Loss
actor_loss = -Q  # 负号：想最大化Q值

# 4. 梯度反向传播到VLM+DPT
actor_loss.backward()
    ↓ PyTorch自动微分
∂(-Q)/∂action → action的梯度 (Critic告诉我们"如何调整action")
    ↓ 通过FC层
∂action/∂features → DPT features的梯度
    ↓ 通过DPT Head
∂features/∂hidden_states → VLM的梯度
    ↓
VLM+DPT参数更新！
```

#### 数学表达

```
梯度 = ∂(-Q)/∂θ_VLM+DPT
     = -∂Q(s,a)/∂a × ∂a/∂θ

其中:
- ∂Q/∂a: Critic说"调整action的方向"
- ∂a/∂θ: Actor说"调整参数能产生那个action"
- θ: VLM+DPT的所有可训练参数
```

### Critic如何学习Q值？

```python
# Critic通过Bellman方程从reward学习
target_Q = reward + γ × Q_target(next_state, next_action)
critic_loss = MSE(Q(state, action), target_Q)
critic_loss.backward()  # 更新Critic

# Critic的作用:
# 1. 密集化reward (环境只在结束给reward，Critic能估计每步价值)
# 2. 平滑梯度 (直接用reward很不稳定)
# 3. 长期规划 (考虑未来reward，通过γ折扣)
```

### 对比：监督学习 vs RL

| 方面 | 监督学习 | 强化学习 (FTRL) |
|------|---------|----------------|
| **目标** | 模仿专家数据 | 最大化环境reward |
| **Loss来源** | `MSE(pred, ground_truth)` | `-Q(s, Actor(s))` |
| **梯度信号** | 明确且稳定 | 通过Critic间接获得 |
| **优化目标** | `min MSE` | `max E[Q(s,π(s))]` |
| **数据需求** | 大量标注数据 | 环境交互 |
| **潜力** | 受限于标注质量 | 可能超越人类 |

### 为什么Critic不保存？

```python
# APPLR和FTRL都只保存Actor
def save(self, dir, filename):
    with open(join(dir, filename + "_actor"), "wb") as f:
        pickle.dump(self.actor.state_dict(), f)
    # 不保存Critic！

# 原因:
# 1. Critic只在训练时用 (计算Q值指导Actor)
# 2. 推理/数据收集时不需要Critic
# 3. 下次训练可以重新创建Critic (或加载上次的继续训练)
```

### 完整的FTRL流程

```
阶段1: 监督学习 (已完成)
├─ 数据: (costmap, optimal_params) pairs
├─ 训练: MSE(VLM_DPT(image), optimal_params)
└─ 结果: 基础的参数预测能力

阶段2: 数据收集 (Python 3.8 + ROS)
├─ qwen_server.py (Python 3.10): 提供VLM+DPT推理服务
│   └─ 加载监督学习的checkpoint (阶段1的输出)
├─ evaluate_ftrl_single.py:
│   ├─ result = qwen_client.infer_from_server(image_path)
│   ├─ action = qwen_client.get_parameters_array(result)
│   ├─ env.step(action) → reward
│   └─ save (image, action, reward) → buffer/
└─ 结果: (obs, action, reward, next_obs, done) 轨迹数据

阶段3: RL训练 (Python 3.10)
├─ rlft/train.py:
│   ├─ 读取buffer中的所有轨迹
│   ├─ Critic学习Q(s,a) ← 从reward
│   ├─ Actor优化: max Q(s, Actor(s))
│   └─ 保存更新后的Actor
└─ 结果: 超越监督学习的性能

阶段4: 循环更新
├─ 重启ftrl_server，加载最新Actor
└─ 继续收集数据 → 训练 → 更新...
```

### 环境隔离：为什么需要HTTP？

```
问题: VLM+DPT需要Python 3.10，但ROS只支持Python 3.8

解决:
┌─────────────────────────────────┐
│ evaluate_ftrl_single.py         │
│ (Python 3.8 + ROS)              │
│ ├─ Gym环境                      │
│ ├─ Gazebo仿真                   │
│ └─ HTTP调用 →                   │
└─────────────────────────────────┘
              ↓ HTTP
┌─────────────────────────────────┐
│ qwen_server.py                  │
│ (Python 3.10)                   │
│ ├─ VLM+DPT (监督学习checkpoint) │
│ └─ 返回7个参数                  │
└─────────────────────────────────┘

注意:
- 数据收集阶段使用 script/qwen/qwen_server.py
- ftrl_server.py 是为了将来训练后加载新Actor（可选）
- 当前实现：直接复用成熟的qwen_server.py
```

### 好轨迹 vs 坏轨迹

**TD3自动处理**：
```python
# 好轨迹: reward高 → Q值高 → Actor朝这个方向更新
trajectory_good = [(s, a, +10.0, ...), ...]
→ VLM+DPT学习"产生这些action"

# 坏轨迹: reward低 → Q值低 → Actor远离这个方向
trajectory_bad = [(s, a, -10.0, ...), ...]
→ VLM+DPT学习"避免这些action"

# 不需要手动过滤！Critic会自动给坏轨迹低Q值
```

**可选增强**：
- Reward Shaping: 精细化奖励信号
- Prioritized Replay: 优先学习TD error大的样本
- 但默认TD3机制已经足够好！

## 🔗 相关文档

- [RLFT技术文档](../../rlft/README.md) - 详细实现说明
- [APPLR论文](../../../applr.pdf) - Baseline方法
- [CLAUDE.md](../../../CLAUDE.md) - 项目总览

---

**Questions? 联系项目维护者**
