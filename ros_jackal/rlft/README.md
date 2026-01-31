# RLFT (Reinforcement Learning Fine-Tuning)

VLM+DPT 与 TD3 结合的强化学习微调实现

## 🎯 架构总览

### TD3 的四个网络

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TD3 算法包含 4 个网络                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────┐          ┌─────────────────┐                              │
│   │   Actor     │          │   Actor_target  │                              │
│   │  (主网络)    │  ──────► │   (目标网络)     │    soft update (τ=0.005)    │
│   └─────────────┘          └─────────────────┘                              │
│         │                                                                   │
│         │ 选择动作                                                           │
│         ▼                                                                   │
│   ┌─────────────┐          ┌─────────────────┐                              │
│   │   Critic    │          │  Critic_target  │                              │
│   │  (主网络)    │  ──────► │   (目标网络)     │    soft update (τ=0.005)    │
│   └─────────────┘          └─────────────────┘                              │
│         │                          │                                        │
│         │ 评估 Q(s,a)              │ 计算 target Q                           │
│         ▼                          ▼                                        │
│   ┌─────────────────────────────────────────┐                               │
│   │          Bellman 更新                    │                               │
│   │  Q(s,a) ← r + γ * target_Q(s', a')      │                               │
│   └─────────────────────────────────────────┘                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### VLM+DPT 组件共享关系

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         组件共享关系                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                    Actor 和 Critic 共享同一个 feature_extractor              │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    feature_extractor (共享)                          │   │
│   │  ┌──────────────────────────────────────────────────────────────┐   │   │
│   │  │  VLM (Qwen2.5-VL-3B)  │  冻结，2B参数                         │   │   │
│   │  ├──────────────────────────────────────────────────────────────┤   │   │
│   │  │  DPT Head             │  可训练，3.89M参数                    │   │   │
│   │  ├──────────────────────────────────────────────────────────────┤   │   │
│   │  │  History Encoder      │  可训练，1.68M参数                    │   │   │
│   │  └──────────────────────────────────────────────────────────────┘   │   │
│   │                              ↓                                       │   │
│   │                         256-d 特征                                   │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                    ↓                              ↓                         │
│            ┌──────────────┐              ┌──────────────┐                   │
│            │    Actor     │              │    Critic    │                   │
│            │  FC → 动作    │              │  Q-head → Q值│                   │
│            └──────────────┘              └──────────────┘                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Target 网络的独立组件

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   主网络 (训练)                              Target 网络 (评估)              │
│   ═══════════════                           ═══════════════════             │
│                                                                             │
│   feature_extractor                                                         │
│   ┌─────────────────┐                       ┌─────────────────┐             │
│   │ VLM base (冻结)  │ ◄═══════共享═══════► │ VLM base (冻结)  │             │
│   │ 4-bit, ~2B      │                       │                 │             │
│   └─────────────────┘                       └─────────────────┘             │
│                                                                             │
│   ┌─────────────────┐      soft update      ┌─────────────────┐             │
│   │ LoRA (可训练)    │ ─────────────────────►│ LoRA_target     │             │
│   │ ~330M           │      τ=0.005          │ (state_dict)    │             │
│   └─────────────────┘                       └─────────────────┘             │
│                                              (独立副本)                      │
│                                                                             │
│   ┌─────────────────┐      soft update      ┌─────────────────┐             │
│   │ DPT (可训练)    │ ─────────────────────►│ DPT_target      │             │
│   │ ~3.89M          │      τ=0.005          │                 │             │
│   └─────────────────┘                       └─────────────────┘             │
│                                              (独立副本)                      │
│                                                                             │
│   ┌─────────────────┐      soft update      ┌─────────────────┐             │
│   │ History (可训练) │ ─────────────────────►│ History_target  │             │
│   │ ~1.68M          │      τ=0.005          │                 │             │
│   └─────────────────┘                       └─────────────────┘             │
│                                              (独立副本)                      │
│                                                                             │
│   ┌─────────────────┐      soft update      ┌─────────────────┐             │
│   │ Actor.fc        │ ─────────────────────►│ Actor_target.fc │             │
│   └─────────────────┘      τ=0.005          └─────────────────┘             │
│                                                                             │
│   ┌─────────────────┐      soft update      ┌─────────────────┐             │
│   │ Critic Q-heads  │ ─────────────────────►│ Critic_target   │             │
│   └─────────────────┘      τ=0.005          │ Q-heads         │             │
│                                             └─────────────────┘             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 训练数据流（一次迭代）

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          TD3 训练一次迭代                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. 从 Buffer 采样: (state, action, reward, next_state, done)               │
│                                                                             │
│  ════════════════════════════════════════════════════════════════════════   │
│  2. 计算 Target Q (用 target 网络，不更新)                                   │
│  ════════════════════════════════════════════════════════════════════════   │
│                                                                             │
│     next_state (costmap)                                                    │
│          │                                                                  │
│          ▼                                                                  │
│     ┌─────────────────────────────────────┐                                 │
│     │ VLM (共享) → DPT_target → 256-d     │  ← 使用 target 版本!            │
│     └─────────────────────────────────────┘                                 │
│          │                                                                  │
│          ▼                                                                  │
│     ┌─────────────────┐                                                     │
│     │ Actor_target.fc │ → next_action                                       │
│     └─────────────────┘                                                     │
│          │                                                                  │
│          ▼                                                                  │
│     ┌─────────────────────────────────────┐                                 │
│     │ Critic_target(next_state, next_action) │ → target_Q                   │
│     └─────────────────────────────────────┘                                 │
│          │                                                                  │
│          ▼                                                                  │
│     target = reward + γ * target_Q                                          │
│                                                                             │
│  ════════════════════════════════════════════════════════════════════════   │
│  3. 更新 Critic (最小化 TD error)                                           │
│  ════════════════════════════════════════════════════════════════════════   │
│                                                                             │
│     state (costmap)                                                         │
│          │                                                                  │
│          ▼                                                                  │
│     ┌─────────────────────────────────────┐                                 │
│     │ VLM → DPT → 256-d                   │  ← 使用主网络                    │
│     └─────────────────────────────────────┘                                 │
│          │                                                                  │
│          ▼                                                                  │
│     ┌─────────────────────────────────────┐                                 │
│     │ Critic(state, action)               │ → current_Q                     │
│     └─────────────────────────────────────┘                                 │
│          │                                                                  │
│          ▼                                                                  │
│     critic_loss = MSE(current_Q, target)                                    │
│     critic_loss.backward()  → 更新 Critic Q-heads                           │
│                               (DPT 不更新，因为 detach)                      │
│                                                                             │
│  ════════════════════════════════════════════════════════════════════════   │
│  4. 更新 Actor (最大化 Q 值，每2步一次)                                      │
│  ════════════════════════════════════════════════════════════════════════   │
│                                                                             │
│     state (costmap)                                                         │
│          │                                                                  │
│          ▼                                                                  │
│     ┌─────────────────────────────────────┐                                 │
│     │ VLM → DPT → 256-d                   │  ← 梯度会传到 DPT!              │
│     └─────────────────────────────────────┘                                 │
│          │                                                                  │
│          ▼                                                                  │
│     ┌─────────────────┐                                                     │
│     │ Actor.fc        │ → predicted_action                                  │
│     └─────────────────┘                                                     │
│          │                                                                  │
│          ▼                                                                  │
│     actor_loss = -Critic.Q1(state, predicted_action)                        │
│     actor_loss.backward()  → 更新 Actor.fc + DPT + History                  │
│                                                                             │
│  ════════════════════════════════════════════════════════════════════════   │
│  5. Soft Update (缓慢更新 target)                                           │
│  ════════════════════════════════════════════════════════════════════════   │
│                                                                             │
│     DPT_target      = 0.995 * DPT_target      + 0.005 * DPT                 │
│     History_target  = 0.995 * History_target  + 0.005 * History             │
│     Actor_target.fc = 0.995 * Actor_target.fc + 0.005 * Actor.fc            │
│     Critic_target   = 0.995 * Critic_target   + 0.005 * Critic              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 为什么需要 Target 网络？

```
问题: 如果直接用主网络计算 target Q
─────────────────────────────────────────────────────────────────
target = r + γ * Critic(next_state, Actor(next_state))
                    ↑                    ↑
              正在被更新              正在被更新

→ target 每步都剧烈变化 → 训练不稳定 → 发散


解决: Target 网络缓慢追踪
─────────────────────────────────────────────────────────────────
target = r + γ * Critic_target(next_state, Actor_target(next_state))
                    ↑                           ↑
              每步只变化 0.5%               每步只变化 0.5%

→ target 稳定变化 → 训练稳定 → 收敛
```

## 📁 目录结构

```
rlft/
├── __init__.py          # 包初始化
├── vlm_net.py           # VLM+DPT网络定义
├── rl.py                # TD3算法实现 (复用自td3/)
├── train.py             # FTRL训练脚本
└── README.md            # 本文档
```

## 🔧 核心组件

### 1. VLM_DPT_FeatureExtractor
从监督学习checkpoint加载VLM+DPT作为特征提取器

**关键设计**:
- 加载预训练的Qwen2.5-VL + LoRA
- 加载预训练的DPT Head
- 支持选择性冻结（VLM/DPT独立控制）
- 只提取DPT的中间特征（256-d），不使用回归层

### 2. VLM_DPT_Actor
TD3的Actor网络

**架构**:
```
Costmap Image → VLM+DPT特征提取 → FC层 → 7个导航参数
```

**训练策略**:
- VLM: 冻结（太大，更新慢）
- DPT Head: 可训练（FTRL微调）
- FC层: 可训练

### 3. VLM_DPT_Critic
TD3的Twin Critic网络

**架构**:
```
Costmap Image → VLM+DPT特征提取 → [256-d]
                                    ↓
Action (7个参数) → MLP编码 → [64-d] ↓
                                    ↓
            Concat → Fusion MLP → Q值
```

**训练策略**:
- VLM+DPT: 全部冻结（节省显存）
- Action编码器: 可训练
- Fusion MLP: 可训练

## 🚀 快速开始

### 1. 准备checkpoint

确保你有监督学习训练好的VLM+DPT checkpoint：

```bash
checkpoint-2500/
├── adapter_config.json      # LoRA配置
├── adapter_model.bin         # LoRA权重
└── regression_head/
    └── pytorch_model.bin     # DPT Head权重
```

### 2. 修改配置文件

编辑 `script/ft_qwen/configs/ftrl_vlm_dwa.yaml`:

```yaml
training_config:
  # 修改为你的checkpoint路径
  vlm_checkpoint_path: "/path/to/your/checkpoint-2500"
```

### 3. 启动训练

**方式1: 使用启动脚本（推荐）**
```bash
cd /path/to/ros_jackal
./script/ft_qwen/run_ftrl.sh
```

**方式2: 直接运行Python**
```bash
cd /path/to/ros_jackal
python rlft/train.py \
  --config_path script/ft_qwen/configs/ \
  --config_file ftrl_vlm_dwa \
  --buffer_path buffer/ftrl_vlm \
  --logging_path logging/ftrl_vlm
```

### 4. 监控训练

使用TensorBoard查看训练进度：

```bash
tensorboard --logdir logging/ftrl_vlm
```

**关键指标**:
- `train/Test_nav_metric`: 测试集导航性能（越高越好）
- `train/Success_rate`: 成功率
- `train/Actor_loss`: Actor损失
- `train/Critic_loss`: Critic损失

## ⚙️ 配置说明

### VLM+DPT配置

```yaml
training_config:
  # Checkpoint路径
  vlm_checkpoint_path: "/path/to/checkpoint"

  # 冻结策略
  freeze_vlm_actor: true     # Actor的VLM冻结（推荐）
  freeze_dpt_actor: false    # Actor的DPT可训练（FTRL）
  freeze_dpt_critic: true    # Critic的DPT冻结（省显存）
```

### 学习率配置

```yaml
training_config:
  # VLM微调需要更小的学习率
  actor_lr: 1.0e-5    # 比APPLR小10倍
  critic_lr: 3.0e-4
```

### TD3超参数

```yaml
training_config:
  policy_args:
    gamma: 0.99              # 折扣因子
    tau: 0.005               # 软更新系数
    policy_noise: 0.2        # 目标策略平滑噪声
    noise_clip: 0.5          # 噪声裁剪
    n_step: 4                # N-step return
    update_actor_freq: 2     # Actor延迟更新
    exploration_noise: 0.1   # 探索噪声
```

### 训练参数

```yaml
training_config:
  training_args:
    max_step: 1000000         # 总训练步数
    collect_per_step: 1000    # 每次收集步数
    update_per_step: 50       # 每次更新次数
    batch_size: 256           # 批大小
```

## 🎯 与APPLR的区别

| 方面 | APPLR (Baseline) | RLFT (本实现) |
|------|------------------|---------------|
| 特征提取 | CNN (3层Conv) | VLM+DPT (预训练) |
| 初始化 | 随机初始化 | 监督学习预训练 |
| 训练数据需求 | 5M samples | 预期<1M samples |
| Actor参数量 | ~1M | ~8B (大部分冻结) |
| 样本效率 | 低 | 高（预训练加持） |
| 训练时间 | 6小时 (500 CPU) | 待测试 |

## 💡 关键技术点

### 1. Critic为什么不能直接复用Actor的checkpoint？

**问题**: Actor和Critic的输入空间不同
- Actor: `state` → `action`
- Critic: `(state, action)` → `Q值`

**解决**: Critic使用VLM+DPT提取state特征，额外用MLP编码action，然后fusion

### 2. 为什么要冻结VLM？

**原因**:
- VLM有8B参数，RL更新太慢
- VLM的视觉理解能力已经很强，不需要继续训练
- 节省显存和计算

**FTRL策略**: 只微调DPT Head（256-d特征空间的回归）

### 3. 为什么Critic的DPT也冻结？

**原因**:
- Critic不需要直接预测参数，只需要评估好坏
- 冻结DPT可以节省大量显存（双Q网络需要2个VLM）
- Critic的fusion层已经足够学习Q值

## ⚠️ 已知问题与架构分析 (2026-01-29)

### 问题总览

| 优先级 | 问题 | 影响 | 状态 |
|-------|------|------|------|
| 🔴 P0 | Target Network 共享 Feature Extractor | 打破 TD3 核心机制 | ✅ 已修复 (LoRA+DPT+History) |
| 🟡 P1 | 冻结策略过于激进 | 可训练参数太少 (~2K) | ✅ 已调整 (~335M 可训练) |
| 🟡 P1 | Critic detach 阻断梯度 | VLM 特征无法被优化 | 设计权衡 |
| 🟢 P2 | Action 归一化可能不一致 | 分布偏移风险 | 待验证 |

---

### 🔴 问题1: Target Network 共享 Feature Extractor（严重）

**位置**: `rl.py:125-202`

**现象**:
```python
def _create_actor_target(self, actor):
    actor_target = VLM_DPT_Actor(
        feature_extractor=actor.feature_extractor,  # ← 共享同一个对象！
        ...
    )
```

**TD3 设计 vs 当前实现**:

```
标准 TD3:
─────────────────────────────────────────────────
Actor (全部权重)  ──soft update (τ=0.005)──►  Actor_target
   ↓ 训练                                        ↓ 缓慢追踪
权重变化 0.5%                               权重变化 0.5% × τ

当前 RLFT:
─────────────────────────────────────────────────
Actor.feature_extractor ◄═══共享═══► Actor_target.feature_extractor
        (~2B 参数)                        (同一个对象！)

Actor.fc ──────soft update──────────► Actor_target.fc
   (~2K 参数)                             (正常追踪)
```

**问题本质**:
- TD3 的 target network 应该**缓慢追踪**主网络（每步只更新 0.5%）
- 但 VLM+DPT 部分是**共享**的，主网络更新时 target **立即**看到相同权重
- 只有 FC 层有 soft update，占总参数的 0.0001%

**影响**:
- Q-value 估计不稳定（target 剧烈变化）
- Bellman 更新目标不稳定
- 可能导致训练震荡或发散

**为什么这样设计**:
```python
# rl.py:87-88 注释
# 🔧 关键修复: 不能 deepcopy 4-bit 量化的 VLM
# actor_target 共享 feature_extractor，只复制 FC 层
```
4-bit 量化的 VLM 无法用 `copy.deepcopy()`，会破坏量化状态。这是工程限制。

**严重程度取决于冻结策略**:

| 配置 | 影响 |
|-----|------|
| `freeze_vlm=True, freeze_dpt=True` | **较小** - 都冻结，共享无影响 |
| `freeze_vlm=True, freeze_dpt=False` | **中等** - DPT 更新时 target 立即变化 |
| `freeze_vlm=False` | **严重** - VLM 更新时 target 剧烈变化 |

**修复方案** (2026-01-29 已实现):

```python
# rl.py: _create_actor_target()
# 为可训练的 LoRA/DPT/History 创建独立副本

# LoRA: 使用 PEFT API 获取 state_dict
if lora_trainable:
    from peft import get_peft_model_state_dict
    self.actor_target_lora_state = {
        k: v.clone().detach().cpu()
        for k, v in get_peft_model_state_dict(base_model).items()
    }

# DPT/History: 直接 deepcopy
if dpt_trainable:
    self.actor_target_dpt = copy.deepcopy(actor.feature_extractor.dpt_head)

if history_trainable:
    self.actor_target_history = copy.deepcopy(actor.feature_extractor.history_encoder)

# rl.py: _soft_update_actor()
# 对 LoRA/DPT/History target 进行 soft update

# rl.py: _swap_to_target_components(), _restore_original_components()
# 临时替换 feature_extractor 的 LoRA/DPT/History 为 target 版本
# 使用 PEFT 的 set_peft_model_state_dict() 切换 LoRA 权重
```

---

### 🟡 问题2: 冻结策略过于激进

**已调整配置** (`ftrl_vlm_ddp.yaml`, 2026-01-29):
```yaml
freeze_vlm_actor: true       # VLM 冻结（保护场景理解）
freeze_dpt_actor: false      # DPT 可训练 ✅
freeze_history_actor: false  # History 可训练 ✅
```

**调整后**:

| 组件 | 参数量 | 可训练 |
|-----|--------|-------|
| VLM (Qwen2.5-VL-3B) | ~2B | ❌ 冻结 |
| DPT Head | ~3.89M | ✅ 训练 |
| History Encoder | ~1.68M | ✅ 训练 |
| Actor FC | ~2K | ✅ 训练 |
| **总计可训练** | **~5.5M** | |

**同时调整的其他参数**:
- `actor_lr`: 3e-4 → 5e-5（训练更多参数需要更小 LR）
- `batch_size`: 128 → 64（DPT 梯度需要更多显存）

---

### 🟡 问题3: Critic detach 阻断梯度

**位置**: `vlm_net.py:727-728`

```python
# VLM_DPT_Critic.forward()
if self.detach_features:
    vlm_feat = vlm_feat.detach()  # ← 阻断梯度
```

**设计意图**: 防止 Critic 的梯度污染 Actor 的 VLM/DPT 更新

**问题**:
- Critic 只能通过 Q-head 学习，无法优化 VLM 特征
- 如果 VLM 提取的特征不适合 Q-value 估计，Critic 无法改进

**权衡**:
- 如果 VLM 冻结: detach 无影响（本来就没梯度）
- 如果 DPT 可训练: detach 意味着只有 Actor loss 能更新 DPT

**建议**: 如果放开 DPT 训练，可以考虑 `detach_features=False`

---

### 🟢 问题4: Action 归一化可能不一致

**位置**: `rl.py:341-344, 398-405`

**流程**:
```python
# Actor 输出
actor_output = self.actor(state)  # [-1, 1]
predicted_action = actor_output * _action_scale + _action_bias  # 映射到参数空间

# 归一化给 Critic
predicted_action_normalized = (predicted_action - param_mean) / param_std
```

**潜在问题**:
- `_action_scale/bias` 来自 `action_space` 边界（理论范围）
- `param_mean/std` 来自训练数据分布（实际分布）
- 如果两者不一致，Actor 的输出范围可能与训练数据不匹配

**示例**:
```
action_space: [0.1, 2.0] (max_vel_x 的理论范围)
训练数据分布: mean=0.8, std=0.3 (实际采集的数据)

如果 Actor 输出 -1 → 映射到 0.1
归一化后: (0.1 - 0.8) / 0.3 = -2.33 (超出训练分布)
```

---

### 核心矛盾

```
┌─────────────────────────────────────────────────────────────┐
│  如果全冻结 (当前配置):                                      │
│    ✓ Target 共享问题不严重                                   │
│    ✗ 可训练参数太少 (~2K)，学习能力极其有限                   │
│                                                             │
│  如果放开 DPT:                                               │
│    ✓ 可训练参数增加 (~5M)，能真正做 RLFT                     │
│    ✗ Target 共享问题变严重，需要修复                         │
└─────────────────────────────────────────────────────────────┘
```

---

### 建议的修复方案

#### 方案 A: 保持现状，接受只训练 FC（保守）

适用场景: Stage 1 监督学习效果已经很好，RLFT 只需微调决策边界

#### 方案 B: 为 DPT 创建独立 Target（推荐）

```python
def _create_actor_target(self, actor):
    import copy

    # VLM 共享（4-bit 无法复制，且冻结）
    # DPT 独立复制（只有 ~4M，可以 deepcopy）
    dpt_target = copy.deepcopy(actor.feature_extractor.dpt_head)

    # History 也独立复制
    history_target = copy.deepcopy(actor.feature_extractor.history_encoder)

    # 创建 target，使用独立的 DPT/History
    ...
```

然后在 `_soft_update_actor` 中也更新 DPT/History 的 target。

#### 方案 C: 使用 Polyak averaging 替代共享

对于无法 deepcopy 的组件，使用 exponential moving average (EMA) 的方式：
- 维护 DPT 权重的 EMA 副本
- 推理时切换到 EMA 权重

---

## 🐛 常见问题

### Q1: 显存不足怎么办？

**解决方案**:
1. 减小batch_size（256 → 128 → 64）
2. 使用4-bit量化加载VLM
3. 只在Actor中使用VLM，Critic用轻量CNN

### Q2: 训练不稳定？

**解决方案**:
1. 减小actor_lr（1e-5 → 5e-6）
2. 增加pre_collect（10000 → 50000）
3. 减小exploration_noise_start（0.05 → 0.02）

### Q3: VLM加载失败？

**检查**:
1. checkpoint路径是否正确
2. 是否包含`adapter_config.json`（LoRA）
3. 是否包含`regression_head/pytorch_model.bin`（DPT）

### Q4: 性能不如监督学习？

**可能原因**:
1. RL探索破坏了预训练知识 → 减小exploration noise
2. 学习率太大 → 减小actor_lr
3. 训练步数不够 → 增加max_step

## 📊 预期性能

**监督学习baseline**:
- MAE: ~0.05-0.1（归一化后）
- 推理速度: ~100-500ms/frame

**FTRL目标**:
- 导航成功率: 超过监督学习
- 样本效率: <1M steps（vs APPLR的5M）
- 训练时间: ~24小时（单GPU）

## 🔬 实验建议

### 消融实验

1. **VLM冻结vs微调**
   - 配置: `freeze_vlm_actor: true/false`
   - 对比训练速度和性能

2. **DPT冻结vs微调**
   - 配置: `freeze_dpt_actor: true/false`
   - 验证FTRL的必要性

3. **不同学习率**
   - `actor_lr: [1e-6, 5e-6, 1e-5, 5e-5]`
   - 找最优学习率

### 对比实验

1. **FTRL vs 监督学习**
   - 在相同测试环境评估
   - 对比成功率、轨迹平滑度

2. **FTRL vs APPLR**
   - 样本效率对比
   - 性能上界对比

## 🚧 待实现功能

- **分布式数据收集**: 利用远程主机 CPU 加速 Gazebo 仿真 → 详见 [TODO_DISTRIBUTED.md](TODO_DISTRIBUTED.md)

---

## 📚 参考文献

- APPLR: Adaptive Planner Parameter Learning from Reinforcement
- TD3: Twin Delayed Deep Deterministic Policy Gradient
- DPT: Dense Prediction Transformer (参考DUSt3R)
- Qwen2.5-VL: 视觉语言模型

## 🤝 贡献

如有问题或改进建议，请联系项目维护者。

---

**Happy Fine-Tuning! 🚀**
