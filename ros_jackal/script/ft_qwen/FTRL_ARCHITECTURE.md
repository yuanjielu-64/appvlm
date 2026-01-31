# FTRL架构详解 - VLM+DPT Feature Extractor的256维特征

## 🎯 核心理解

**`actor_feature_extractor` 的输出是256维特征向量，这个256维来自DPT Head的中间pooling操作，NOT是最终的参数预测！**

---

## 📐 完整的数据流

### 监督学习阶段（qwen_dpt/lmms-finetune-qwen）

```python
# models/qwen2_5_vl_dpt_regression.py: DPTHead.forward()

[Costmap图像]
    ↓
Qwen2.5-VL-3B
    ↓ 提取最后4层hidden states
    ↓ [layer_-4, layer_-3, layer_-2, layer_-1]
    ↓
DPTHead.forward(multi_layer_hidden_states):

    # Step 1: 投影到统一256维空间
    projected = [
        proj(hidden_state)  # [B, seq_len, 2048] → [B, seq_len, 256]
        for proj, hidden_state in zip(self.projections, multi_layer_hidden_states)
    ]

    # Step 2: 渐进式融合（top-down refinement）
    projected = [p.transpose(1, 2) for p in projected]  # [B, 256, seq_len]
    fused = projected[-1]  # 从最高层开始
    for i in range(len(self.fusion_blocks) - 1, -1, -1):
        skip = projected[i]
        fused = self.fusion_blocks[i](fused, skip)  # 逐层融合
    fused = fused.transpose(1, 2)  # [B, seq_len, 256]

    # Step 3: 空间注意力池化 ✅✅✅ 关键点！
    attention_weights = self.spatial_attention(fused)  # [B, seq_len, 1]
    pooled = (fused * attention_weights).sum(dim=1)    # [B, 256] ← 256维特征！

    # Step 4: 回归MLP（监督学习用，RL不用！）
    predictions = self.mlp(pooled)  # [B, 256] → [B, 7参数]
    return predictions
```

**监督学习训练的checkpoint包含**：
```
checkpoint-5000/
├── adapter_config.json              # LoRA配置
├── adapter_model.safetensors        # LoRA权重 (330.91M)
├── regression_head/
│   └── pytorch_model.bin            # DPT Head (3.89M)
│       ├── projections.*            # Step 1的投影层
│       ├── fusion_blocks.*          # Step 2的融合层
│       ├── spatial_attention.*      # Step 3的注意力池化
│       └── mlp.*                    # Step 4的回归MLP
├── history_encoder/
│   └── pytorch_model.bin            # History Encoder (1.68M)
├── history_config.json              # 历史帧配置
└── normalization/
    ├── param_mean.npy
    └── param_std.npy
```

---

### RL微调阶段（rlft/vlm_net.py）

```python
# vlm_net.py: VLM_DPT_FeatureExtractor

class VLM_DPT_FeatureExtractor:
    def __init__(self, checkpoint_path, freeze_vlm=True, freeze_dpt=False, ...):
        # 1. 加载VLM Base (2.03B, 4-bit量化)
        self.base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            quantization_config=bnb_config,  # 4-bit量化
            device_map=device
        )

        # 2. 加载LoRA (330.91M)
        if os.path.exists(adapter_config.json):
            self.base_model = PeftModel.from_pretrained(
                self.base_model,
                checkpoint_path,
                is_trainable=(not freeze_vlm)  # ✅ 根据freeze_vlm决定
            )

            # 关键决策：是否merge LoRA
            if freeze_vlm:
                # VLM冻结：merge LoRA以节省显存
                self.base_model = self.base_model.merge_and_unload()
                print("✓ LoRA merged into base model (VLM frozen)")
            else:
                # VLM可训练：保持LoRA作为独立层，以便训练和保存
                print(f"✓ LoRA loaded as trainable layers: {lora_params:,} parameters")

        # 3. 加载DPT head (3.89M)
        self.dpt_head = DPTHead(
            hidden_size=2048,
            num_params=8,
            feature_dim=256,   # ✅ 关键：256维特征
            num_layers=4,
            use_history=use_history
        )
        state_dict = torch.load(regression_head_path)
        self.dpt_head.load_state_dict(state_dict, strict=False)

        # 4. 加载History Encoder (1.68M, 可选)
        if use_history:
            self.history_encoder = HistoryEncoder(...)
            history_state_dict = torch.load(history_encoder_path)
            self.history_encoder.load_state_dict(history_state_dict)

        # 5. 冻结策略
        if freeze_vlm:
            for param in self.base_model.parameters():
                param.requires_grad = False
        if freeze_dpt:
            for param in self.dpt_head.parameters():
                param.requires_grad = False

    def forward(self, images, prompt=None, history_images=None):
        # VLM前向传播
        outputs = self.base_model.model(**inputs, output_hidden_states=True)
        multi_layer_hidden_states = outputs.hidden_states[-4:]

        # 历史帧处理 (可选)
        history_feat = None
        if self.use_history and history_images is not None:
            history_feat = self.history_encoder(history_images)

        # ✅ 关键：只提取到256维特征，不经过mlp！
        features = self._extract_dpt_features(multi_layer_hidden_states, history_feat)
        return features  # [B, 256]

    def _extract_dpt_features(self, multi_layer_hidden_states, history_feat=None):
        """
        提取DPT的中间特征 (256-d pooled)，不经过最后的MLP回归层
        """
        # Step 1: 投影所有层到统一特征空间
        projected = [
            proj(hidden_state)
            for proj, hidden_state in zip(
                self.dpt_head.projections,
                multi_layer_hidden_states
            )
        ]

        # Step 2: 转换为Conv1d格式 [B, feature_dim, seq_len]
        projected = [p.transpose(1, 2) for p in projected]

        # Step 3: 渐进式融合 (top-down refinement)
        fused = projected[-1]
        for i in range(len(self.dpt_head.fusion_blocks) - 1, -1, -1):
            skip = projected[i]
            fused = self.dpt_head.fusion_blocks[i](fused, skip)

        fused = fused.transpose(1, 2)  # [B, seq_len, feature_dim]

        # Step 4: Spatial attention pooling ✅✅✅ 就是这里！
        attention_weights = self.dpt_head.spatial_attention(fused)  # [B, seq_len, 1]
        pooled = (fused * attention_weights).sum(dim=1)  # [B, 256] ← 256维特征！

        # Step 5: 历史特征融合 (可选)
        if self.use_history and history_feat is not None:
            combined = torch.cat([pooled, history_feat], dim=-1)  # [B, 512]
            pooled = self.dpt_head.history_fusion(combined)  # [B, 256]

        return pooled  # [B, 256]
        # ❌ 注意：没有调用 self.dpt_head.mlp ！


# vlm_net.py: VLM_DPT_Actor

class VLM_DPT_Actor:
    def __init__(self, feature_extractor, action_dim=7):
        self.feature_extractor = feature_extractor  # 上面的FeatureExtractor
        self.fc = nn.Linear(256, action_dim)  # ✅ 新的决策层！256 → 7

    def forward(self, images, prompt=None, history_images=None):
        features = self.feature_extractor(images, prompt, history_images)
        # features: [B, 256] ← 来自pooled
        action = torch.tanh(self.fc(features))  # [B, 7] ← 用新FC层预测
        return action
```

---

## 💾 Checkpoint保存和加载

### 三层Checkpoint系统

```
┌─────────────────────────────────────────────────────────┐
│ 1. 监督学习Checkpoint (只读)                            │
│    model/ddp/checkpoint-5000/                           │
│    ├─ VLM base (2.03B, 从HuggingFace)                  │
│    ├─ LoRA (330.91M, adapter_model.safetensors)       │
│    ├─ DPT Head (3.89M, regression_head/)              │
│    └─ History Encoder (1.68M, history_encoder/)       │
└─────────────────────────────────────────────────────────┘
                    ↓ 引用
┌─────────────────────────────────────────────────────────┐
│ 2. RL训练Checkpoint (读写)                              │
│    logging/.../policy_step_1000_*                       │
│    ├─ policy_step_1000_actor (22MB)                    │
│    │   └─ DPT + History + FC 训练后参数                │
│    ├─ policy_step_1000_vlm_info (154B) ✅              │
│    │   └─ 记录监督学习checkpoint路径                   │
│    ├─ policy_step_1000_noise (21B)                     │
│    │   └─ 探索噪声                                      │
│    └─ policy_step_1000_lora_adapter/ (~1.3GB, 可选)   │
│        └─ 如果freeze_vlm=False，保存LoRA更新          │
└─────────────────────────────────────────────────────────┘
                    ↓ 实时同步
┌─────────────────────────────────────────────────────────┐
│ 3. Condor实时Policy (临时, buffer目录)                  │
│    buffer/ddp_rlft/                                     │
│    ├─ policy_actor (22MB) - 最新的DPT+FC               │
│    ├─ policy_vlm_info (154B) - VLM配置信息            │
│    └─ policy_noise (21B) - 探索噪声                    │
│    用于在线数据收集                                      │
└─────────────────────────────────────────────────────────┘
```

### policy_vlm_info 的作用

**它是一个指针文件**，记录如何重建VLM：

```python
# policy_vlm_info 内容
{
    'checkpoint_path': '/path/to/model/ddp/checkpoint-5000',  # ← VLM从哪里加载
    'use_4bit': True,          # 重新加载时用4-bit量化
    'use_history': True,       # 是否使用History Encoder
    'vlm_trainable': False     # 是否有LoRA更新需要加载
}
```

**为什么需要它？**
- VLM base (2.03B) 因为4-bit量化无法用pickle保存
- 每次启动都从HuggingFace重新加载VLM base
- 然后根据vlm_info从正确的checkpoint加载LoRA
- 确保在不同环境/进程中能正确重建完整模型

### 保存策略 (rl.py:209-271)

```python
def save(self, dir, filename):
    """
    保存策略：
    1. VLM base: 跳过（4-bit量化，无法pickle）
    2. LoRA adapters: 如果可训练，使用PEFT保存 ✅ 新增
    3. DPT + History + FC：保存可训练参数
    """
    state_dict_to_save = {}

    for name, param in self.actor.named_parameters():
        # ❌ 跳过VLM base参数（4-bit量化，无法pickle）
        if 'feature_extractor.base_model' in name:
            continue

        # ✅ 保存DPT、History、FC参数
        state_dict_to_save[name] = param.detach().cpu()

    # 保存DPT + History + FC
    pickle.dump(state_dict_to_save, f)  # → policy_*_actor

    # 保存探索噪声
    pickle.dump(self.exploration_noise, f)  # → policy_*_noise

    # ✅ 检查VLM是否有可训练的LoRA参数
    vlm_trainable = any(p.requires_grad for p in base_model.parameters())

    if vlm_trainable:
        # ✅ 如果VLM可训练，保存LoRA adapters
        from peft import PeftModel
        if isinstance(base_model, PeftModel):
            lora_save_path = join(dir, filename + "_lora_adapter")
            base_model.save_pretrained(lora_save_path)
            print(f"✓ LoRA adapters saved to {lora_save_path}")

    # 保存VLM配置信息
    pickle.dump({
        'checkpoint_path': checkpoint_path,
        'use_4bit': True,
        'use_history': use_history,
        'vlm_trainable': vlm_trainable  # ✅ 新增
    }, f)  # → policy_*_vlm_info
```

### 加载策略 (rl.py:273-329)

```python
def load(self, dir, filename):
    """
    加载策略：
    1. VLM base已在初始化时从checkpoint加载
    2. 加载DPT + History + FC的训练后参数
    3. 如果有LoRA更新，加载LoRA adapters ✅ 新增
    """
    # 1. 加载DPT + History + FC
    saved_state_dict = pickle.load(f)  # ← policy_*_actor
    self.actor.load_state_dict(saved_state_dict, strict=False)

    # 2. ✅ 加载LoRA adapters（如果存在）
    lora_save_path = join(dir, filename + "_lora_adapter")
    if exists(lora_save_path):
        from peft import PeftModel
        if isinstance(base_model, PeftModel):
            # 先unload旧LoRA，加载新的
            base_model = base_model.unmerge_and_unload()
            base_model = PeftModel.from_pretrained(
                base_model,
                lora_save_path,
                is_trainable=True  # 如果继续训练
            )
            print(f"✓ LoRA adapters loaded from {lora_save_path}")

    # 3. 加载噪声
    self.exploration_noise = pickle.load(f)  # ← policy_*_noise
```

### Collector的原子性保存 (collector.py:641-656)

```python
def save_policy(self):
    """
    将当前policy保存到buffer_path/
    使用原子性重命名避免race condition
    """
    # Step 1: 保存为临时文件
    self.policy.save(self.buffer_path, "policy_copy")
    # 创建: policy_copy_actor, policy_copy_noise, policy_copy_vlm_info

    # Step 2: 原子性重命名（防止actor读取到损坏的文件）
    shutil.move(
        join(self.buffer_path, "policy_copy_actor"),
        join(self.buffer_path, "policy_actor")  # ← 最终文件
    )
    shutil.move(
        join(self.buffer_path, "policy_copy_noise"),
        join(self.buffer_path, "policy_noise")
    )
    shutil.move(
        join(self.buffer_path, "policy_copy_vlm_info"),
        join(self.buffer_path, "policy_vlm_info")  # ✅ 修复：之前忘记重命名
    )
```

---

## 🔧 两种训练模式

### 模式1: VLM冻结 (freeze_vlm=True) - 当前默认 ✅

```python
VLM Base (2.03B)  → 冻结 → merge LoRA → 无需保存
LoRA (330.91M)    → 合并到base → 无需保存
DPT (3.89M)       → 可训练 → 保存
History (1.68M)   → 可训练 → 保存
FC (2K)           → 可训练 → 保存
─────────────────────────────────────────────
保存大小: 22MB (只有DPT + History + FC)
```

**保存文件**:
```
logging/.../
├── policy_step_1000_actor        # 22MB: DPT + History + FC
├── policy_step_1000_noise        # 探索噪声
└── policy_step_1000_vlm_info     # VLM路径信息
```

**启动参数**:
```yaml
training_config:
  freeze_vlm_actor: true   # VLM冻结
  freeze_dpt_actor: false  # DPT可训练
  actor_lr: 1.0e-5
```

### 模式2: VLM可训练 (freeze_vlm=False) - 消融实验

```python
VLM Base (2.03B)  → 冻结 → 无需保存 (从HF加载)
LoRA (330.91M)    → 可训练 → 保存LoRA adapters ✅
DPT (3.89M)       → 可训练 → 保存
History (1.68M)   → 可训练 → 保存
FC (2K)           → 可训练 → 保存
─────────────────────────────────────────────
保存大小: ~1.3GB (LoRA 1.3GB + DPT等 22MB)
```

**保存文件**:
```
logging/.../
├── policy_step_1000_actor           # 22MB: DPT + History + FC
├── policy_step_1000_lora_adapter/   # ~1.3GB: LoRA增量参数 ✅
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── policy_step_1000_noise           # 探索噪声
└── policy_step_1000_vlm_info        # VLM路径信息 (vlm_trainable=True)
```

**启动参数**:
```yaml
training_config:
  freeze_vlm_actor: false  # VLM可训练 ✅
  freeze_dpt_actor: false  # DPT可训练
  actor_lr: 5.0e-6         # 更小的学习率保护预训练
```

---

## 📊 参数统计

### 完整模型参数分布 (freeze_vlm=True)

```
======================================================================
📊 VLM_DPT_FeatureExtractor Parameter Summary
======================================================================
VLM Base (Qwen2.5-VL-3B):
  ├─ Total:      2,034,024,448 parameters (2.03B)
  ├─ Trainable:             0 parameters (0.00M)
  └─ Frozen:     2,034,024,448 parameters (2.03B)

DPT Head:
  ├─ Total:         3,891,337 parameters (3.89M)
  ├─ Trainable:     3,891,337 parameters (3.89M)
  └─ Frozen:                0 parameters (0.00M)

History Encoder:
  ├─ Total:         1,676,352 parameters (1.68M)
  ├─ Trainable:     1,676,352 parameters (1.68M)
  └─ Frozen:                0 parameters (0.00M)

──────────────────────────────────────────────────────────────────────
Total Feature Extractor:
  ├─ Total:      2,039,592,137 parameters (2.04B)
  ├─ Trainable:     5,567,689 parameters (5.57M)
  ├─ Frozen:     2,034,024,448 parameters (2.03B)
  └─ Trainable%:        0.27%
======================================================================
```

### 文件大小验证

```python
# 理论计算
5.57M 参数 × 4 bytes (float32) = 22.28MB

# 实际文件
buffer/ddp_rlft/policy_actor: 22MB ✅ 完全匹配！
```

---

## 🎓 总结

### 核心设计理念

1. **256维特征 = DPT Head的空间注意力池化结果**
   ```python
   pooled = (fused * attention_weights).sum(dim=1)  # [B, 256]
   ```

2. **监督学习和RL的分工**
   - 监督学习：训练场景理解（VLM + DPT前半部分）
   - RL微调：训练决策策略（Actor.fc层 + 可选DPT微调）

3. **为什么这样设计**
   - 场景理解是通用的（可以从监督学习复用）
   - 决策策略是任务相关的（RL重新学习）
   - 样本效率高（因为场景理解已经很强了）

### 保存策略汇总

| 组件 | 参数量 | 是否保存 | 大小 | 原因 |
|------|-------|---------|------|------|
| **VLM Base** | 2.03B | ❌ 不保存 | 0 MB | 4-bit量化，从HF重新加载 |
| **LoRA (冻结)** | 330.91M | ❌ 不保存 | 0 MB | 合并到VLM，从监督学习加载 |
| **LoRA (可训练)** | 330.91M | ✅ 保存 | ~1.3 GB | 使用PEFT.save_pretrained() |
| **DPT Head** | 3.89M | ✅ 保存 | 15.56 MB | 训练更新 |
| **History Encoder** | 1.68M | ✅ 保存 | 6.72 MB | 训练更新 |
| **FC层** | 2K | ✅ 保存 | 0.008 MB | 训练更新 |

### 技术要点

1. **4-bit量化的VLM无法用pickle保存**
   - 只能从HuggingFace重新加载
   - 使用policy_vlm_info记录checkpoint路径

2. **LoRA的条件保存**
   - freeze_vlm=True: merge后不保存
   - freeze_vlm=False: 使用PEFT保存adapter

3. **原子性保存**
   - 使用policy_copy_*临时文件
   - 原子性重命名防止race condition

---

**最后更新**: 2026-01-19
**关键文件**:
- `src/ros_jackal/rlft/vlm_net.py`: VLM_DPT_FeatureExtractor和Actor定义
- `src/ros_jackal/rlft/rl.py`: TD3 save/load逻辑
- `src/ros_jackal/rlft/collector.py`: Condor保存逻辑
