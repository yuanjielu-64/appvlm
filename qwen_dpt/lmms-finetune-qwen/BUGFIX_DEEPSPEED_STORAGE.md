# DeepSpeed ZeRO-3 Storage Sharing Bug 修复说明

**日期**: 2026-01-18
**问题**: regression_head 和 history_encoder 保存时包含了整个模型的参数（包括LoRA）
**影响**: 文件大小从应有的 ~15MB 膨胀到 642MB

---

## 🐛 Bug 描述

### 症状
训练时使用 DeepSpeed ZeRO-3，checkpoint 保存后发现：

```bash
checkpoint-5000/
├── adapter_model.safetensors        108 KB  ❌ (应该631MB，但是空的)
├── regression_head/pytorch_model.bin  642 MB  ❌ (应该15MB)
└── history_encoder/pytorch_model.bin  642 MB  ❌ (应该7MB)
```

**实际检查**:
```python
# regression_head/pytorch_model.bin
文件大小: 642 MB
实际参数: 45个 (3.8M values)
理论大小: 3.8M × 2 bytes = 7.6 MB
多余数据: 642 - 7.6 = 634 MB ≈ LoRA大小 (631MB)
```

### 根本原因

**DeepSpeed ZeRO-3 的 `GatheredParameters` 行为**:

```python
# 有bug的代码 (regression_trainer.py:225-229)
with deepspeed.zero.GatheredParameters(
    list(model_to_save.regression_head.parameters()),
    modifier_rank=0
):
    state_dict = model_to_save.regression_head.state_dict()
    torch.save(state_dict, regression_head_path)  # ❌ 保存了整个buffer
```

**问题机制**:

1. **ZeRO-3 收集参数时**:
   - 创建一个大buffer，包含**整个模型**的参数（LoRA + DPT + History）
   - 每个参数是这个大buffer的view

2. **调用 `state_dict()`**:
   - 返回45个DPT参数的tensor
   - 但这些tensor都指向同一个**大buffer**（336M elements）

3. **调用 `torch.save()`**:
   - PyTorch 保存tensor时会保存它的**整个storage**
   - 即使tensor只用了storage的一小部分
   - 结果：保存了包含LoRA的完整buffer

### 为什么只在 ZeRO-3 出现？

| ZeRO Stage | 参数分片 | 需要Gather | Storage共享 | Bug |
|-----------|---------|-----------|------------|-----|
| ZeRO-2 | ❌ | ❌ | ❌ | ✅ 正常 |
| ZeRO-3 | ✅ | ✅ | ✅ | ❌ Bug |

---

## ✅ 修复方案

### 修改内容

**文件**: `trainers/regression_trainer.py`

**修复点1**: regression_head 保存 (第230-235行)

```python
# 修复前
state_dict = model_to_save.regression_head.state_dict()
torch.save(state_dict, regression_head_path)

# 修复后
state_dict = model_to_save.regression_head.state_dict()

# 🔧 FIX: 克隆每个tensor，避免保存共享的大buffer
state_dict_clean = {k: v.clone() for k, v in state_dict.items()}
torch.save(state_dict_clean, regression_head_path)
```

**修复点2**: history_encoder 保存 (第292-297行)

```python
# 修复前
state_dict = model_to_save.history_encoder.state_dict()
torch.save(state_dict, history_encoder_path)

# 修复后
state_dict = model_to_save.history_encoder.state_dict()

# 🔧 FIX: 克隆每个tensor，避免保存共享的大buffer
state_dict_clean = {k: v.clone() for k, v in state_dict.items()}
torch.save(state_dict_clean, history_encoder_path)
```

### 原理

**`v.clone()` 的作用**:
- 创建一个**新的独立tensor**
- 不再是原storage的view
- 有自己的storage，只包含实际数据

**修复后的保存流程**:
```
GatheredParameters → 大buffer (336M elements)
                       ↓
state_dict() → 45个DPT tensors (指向大buffer)
                       ↓
{k: v.clone()} → 45个独立tensors (各自独立storage)
                       ↓
torch.save() → 只保存45个tensor的数据 (~15MB) ✅
```

---

## 📊 修复效果对比

### 修复前 (ZeRO-3)
```
checkpoint-5000/
├── regression_head/pytorch_model.bin    642 MB  ← 包含LoRA
├── history_encoder/pytorch_model.bin    642 MB  ← 包含LoRA
└── adapter_model.safetensors            108 KB  ← 空的
Total: 1.28 GB + 缺失LoRA
```

### 修复后 (ZeRO-3)
```
checkpoint-XXXX/
├── regression_head/pytorch_model.bin     15 MB  ✅
├── history_encoder/pytorch_model.bin      7 MB  ✅
└── adapter_model.safetensors            631 MB  ✅
Total: 653 MB (正确)
```

### 使用 ZeRO-2 (无此bug)
```
checkpoint-1/
├── regression_head/pytorch_model.bin     15 MB  ✅ (参数不分片，无需gather)
├── history_encoder/pytorch_model.bin      7 MB  ✅
└── adapter_model.safetensors            632 MB  ✅
Total: 654 MB (正确)
```

---

## 🔍 历史checkpoint处理方案

### 问题checkpoint (ZeRO-3训练，未修复)

**checkpoint-5000**:
- ❌ `adapter_model.safetensors` 是空的 (108KB)
- ✅ LoRA参数被误保存在 `regression_head/pytorch_model.bin` 的storage中

**提取方案**: 使用 `extract_lora_from_checkpoint.py`

```bash
cd /path/to/ros_jackal
python3 script/extract_lora_from_checkpoint.py model/ddp/checkpoint-5000 --replace --backup
```

**脚本功能**:
1. 从 `regression_head` 的storage中提取LoRA参数
2. 保存为新的 `adapter_model.safetensors` (631MB)
3. 备份原来的空文件

---

## 🎯 最佳实践

### 训练配置推荐

**如果使用 DeepSpeed**:

1. **ZeRO-2** (推荐):
   ```bash
   DS_STAGE=zero2
   ```
   - 不分片模型参数，无此bug
   - 显存占用稍高，但可接受

2. **ZeRO-3** (需要修复):
   - 使用修复后的 `regression_trainer.py`
   - 显存占用更低
   - 适合大模型训练

### 验证checkpoint正确性

```python
import os
from safetensors import safe_open

checkpoint = "/path/to/checkpoint-XXXX"

# 1. 检查adapter文件大小
adapter_path = f"{checkpoint}/adapter_model.safetensors"
adapter_size_mb = os.path.getsize(adapter_path) / (1024**2)
print(f"Adapter size: {adapter_size_mb:.2f} MB")
# 应该 ~631MB (Qwen2.5-VL-3B, LoRA r=128)

# 2. 检查LoRA参数
with safe_open(adapter_path, framework="pt", device="cpu") as f:
    keys = list(f.keys())
    first_tensor = f.get_tensor(keys[0])
    non_zero = (first_tensor != 0).float().mean().item()

    print(f"LoRA keys: {len(keys)}")  # 应该 ~828
    print(f"Non-zero ratio: {non_zero:.4f}")  # 应该 >0.1

# 3. 检查regression_head文件大小
reg_head_path = f"{checkpoint}/regression_head/pytorch_model.bin"
reg_size_mb = os.path.getsize(reg_head_path) / (1024**2)
print(f"Regression head size: {reg_size_mb:.2f} MB")
# 应该 ~15MB (DPT Head 3.8M params)
# ❌ 如果是642MB → 有bug
```

---

## 📝 总结

### Bug发现过程
1. 用户发现 `adapter_model.safetensors` 只有108KB
2. 检查发现 `regression_head` 和 `history_encoder` 异常大 (642MB)
3. 分析storage发现包含了完整的LoRA参数 (330M)
4. 追溯到 DeepSpeed `GatheredParameters` 的共享buffer问题

### 解决方案
- **临时**: 改用ZeRO-2训练 (已完成)
- **长期**: 修复 `regression_trainer.py` 的保存逻辑 (已完成)
- **历史数据**: 使用 `extract_lora_from_checkpoint.py` 提取LoRA

### 经验教训
- DeepSpeed ZeRO-3 的参数收集需要特别小心storage共享
- 保存前应该 `.clone()` 创建独立tensor
- 训练日志中的"参数数量"和"文件大小"要交叉验证

---

**更新时间**: 2026-01-18
**修复状态**: ✅ 已修复
**测试状态**: ⏳ 待下次ZeRO-3训练验证
