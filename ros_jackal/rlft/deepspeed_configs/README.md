# DeepSpeed ZeRO 配置

本目录包含 RLFT 训练使用的 DeepSpeed ZeRO 优化配置文件。

## ⚠️ 重要限制

**ZeRO-3 与 4-bit 量化 VLM 不兼容！**

| 配置 | ZeRO-2 | ZeRO-3 |
|-----|--------|--------|
| 4-bit VLM + LoRA | ✅ 兼容 | ❌ 不兼容 |
| 4-bit VLM (frozen) + DPT | ✅ 兼容 | ❌ 不兼容 |
| float16 VLM | ✅ 兼容 | ✅ 兼容 |

原因：bitsandbytes 的 NF4 格式不能被 ZeRO-3 分片。

**结论：RLFT 使用 4-bit VLM，只能用 ZeRO-2！**

## 配置文件说明

### zero2_config.json (推荐 ✅)

ZeRO-2 优化：分片优化器状态和梯度，模型参数保持完整。

**特点：**
- 与 4-bit 量化 VLM 完全兼容
- 显存节省约 50%（相比普通 DDP）
- 无额外通信开销
- LoRA、DPT、History 都可以训练

**适用场景：**
- 3-4 张 GPU 训练
- VLM 使用 4-bit 量化（RLFT 默认配置）

### zero3_config.json (⚠️ 仅供参考)

ZeRO-3 优化：分片优化器状态、梯度和模型参数。

**⚠️ 警告：不能与 4-bit VLM 一起使用！**

**适用场景：**
- 非量化模型（float16/bfloat16）
- 8+ 张 GPU 大规模训练

## 使用方法

### 方法 1：通过配置文件指定（推荐）

在 `ftrl_vlm_ddp.yaml` 中设置：

```yaml
training_config:
  # ZeRO-2（推荐）
  zero_stage: 2

  # 或者指定配置文件
  # zero_config_file: "rlft/deepspeed_configs/zero2_config.json"
```

启动训练：
```bash
torchrun --nproc_per_node=3 rlft/train.py \
    --config_file ddp \
    --policy_name ddp_rlft
```

### 方法 2：通过 accelerate 配置（高级）

使用 accelerate 配置文件：
```bash
accelerate launch --config_file rlft/accelerate_configs/zero2_config.yaml \
    rlft/train.py --config_file ddp --policy_name ddp_rlft
```

### 方法 3：环境变量

```bash
ZERO_STAGE=2 torchrun --nproc_per_node=3 rlft/train.py ...
```

## 4-bit 量化 VLM 注意事项

1. **VLM base 必须冻结**：4-bit 量化的 VLM base 不参与 ZeRO 分片
2. **可训练部分使用 ZeRO**：LoRA、DPT、FC 层正常参与 ZeRO 优化
3. **推荐 ZeRO-2**：与 4-bit 量化完全兼容

## 配置参数说明

| 参数 | ZeRO-2 | ZeRO-3 | 说明 |
|------|--------|--------|------|
| stage | 2 | 3 | ZeRO 优化阶段 |
| offload_optimizer | none | none | 优化器卸载到 CPU（可启用以进一步节省显存） |
| offload_param | - | none | 参数卸载到 CPU（仅 ZeRO-3） |
| overlap_comm | true | true | 通信与计算重叠 |
| contiguous_gradients | true | true | 连续梯度存储 |

## 显存估算

以 VLM + DPT + TD3 训练为例（batch_size=4 per GPU）：

| 模式 | 单卡显存 | 3卡总显存 | 备注 |
|------|---------|----------|------|
| 单卡 | ~24GB | - | 无分布式 |
| DDP | ~24GB | ~72GB | 每卡完整副本 |
| ZeRO-2 | ~14GB | ~42GB | 优化器+梯度分片 |
| ZeRO-3 | ~10GB | ~30GB | 全分片（有通信开销） |

## 故障排除

### 1. ZeRO-3 与 4-bit 模型冲突

错误信息：
```
RuntimeError: Cannot partition 4-bit quantized parameters
```

解决方案：使用 ZeRO-2 或确保 VLM base 冻结。

### 2. 梯度同步失败

错误信息：
```
RuntimeError: Expected all tensors to be on the same device
```

解决方案：确保使用 `accelerator.backward()` 而不是手动 `loss.backward()`。

### 3. 参数保存/加载问题

ZeRO-3 需要特殊处理参数保存：
```python
# 保存时自动收集所有分片
accelerator.save_state(output_dir)

# 或使用 unwrap_model
model = accelerator.unwrap_model(model)
torch.save(model.state_dict(), "model.pt")
```
