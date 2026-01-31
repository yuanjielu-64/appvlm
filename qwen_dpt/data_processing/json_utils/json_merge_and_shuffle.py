#!/usr/bin/env python3
"""
先合并多个 actor_*/actor_*.json，再 shuffle 并分块

流程:
1. 从多个 actor 目录读取 JSON 文件并合并
2. 流式打乱并分块输出到 chunk_*.json
"""

import pathlib, json, random, tempfile, ijson, shutil
from decimal import Decimal

def convert_decimals(obj):
    """递归转换 Decimal 为 float"""
    if isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_decimals(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_decimals(item) for item in obj]
    return obj

# ============ 配置参数 ============
root_dir = pathlib.Path("/home/yuanjielu/robot_navigation/noetic/app_data/ddp_heurstic")
start_actor = 0
end_actor = 300  # 不包含
out_dir = pathlib.Path("/home/yuanjielu/robot_navigation/noetic/app_data/ddp_heurstic/splits_200k")
chunk_size = 300_000

# 参数长度过滤配置
PLANNER = "ddp"  # 选择: dwa, teb, mppi, ddp (必须与root_dir对应！)
FILTER_BY_PARAM_LENGTH = True  # 是否启用参数长度过滤
EXPECTED_PARAM_LENGTHS = {
    "dwa": 9,
    "teb": 9,
    "mppi": 10,
    "ddp": 8
}
# ==================================

# 删除已存在的输出目录
if out_dir.exists():
    shutil.rmtree(out_dir)
    print(f"✓ Removed existing directory: {out_dir}\n")
out_dir.mkdir(parents=True, exist_ok=True)

tmp_dir = tempfile.TemporaryDirectory()
tmp_files = []
chunk = []
total_samples = 0
filtered_samples = 0  # 统计被过滤的样本数

# 获取期望的参数长度
expected_length = EXPECTED_PARAM_LENGTHS.get(PLANNER.lower()) if FILTER_BY_PARAM_LENGTH else None

if FILTER_BY_PARAM_LENGTH:
    print(f"[INFO] 参数长度过滤已启用")
    print(f"       Planner: {PLANNER.upper()}")
    print(f"       Expected parameters: {expected_length}")
    print(f"       只保留参数长度 = {expected_length} 的样本\n")
else:
    print(f"[INFO] 参数长度过滤已禁用\n")

print(f"[Step 1/3] 合并 actor_{start_actor} ~ actor_{end_actor-1} 的 JSON 文件...\n")

# 遍历所有 actor 目录
for i in range(start_actor, end_actor):
    actor_file = root_dir / f"actor_{i}" / f"actor_{i}.json"

    if not actor_file.exists():
        print(f"  [SKIP] {actor_file} not found")
        continue

    print(f"  [READ] actor_{i}/actor_{i}.json", end="")

    # 流式读取单个 actor 文件 (限制最多 2000 个样本)
    max_samples_per_actor = 2000
    with open(actor_file, "r", encoding="utf-8") as f:
        actor_count = 0
        actor_filtered = 0
        for obj in ijson.items(f, "item"):  # 按数组元素迭代

            # 参数长度过滤
            if FILTER_BY_PARAM_LENGTH and expected_length is not None:
                if "parameters" not in obj:
                    actor_filtered += 1
                    filtered_samples += 1
                    continue

                param_len = len(obj["parameters"])
                if param_len != expected_length:
                    actor_filtered += 1
                    filtered_samples += 1
                    continue

            chunk.append(obj)
            actor_count += 1
            total_samples += 1

            # 达到 chunk_size 则局部 shuffle 并写临时文件
            if len(chunk) >= chunk_size:
                random.shuffle(chunk)
                tmp = pathlib.Path(tmp_dir.name) / f"tmp_{len(tmp_files):03d}.jsonl"
                with open(tmp, "w", encoding="utf-8") as tf:
                    for row in chunk:
                        row_cleaned = convert_decimals(row)
                        tf.write(json.dumps(row_cleaned, ensure_ascii=False) + "\n")
                tmp_files.append(tmp)
                chunk.clear()

            # 限制每个 actor 最多读取 2000 个样本
            if actor_count >= max_samples_per_actor:
                break

        if actor_filtered > 0:
            print(f" → {actor_count} samples (filtered: {actor_filtered})")
        else:
            print(f" → {actor_count} samples")

# 处理最后不足 chunk_size 的数据
if chunk:
    random.shuffle(chunk)
    tmp = pathlib.Path(tmp_dir.name) / f"tmp_{len(tmp_files):03d}.jsonl"
    with open(tmp, "w", encoding="utf-8") as tf:
        for row in chunk:
            row_cleaned = convert_decimals(row)
            tf.write(json.dumps(row_cleaned, ensure_ascii=False) + "\n")
    tmp_files.append(tmp)
    chunk.clear()

print(f"\n✓ 总共合并 {total_samples} 条样本")
print(f"✓ 生成 {len(tmp_files)} 个临时文件\n")

# 随机打乱临时文件顺序（实现全局 shuffle）
print(f"[Step 2/3] 随机打乱临时文件顺序...\n")
random.shuffle(tmp_files)

# 重新聚合成最终 chunk 文件
print(f"[Step 3/3] 生成最终 chunk 文件...\n")
buffer = []
out_idx = 0

for tmp in tmp_files:
    with open(tmp, "r", encoding="utf-8") as tf:
        for line in tf:
            buffer.append(json.loads(line))
            if len(buffer) >= chunk_size:
                out_file = out_dir / f"chunk_{out_idx:03d}.json"
                with open(out_file, "w", encoding="utf-8") as of:
                    json.dump(buffer, of, ensure_ascii=False)
                print(f"  ✓ wrote {len(buffer):,} samples to {out_file}")
                buffer.clear()
                out_idx += 1

# 写入最后不足 chunk_size 的数据
if buffer:
    out_file = out_dir / f"chunk_{out_idx:03d}.json"
    with open(out_file, "w", encoding="utf-8") as of:
        json.dump(buffer, of, ensure_ascii=False)
    print(f"  ✓ wrote {len(buffer):,} samples to {out_file}")

# 清理临时目录
tmp_dir.cleanup()

print(f"\n{'='*60}")
print(f"✓ 完成!")
print(f"  总样本数: {total_samples:,}")
if FILTER_BY_PARAM_LENGTH and filtered_samples > 0:
    print(f"  过滤样本数: {filtered_samples:,} ({100*filtered_samples/(total_samples+filtered_samples):.2f}%)")
    print(f"  保留样本数: {total_samples:,} ({100*total_samples/(total_samples+filtered_samples):.2f}%)")
print(f"  输出目录: {out_dir}")
print(f"  生成文件: chunk_000.json ~ chunk_{out_idx:03d}.json")
print(f"{'='*60}\n")
