# pip install ijson
import pathlib, json, random, tempfile, ijson, shutil
from decimal import Decimal

def convert_decimals(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_decimals(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_decimals(item) for item in obj]
    return obj

json_path = pathlib.Path("/home/yuanjielu/Downloads/all_0_299.json")
out_dir = pathlib.Path("/home/yuanjielu/Downloads/splits_200k")

# 删除已存在的输出目录
if out_dir.exists():
    shutil.rmtree(out_dir)
    print(f"Removed existing directory: {out_dir}")

out_dir.mkdir(exist_ok=True)

chunk_size = 200_000
tmp_dir = tempfile.TemporaryDirectory()
tmp_files = []

# 流式读大数组，分块局部 shuffle 后落盘临时文件
with open(json_path, "r", encoding="utf-8") as f:
    chunk = []
    for obj in ijson.items(f, "item"):  # 按数组元素迭代
        chunk.append(obj)
        if len(chunk) >= chunk_size:
            random.shuffle(chunk)
            tmp = pathlib.Path(tmp_dir.name) / f"tmp_{len(tmp_files):03d}.jsonl"
            with open(tmp, "w", encoding="utf-8") as tf:
                for row in chunk:
                    row_cleaned = convert_decimals(row)
                    tf.write(json.dumps(row_cleaned, ensure_ascii=False) + "\n")
            tmp_files.append(tmp)
            chunk.clear()

    if chunk:
        random.shuffle(chunk)
        tmp = pathlib.Path(tmp_dir.name) / f"tmp_{len(tmp_files):03d}.jsonl"
        with open(tmp, "w", encoding="utf-8") as tf:
            for row in chunk:
                row_cleaned = convert_decimals(row)
                tf.write(json.dumps(row_cleaned, ensure_ascii=False) + "\n")
        tmp_files.append(tmp)

# 随机打乱临时文件顺序，再按 200k 聚合成最终 chunk_*.json
random.shuffle(tmp_files)
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
                print(f"wrote {len(buffer)} to {out_file}")
                buffer.clear()
                out_idx += 1

if buffer:
    out_file = out_dir / f"chunk_{out_idx:03d}.json"
    with open(out_file, "w", encoding="utf-8") as of:
        json.dump(buffer, of, ensure_ascii=False)
    print(f"wrote {len(buffer)} to {out_file}")

tmp_dir.cleanup()
