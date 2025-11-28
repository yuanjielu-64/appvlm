import json, random, pathlib

src = pathlib.Path("/scratch/bwang25/appvlm/buffer/dwa_heurstic/sample_200k.yaml")  # 如果你用 YAML 采样，先把其中的 json_path 指向的文件加载
json_path = pathlib.Path("/scratch/bwang25/appvlm/buffer/dwa_heurstic/all_0_149.json")  # 原始大 JSON
out_dir = pathlib.Path("/scratch/bwang25/appvlm/buffer/dwa_heurstic/splits_200k")
out_dir.mkdir(exist_ok=True)

# 先从大 JSON 读数据，然后打乱（全量 450 万）
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)
random.shuffle(data)

# 按块大小切分，每 200k 一份
chunk_size = 200_000
for i in range(0, len(data), chunk_size):
    chunk = data[i:i+chunk_size]
    out_file = out_dir / f"chunk_{i//chunk_size:03d}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(chunk, f, ensure_ascii=False)
    print(f"wrote {len(chunk)} to {out_file}")
print(f"total: {len(data)}")
