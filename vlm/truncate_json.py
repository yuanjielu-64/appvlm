import json
import pathlib

def truncate_json(input_path, output_path, num_samples):
    input_path = pathlib.Path(input_path)
    output_path = pathlib.Path(output_path)

    # 1. 读取原始 JSON
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 2. 截取前 num_samples 条
    truncated = data[:num_samples]

    # 3. 写入新 JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(truncated, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(truncated)} samples to {output_path}")

# 示例使用：
# truncate_json("all_0_149.json", "all_0_149_first_500.json", 500)
truncate_json("/scratch/bwang25/appvlm/buffer/dwa_heurstic/actor_2/actor_2.json", "/scratch/bwang25/appvlm/buffer/dwa_heurstic/test_100.json", 100)
