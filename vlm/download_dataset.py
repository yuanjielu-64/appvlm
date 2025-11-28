import os
import json
from datasets import load_dataset
from tqdm import tqdm

# 设置存储图像的路径
image_root = "/scratch/bwang25/llava_data/test_images"
os.makedirs(image_root, exist_ok=True)

# 设置数据集和split，加载流式数据
subset = load_dataset(
    "lmms-lab/LLaVA-NeXT-Data",
    split="train",  # 加载整个 train 数据集
    streaming=True  # 启用流式加载
)

# 使用 take 方法获取前 1000 条样本
first_1000_samples = subset.take(1000)

# 用于存储转换后的数据
converted = []

# 遍历数据集中的每个样本，处理图片和保存信息
for sample in tqdm(first_1000_samples, desc="Converting"):
    entry = {"id": sample["id"], "conversations": sample["conversations"]}
    
    if sample["image"] is not None:
        # 保存图像到指定文件夹
        entry["image"] = f"{sample['id']}.jpg"
        sample["image"].save(os.path.join(image_root, entry["image"]))
    
    converted.append(entry)

# 保存转换后的数据到JSON文件
json_path = "/scratch/bwang25/llava_data/test_data.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(converted, f, indent=4, ensure_ascii=False)

# 打印保存信息
print(f"Saved {len(converted)} samples to {json_path}")

