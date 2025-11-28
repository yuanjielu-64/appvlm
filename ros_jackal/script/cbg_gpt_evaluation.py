import json
import os
import base64
from pathlib import Path
import argparse
from openai import OpenAI
from tqdm import tqdm


def encode_image(image_path):
    """将图片编码为 base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def evaluate_vlm(json_path, image_root, output_path, api_key, model="gpt-4o"):
    """
    读取 JSON 文件,调用 VLM 生成预测结果

    Args:
        json_path: JSON 文件路径
        image_root: 图片根目录
        output_path: 输出结果路径
        api_key: OpenAI API key
        model: 使用的模型
    """

    # 初始化 OpenAI client
    client = OpenAI(api_key=api_key)

    # 读取 JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []

    print(f"Processing {len(data)} samples...")

    for item in tqdm(data):
        sample_id = item['id']
        image_rel_path = item['image']
        conversations = item['conversations']

        # 获取 human 的 prompt
        human_msg = conversations[0]['value']
        ground_truth = conversations[1]['value']

        # 构建完整图片路径
        image_path = Path(image_root) / image_rel_path

        if not image_path.exists():
            print(f"[SKIP] Image not found: {image_path}")
            continue

        # 编码图片
        base64_image = encode_image(image_path)

        # 准备消息
        # 移除 <image> 标记,因为我们会用 image_url
        prompt_text = human_msg.replace("<image>\n", "")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt_text
                    }
                ]
            }
        ]

        try:
            # 调用 API
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=500,
                temperature=0.0
            )

            prediction = response.choices[0].message.content

            # 保存结果
            result = {
                "id": sample_id,
                "image": image_rel_path,
                "prompt": prompt_text,
                "ground_truth": ground_truth,
                "prediction": prediction
            }
            results.append(result)

        except Exception as e:
            print(f"[ERROR] {sample_id}: {e}")
            continue

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate VLM on generated JSON data')
    parser.add_argument('--json_path', default='../ros_jackal/buffer/dwa_heurstic/actor_97/actor_97.json', help='Path to JSON file')
    parser.add_argument('--image_root', default='../ros_jackal/buffer/dwa_heurstic/', help='Root directory for images')
    parser.add_argument('--output_path', default='evaluation_results.json', help='Output path')
    parser.add_argument('--api_key', default=None, help='OpenAI API key (or use OPENAI_API_KEY env)')
    parser.add_argument('--model', default='gpt-4o', help='Model to use')
    parser.add_argument('--max_samples', type=int, default=1, help='Max samples to evaluate')

    args = parser.parse_args()

    api_key = args.api_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("Please provide API key via --api_key or OPENAI_API_KEY env variable")

    # 如果限制样本数
    if args.max_samples:
        with open(args.json_path, 'r') as f:
            data = json.load(f)
        data = data[:args.max_samples]
        temp_json = 'temp_subset.json'
        with open(temp_json, 'w') as f:
            json.dump(data, f)
        evaluate_vlm(temp_json, args.image_root, args.output_path, api_key, args.model)
        os.remove(temp_json)
    else:
        evaluate_vlm(args.json_path, args.image_root, args.output_path, api_key, args.model)