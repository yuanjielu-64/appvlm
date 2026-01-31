import argparse
from pathlib import Path

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def generate_response(
    image_path: str,
    prompt: str,
    base_model: str,
    lora_path: str,
    device: str = "auto",
    max_new_tokens: int = 256,
) -> str:
    """Run single-image inference with a LoRA-finetuned Qwen2.5-VL model."""
    image = Image.open(image_path).convert("RGB")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()

    processor = AutoProcessor.from_pretrained(base_model)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        images=[image],
        text=text,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return output


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a LoRA-finetuned Qwen2.5-VL model.")
    parser.add_argument("--image_path", type=Path, required=True, help="Path to the input image.")
    parser.add_argument("--prompt", type=str, required=True, help="User prompt to pair with the image.")
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Base (pretrained) model ID or local path.",
    )
    parser.add_argument(
        "--lora_path",
        type=Path,
        default=Path("/scratch/bwang25/checkpoints/qwen2.5-vl-7b-instruct_lora-True_0_200k/checkpoint-25000"),
        help="Path to the LoRA checkpoint directory.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Device map for loading the model, e.g., "auto", "cuda:0", or "cpu".',
    )
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max tokens to generate.")
    return parser.parse_args()


def main():
    args = parse_args()
    result = generate_response(
        image_path=str(args.image_path),
        prompt=args.prompt,
        base_model=args.base_model,
        lora_path=str(args.lora_path),
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )
    print(result)


if __name__ == "__main__":
    main()