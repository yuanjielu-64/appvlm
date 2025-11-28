#!/usr/bin/env python3
import os
import shutil
from huggingface_hub import hf_hub_download

repo_id = "openai/clip-vit-large-patch14-336"
filename = "mm_projector.bin"

target_dir = "/scratch/bwang25/checkpoints/projectors/llavanext-openai_clip-vit-large-patch14-336-Qwen_Qwen2.5-0.5B-Instruct"
os.makedirs(target_dir, exist_ok=True)

local_path = hf_hub_download(repo_id=repo_id, filename=filename)
shutil.copy(local_path, os.path.join(target_dir, filename))

print(f"Projector saved to {target_dir}/{filename}")
