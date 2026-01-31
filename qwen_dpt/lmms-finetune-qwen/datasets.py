import av
import os
import json
from PIL import Image
from typing import Dict, List, Optional

import numpy as np
from torch.utils.data import Dataset


TO_LOAD_IMAGE: Dict[str, bool] = {
    "llava-1.5": True,
    "llava-1.6": True,
    "llava-interleave": True,
    "llava-next-video": True,
    "llava-onevision": True,
    "qwen-vl": False,
    "phi3-v": True,
    "qwen2-vl": True,
    "qwen2.5-vl": True,
    "qwen2.5-vl-regression": True,
    "llama-3.2-vision": True,
    "internvl2": False,
}


def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning 
    which is generalized enough to handle both images and videos.
    """

    def __init__(
        self, 
        data_path: str, 
        model_family_id: str,
        image_folder: Optional[str] = None,
        video_folder: Optional[str] = None,
        num_frames: int = 8,
        user_key: str = "human",
        assistant_key: str = "gpt",
    ) -> None:
        super(LazySupervisedDataset, self).__init__()
        self.list_data_dict = json.load(open(data_path, "r"))
        self.image_folder = image_folder
        self.video_folder = video_folder
        self.num_frames = num_frames
        self.load_image = TO_LOAD_IMAGE[model_family_id]
        self.user_key = user_key
        self.assistant_key = assistant_key

        self.is_text_only = [
            "image" not in source and "video" not in source
            for source in self.list_data_dict
        ]

    def __len__(self) -> int:
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, List]:      
        source = self.list_data_dict[i]

        images = []
        if "image" in source:
            # here we do not do any image preprocessing but rather
            # let the processor handle everything
            # in some cases this may cause slight differences
            # but should totally be fine (e.g., official llava-1.5 does padding,
            # but llava-1.5-hf (huggingface's implementation) does not)
            if isinstance(source["image"], list):
                image_sources = source["image"]
            elif isinstance(source["image"], str):
                image_sources = [source["image"]]
            else:
                raise ValueError(f"Invalid image source type: {type(source['image'])}")
            
            for image_path in image_sources:
                if self.image_folder is not None:
                    image_path = os.path.join(self.image_folder, image_path)
                images.append(
                    Image.open(image_path).convert("RGB")
                    if self.load_image else image_path
                )

        videos = []
        if "video" in source:
            if isinstance(source["video"], list):
                video_sources = source["video"]
            elif isinstance(source["video"], str):
                video_sources = [source["video"]]
            else:
                raise ValueError(f"Invalid video source type: {type(source['video'])}")

            num_frames = [self.num_frames] * len(video_sources)

            for video_path, cur_num_frames in zip(video_sources, num_frames):
                if self.video_folder is not None:
                    video_path = os.path.join(self.video_folder, video_path)
                
                container = av.open(video_path)
                total_frames = container.streams.video[0].frames
                indices = np.arange(0, total_frames, total_frames / cur_num_frames).astype(int)
                clip = read_video_pyav(container, indices)

                videos.append(clip)
        
        system_prompt = None
        if "system_prompt" in source:
            system_prompt = source["system_prompt"]

        convs = []
        assert len(source["conversations"]) > 0, "No conversations found"
        for i, conv in enumerate(source["conversations"]):
            assert conv["from"] == (self.user_key if i % 2 == 0 else self.assistant_key), "Invalid conversation"
            convs.append(conv["value"])
        assert len(convs) % 2 == 0, "Odd number of conversations"
        
        return dict(
            images=images,
            videos=videos,
            conversations=convs,
            system_prompt=system_prompt
        )


class RegressionDataset(LazySupervisedDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        valid_indices = []
        for i, item in enumerate(self.list_data_dict):

            image_key = "images" if "images" in item else "image"
            if image_key not in item:
                continue

            image_sources = item[image_key] if isinstance(item[image_key], list) else [item[image_key]]

            all_exist = True
            for img_path in image_sources:
                full_path = os.path.join(self.image_folder, img_path) if self.image_folder else img_path
                if not os.path.exists(full_path):
                    # print(f"[WARNING] Image not found, skipping sample {i}: {full_path}")
                    all_exist = False
                    break

            if all_exist:
                valid_indices.append(i)

        self.list_data_dict = [self.list_data_dict[i] for i in valid_indices]
        print(f"[INFO] Filtered dataset: {len(valid_indices)} / {len(self.list_data_dict) + len(valid_indices) - len(self.list_data_dict)} valid samples")

    def __getitem__(self, i) -> Dict[str, List]:
        source = self.list_data_dict[i]

        images = []
        image_key = "images" if "images" in source else "image"
        if image_key in source:
            if isinstance(source[image_key], list):
                image_sources = source[image_key]
            elif isinstance(source[image_key], str):
                image_sources = [source[image_key]]
            else:
                raise ValueError(f"Invalid image source type: {type(source[image_key])}")

            for image_path in image_sources:
                if self.image_folder is not None:
                    image_path = os.path.join(self.image_folder, image_path)

                images.append(
                    Image.open(image_path).convert("RGB")
                    if self.load_image else image_path
                )

        # 加载视频
        videos = []
        if "video" in source:
            if isinstance(source["video"], list):
                video_sources = source["video"]
            elif isinstance(source["video"], str):
                video_sources = [source["video"]]
            else:
                raise ValueError(f"Invalid video source type: {type(source['video'])}")

            num_frames = [self.num_frames] * len(video_sources)

            for video_path, cur_num_frames in zip(video_sources, num_frames):
                if self.video_folder is not None:
                    video_path = os.path.join(self.video_folder, video_path)

                container = av.open(video_path)
                total_frames = container.streams.video[0].frames
                indices = np.arange(0, total_frames, total_frames / cur_num_frames).astype(int)
                clip = read_video_pyav(container, indices)

                videos.append(clip)

        conversations = source.get("conversations", [])

        system_prompt = source.get("system_prompt", None)

        result = dict(
            images=images,
            videos=videos,
            conversations=conversations,
            system_prompt=system_prompt
        )

        if "parameters" in source:
            result["parameters"] = source["parameters"]
        if "prev_parameters" in source:
            result["prev_parameters"] = source["prev_parameters"]

        return result

    def compute_normalization_stats(self, max_samples: int = 200000):

        max_samples = min(max_samples, len(self.list_data_dict))
        all_params = []

        # 收集参数并检查长度一致性
        param_lengths = {}
        for i in range(max_samples):
            item = self.list_data_dict[i]
            if "parameters" in item:
                params = item["parameters"]
                param_len = len(params)

                # 统计参数长度
                if param_len not in param_lengths:
                    param_lengths[param_len] = 0
                param_lengths[param_len] += 1

                all_params.append(params)

        # 检查参数长度一致性
        if len(param_lengths) > 1:
            print(f"⚠️  WARNING: Found inconsistent parameter lengths in dataset!")
            print(f"   Parameter length distribution:")
            for length, count in sorted(param_lengths.items()):
                print(f"     - {length} params: {count} samples ({100*count/len(all_params):.2f}%)")

            # 找到最常见的参数长度
            most_common_length = max(param_lengths.items(), key=lambda x: x[1])[0]
            print(f"   Using most common length: {most_common_length}")
            print(f"   Filtering out {sum(1 for p in all_params if len(p) != most_common_length)} samples with different lengths")

            # 只保留最常见长度的参数
            all_params = [p for p in all_params if len(p) == most_common_length]

        if len(all_params) == 0:
            raise ValueError("No valid parameters found in dataset!")

        all_params = np.array(all_params)
        param_mean = all_params.mean(axis=0)
        param_std = all_params.std(axis=0)

        print(f"Normalization computed from {len(all_params)} samples")
        print(f"Parameter shape: {all_params.shape}")

        return param_mean, param_std