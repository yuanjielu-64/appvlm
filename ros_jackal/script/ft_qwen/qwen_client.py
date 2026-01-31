"""
Qwen Server 客户端示例
演示如何从ROS环境 (Python 3.8) 调用Qwen服务 (Python 3.10)
"""

import requests
import numpy as np
import base64
import time
from typing import List, Dict, Optional
import subprocess
import os

class QwenClient:
    def __init__(
        self,
        qwen_url: str = "http://localhost:5000",
        algorithm: str = "DWA",
        timeout: float = 120.0,
        auto_start: bool = False,
        qwen_script_path: Optional[str] = None,
        num_history_frames: Optional[int] = None,
    ):
        """
        Args:
            qwen_url: Qwen服务的URL
            algorithm: 规划算法 (DWA/TEB/MPPI/DDP)
            timeout: 请求超时时间 (秒)
            auto_start: 是否自动启动Qwen服务
            qwen_script_path: qwen_server.py的路径
            num_history_frames: 历史帧数量（None则从服务端查询）
        """
        self.qwen_url = qwen_url
        self.algorithm = algorithm
        self.timeout = timeout
        self.qwen_process = None
        self.num_history_frames = num_history_frames  # 可能是 None

        if auto_start:
            if not qwen_script_path:
                raise ValueError("auto_start=True requires qwen_script_path")
            self.start_qwen_service(qwen_script_path)

        # 如果未指定 num_history_frames，从服务端查询
        if self.num_history_frames is None:
            self._fetch_server_config()

    def start_qwen_service(self, qwen_script_path: str, **kwargs):

        conda_python = '/home/yuanjielu/miniforge3/envs/lmms-finetune-qwen/bin/python'

        cmd = [conda_python, qwen_script_path]

        if 'base_model' in kwargs:
            cmd.extend(['--base_model', kwargs['base_model']])
        if 'lora_path' in kwargs:
            cmd.extend(['--lora_path', kwargs['lora_path']])
        if 'port' in kwargs:
            cmd.extend(['--port', str(kwargs['port'])])
        if 'algorithm' in kwargs:
            cmd.extend(['--algorithm', kwargs['algorithm']])

        print(f"Starting Qwen service: {' '.join(cmd)}")
        self.qwen_process = subprocess.Popen(cmd)

        self.wait_for_service(timeout=60)

    def wait_for_service(self, timeout: float = 120):

        print(f"Waiting for Qwen service at {self.qwen_url}...")
        start = time.time()

        while time.time() - start < timeout:
            try:
                resp = requests.get(f'{self.qwen_url}/health', timeout=1)
                if resp.json()['status'] == 'ok':
                    print("✓ Qwen service ready!")
                    return True
            except:
                time.sleep(2)

        raise TimeoutError(f"Qwen service failed to start within {timeout}s")

    def infer_from_server(
        self,
        image_path: str,
        linear_vel: float = 0.0,
        angular_vel: float = 0.0,
        algorithm: Optional[str] = None
    ) -> Dict:
        """
        从图像路径推理

        Args:
            image_path: 图像文件路径（完整路径，例如 /path/to/actor_0/HB_000025.png）
            linear_vel: 当前线速度
            angular_vel: 当前角速度
            algorithm: 规划算法 (None则使用初始化时的算法)

        Returns:
            推理结果字典，包含:
                - parameters: 参数字典
                - parameters_array: 参数数组
                - raw_output: 模型原始输出
                - inference_time: 推理耗时
        """
        try:
            print(f"[DEBUG] Image path: {image_path}")
            print(f"[DEBUG] Image exists: {os.path.exists(image_path)}")

            if not os.path.exists(image_path):
                print(f"[DEBUG] ✗ Image file not found!")
                return None

            # 编码当前图像为 base64
            image_base64 = self.encode_image(image_path)
            print(f"[DEBUG] Current frame encoded to base64 ({len(image_base64)} bytes)")

            # 读取历史帧（客户端处理，无需服务端访问文件系统）
            history_images_base64 = self._read_history_frames(image_path)
            print(f"[DEBUG] History frames loaded: {len(history_images_base64)}")

            payload = {
                "image_base64": image_base64,    # 当前帧的图像数据
                "history_images_base64": history_images_base64,  # 历史帧列表 [frame-1, frame-2, ..., frame-N] (从新到旧)
                "linear_vel": linear_vel,
                "angular_vel": angular_vel,
                "algorithm": algorithm or self.algorithm
            }

            print(f"[DEBUG] Sending request to {self.qwen_url}/infer")

            response = requests.post(
                f'{self.qwen_url}/infer',
                json=payload,
                timeout=self.timeout
            )

            print(f"[DEBUG] Response status code: {response.status_code}")

            response.raise_for_status()
            result = response.json()

            print(f"[DEBUG] Response success: {result.get('success', False)}")
            if not result.get('success'):
                print(f"[DEBUG] Error message: {result.get('error', 'Unknown error')}")

            return result

        except requests.exceptions.Timeout:
            print(f"⚠ Qwen timeout after {self.timeout}s")
            print(f"[DEBUG] Image path was: {image_path}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"⚠ Qwen HTTP request error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"[DEBUG] Response text: {e.response.text}")
            return None
        except Exception as e:
            print(f"⚠ Qwen inference error: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def encode_image(self, image_path: str) -> str:
        """将图像文件编码为Base64"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def get_parameters_array(self, result: Dict) -> Optional[np.ndarray]:
        """从推理结果提取参数数组"""
        if result and result.get('success'):
            return np.array(result['parameters_array'])
        return None

    def list_algorithms(self) -> Dict:
        """获取支持的算法列表"""
        try:
            response = requests.get(f'{self.qwen_url}/algorithms', timeout=2)
            return response.json()
        except Exception as e:
            print(f"⚠ Failed to get algorithms: {e}")
            return {}

    def close(self):
        """关闭Qwen服务"""
        if self.qwen_process:
            print("Terminating Qwen service...")
            self.qwen_process.terminate()
            self.qwen_process.wait(timeout=5)

    def _fetch_server_config(self):
        """从服务端查询配置信息（特别是 num_history_frames）"""
        try:
            response = requests.get(f'{self.qwen_url}/config', timeout=5)
            response.raise_for_status()
            config = response.json()

            self.num_history_frames = config.get('num_history_frames', 0)
            print(f"[INFO] Fetched server config: num_history_frames={self.num_history_frames}")

        except requests.exceptions.RequestException as e:
            print(f"[WARN] Failed to fetch server config: {e}")
            print(f"[WARN] Using default num_history_frames=0 (no history)")
            self.num_history_frames = 0
        except Exception as e:
            print(f"[WARN] Unexpected error fetching config: {e}")
            self.num_history_frames = 0

    def _parse_filename(self, filename: str):
        """
        解析文件名，提取方法名和帧号

        Args:
            filename: 例如 "HB_000025.png" 或 "FTRL_0000000.png"

        Returns:
            (method, frame_id) 或 (None, None)
        """
        import re
        match = re.match(r'([A-Z]+)_(\d+)\.png', filename)  # 支持变长数字
        if match:
            return match.group(1), int(match.group(2))
        return None, None

    def _read_history_frames(self, image_path: str) -> List[str]:
        """
        读取历史帧并编码为base64列表

        顺序：从新到旧 [frame-1, frame-2, frame-3, frame-4]
        Fallback策略：如果frame-i不存在，用最近的可用帧填充

        Args:
            image_path: 当前帧路径，例如 /path/to/actor_0/HB_000025.png

        Returns:
            历史帧的base64编码列表，从新到旧 [frame-1, frame-2, ..., frame-N]
        """
        history_base64_list = []

        if self.num_history_frames <= 0:
            return history_base64_list

        # 解析文件名
        image_dir = os.path.dirname(image_path)
        image_filename = os.path.basename(image_path)
        method, frame_id = self._parse_filename(image_filename)

        if method is None or frame_id is None:
            print(f"[DEBUG] Cannot parse filename: {image_filename}, no history frames")
            return history_base64_list

        # 首先尝试加载所有历史帧，记录哪些成功加载
        loaded_frames = {}  # {offset: base64_string}

        for i in range(1, self.num_history_frames + 1):  # i=1,2,3,4
            hist_frame_id = frame_id - i  # 当前25: 读取 24, 23, 22, 21
            hist_filename = f"{method}_{hist_frame_id:06d}.png"
            hist_path = os.path.join(image_dir, hist_filename)

            # 尝试读取历史帧
            if hist_frame_id >= 0 and os.path.exists(hist_path):
                try:
                    hist_base64 = self.encode_image(hist_path)
                    loaded_frames[i] = hist_base64
                    print(f"[DEBUG] History frame-{i} loaded: {hist_filename}")
                except Exception as e:
                    print(f"[DEBUG] Failed to load {hist_filename}: {e}")
            else:
                print(f"[DEBUG] History frame-{i} not found: {hist_filename}")

        # 构建最终的历史帧列表，按从新到旧的顺序
        last_available_base64 = None  # 记录最近的可用帧

        for i in range(1, self.num_history_frames + 1):  # i=1,2,3,4
            if i in loaded_frames:
                # 该帧存在，使用它
                history_base64_list.append(loaded_frames[i])
                last_available_base64 = loaded_frames[i]
                print(f"[DEBUG] Using frame-{i} (exists)")
            else:
                # 该帧不存在，使用最近的可用帧填充
                if last_available_base64 is not None:
                    history_base64_list.append(last_available_base64)
                    print(f"[DEBUG] Using frame-{i} (fallback to most recent available)")
                else:
                    # 如果连frame-1都不存在，用当前帧填充
                    current_image_base64 = self.encode_image(image_path)
                    history_base64_list.append(current_image_base64)
                    last_available_base64 = current_image_base64
                    print(f"[DEBUG] Using frame-{i} (fallback to current frame)")

        return history_base64_list


