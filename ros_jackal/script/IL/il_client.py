"""
Transformer IL Server 客户端
用于从ROS环境调用IL推理服务
"""

import requests
import numpy as np
import base64
import time
import os
import re
from typing import List, Dict, Optional


class ILClient:
    def __init__(
        self,
        il_url: str = "http://localhost:6000",
        algorithm: str = "DWA",
        timeout: float = 30.0,
        num_history_frames: Optional[int] = None,
    ):
        """
        Args:
            il_url: IL服务的URL
            algorithm: 规划算法 (DWA/TEB/MPPI/DDP)
            timeout: 请求超时时间 (秒)
            num_history_frames: 历史帧数量（None则从服务端查询）
        """
        self.il_url = il_url
        self.algorithm = algorithm
        self.timeout = timeout
        self.num_history_frames = num_history_frames
        self.img_id = 0

        # 如果未指定 num_history_frames，从服务端查询
        if self.num_history_frames is None:
            self._fetch_server_config()

    def wait_for_service(self, timeout: float = 60):
        """等待服务就绪"""
        print(f"Waiting for IL service at {self.il_url}...")
        start = time.time()

        while time.time() - start < timeout:
            try:
                resp = requests.get(f'{self.il_url}/health', timeout=2)
                if resp.json()['status'] == 'ok':
                    print("IL service ready!")
                    return True
            except:
                time.sleep(1)

        raise TimeoutError(f"IL service failed to start within {timeout}s")

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
            image_path: 图像文件路径
            linear_vel: 当前线速度
            angular_vel: 当前角速度
            algorithm: 规划算法 (None则使用初始化时的算法)

        Returns:
            推理结果字典
        """
        try:
            if not os.path.exists(image_path):
                print(f"[ERROR] Image not found: {image_path}")
                return None

            # 编码当前图像
            image_base64 = self.encode_image(image_path)

            # 读取历史帧
            history_images_base64 = self._read_history_frames(image_path)

            payload = {
                "image_base64": image_base64,
                "history_images_base64": history_images_base64,
                "linear_vel": linear_vel,
                "angular_vel": angular_vel,
                "algorithm": algorithm or self.algorithm
            }

            response = requests.post(
                f'{self.il_url}/infer',
                json=payload,
                timeout=self.timeout
            )

            response.raise_for_status()
            result = response.json()

            return result

        except requests.exceptions.Timeout:
            print(f"IL timeout after {self.timeout}s")
            return None
        except requests.exceptions.RequestException as e:
            print(f"IL HTTP error: {e}")
            return None
        except Exception as e:
            print(f"IL inference error: {e}")
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

    def _fetch_server_config(self):
        """从服务端查询配置信息"""
        try:
            response = requests.get(f'{self.il_url}/config', timeout=5)
            response.raise_for_status()
            config = response.json()

            self.num_history_frames = config.get('num_history_frames', 2)
            print(f"[INFO] Fetched server config: num_history_frames={self.num_history_frames}")

        except Exception as e:
            print(f"[WARN] Failed to fetch server config: {e}")
            self.num_history_frames = 2

    def _parse_filename(self, filename: str):
        """解析文件名，提取方法名和帧号"""
        match = re.match(r'([A-Z]+)_(\d{6})\.png', filename)
        if match:
            return match.group(1), int(match.group(2))
        return None, None

    def _read_history_frames(self, image_path: str) -> List[str]:
        """读取历史帧并编码为base64列表"""
        history_base64_list = []

        if self.num_history_frames <= 0:
            return history_base64_list

        image_dir = os.path.dirname(image_path)
        image_filename = os.path.basename(image_path)
        method, frame_id = self._parse_filename(image_filename)

        if method is None or frame_id is None:
            return history_base64_list

        # 加载历史帧
        loaded_frames = {}
        for i in range(1, self.num_history_frames + 1):
            hist_frame_id = frame_id - i
            hist_filename = f"{method}_{hist_frame_id:06d}.png"
            hist_path = os.path.join(image_dir, hist_filename)

            if hist_frame_id >= 0 and os.path.exists(hist_path):
                try:
                    loaded_frames[i] = self.encode_image(hist_path)
                except Exception as e:
                    print(f"[WARN] Failed to load {hist_filename}: {e}")

        # 构建历史帧列表（用fallback策略）
        last_available = None
        for i in range(1, self.num_history_frames + 1):
            if i in loaded_frames:
                history_base64_list.append(loaded_frames[i])
                last_available = loaded_frames[i]
            else:
                if last_available is not None:
                    history_base64_list.append(last_available)
                else:
                    # 用当前帧填充
                    current_base64 = self.encode_image(image_path)
                    history_base64_list.append(current_base64)
                    last_available = current_base64

        return history_base64_list

    def close(self):
        """关闭客户端（预留接口）"""
        pass
