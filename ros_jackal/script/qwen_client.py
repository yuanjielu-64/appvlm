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
    ):
        """
        Args:
            qwen_url: Qwen服务的URL
            algorithm: 规划算法 (DWA/TEB/MPPI/DDP)
            timeout: 请求超时时间 (秒)
            auto_start: 是否自动启动Qwen服务
            qwen_script_path: qwen_server.py的路径
        """
        self.qwen_url = qwen_url
        self.algorithm = algorithm
        self.timeout = timeout
        self.qwen_process = None

        if auto_start:
            if not qwen_script_path:
                raise ValueError("auto_start=True requires qwen_script_path")
            self.start_qwen_service(qwen_script_path)

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

    def infer_from_path(
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
            推理结果字典，包含:
                - parameters: 参数字典
                - parameters_array: 参数数组
                - raw_output: 模型原始输出
                - inference_time: 推理耗时
        """
        try:
            payload = {
                "image_path": image_path,
                "linear_vel": linear_vel,
                "angular_vel": angular_vel,
                "algorithm": algorithm or self.algorithm
            }

            print(f"[DEBUG] Sending request to {self.qwen_url}/infer")
            print(f"[DEBUG] Image path: {image_path}")
            print(f"[DEBUG] Image exists: {os.path.exists(image_path)}")

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

    def infer_from_base64(
        self,
        image_base64: str,
        linear_vel: float = 0.0,
        angular_vel: float = 0.0,
        algorithm: Optional[str] = None
    ) -> Dict:
        """从Base64编码的图像推理"""
        try:
            payload = {
                "image_base64": image_base64,
                "linear_vel": linear_vel,
                "angular_vel": angular_vel,
                "algorithm": algorithm or self.algorithm
            }

            response = requests.post(
                f'{self.qwen_url}/infer',
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()

        except Exception as e:
            print(f"⚠ Qwen inference error: {e}")
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


# ============================================================
# 使用示例
# ============================================================

def example_basic_usage():
    """基础使用示例"""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)

    # 创建客户端 (假设服务已启动)
    client = QwenClient(
        qwen_url="http://localhost:5000",
        algorithm="DWA",
        timeout=10.0
    )

    # 推理
    image_path = "/path/to/your/navigation_scene.png"
    result = client.infer_from_path(
        image_path=image_path,
        linear_vel=0.5,
        angular_vel=0.1
    )

    if result:
        print(f"✓ Inference successful!")
        print(f"  Time: {result['inference_time']:.3f}s")
        print(f"  Parameters: {result['parameters']}")
        print(f"  Array: {result['parameters_array']}")


def example_auto_start():
    """自动启动服务示例"""
    print("=" * 60)
    print("Example 2: Auto-start Service")
    print("=" * 60)

    qwen_script = "/home/yuanjielu/robot_navigation/noetic/appvlm_ws/src/ros_jackal/script/qwen_server.py"

    # 自动启动服务
    client = QwenClient(
        qwen_url="http://localhost:5000",
        algorithm="DWA",
        auto_start=True,
        qwen_script_path=qwen_script
    )

    try:
        # 执行推理
        result = client.infer_from_path(
            image_path="/path/to/image.png",
            linear_vel=1.0,
            angular_vel=0.0
        )

        if result:
            params = client.get_parameters_array(result)
            print(f"Parameters: {params}")

    finally:
        # 关闭服务
        client.close()


def example_integration_with_env():
    """集成到ROS环境的示例"""
    print("=" * 60)
    print("Example 3: Integration with ROS Environment")
    print("=" * 60)

    class MockDWAEnv:
        """模拟的DWA环境"""

        def __init__(self):
            # 启动Qwen客户端
            self.qwen_client = QwenClient(
                qwen_url="http://localhost:5000",
                algorithm="DWA",
                timeout=120.0  # 首次推理需要更长时间
            )

            # 默认参数 (fallback)
            self.default_params = [0.5, 1.0, 8, 20, 0.8, 1.0, 0.3]

        def get_llm_action(self, image_path: str, linear_vel: float, angular_vel: float) -> np.ndarray:
            """从LLM获取规划器参数"""
            result = self.qwen_client.infer_from_path(
                image_path=image_path,
                linear_vel=linear_vel,
                angular_vel=angular_vel
            )

            if result and result.get('success'):
                return np.array(result['parameters_array'])
            else:
                print("⚠ Using default parameters")
                return np.array(self.default_params)

        def step(self):
            """环境step函数"""
            # 获取当前状态
            image_path = "/tmp/current_scene.png"
            linear_vel = 0.5
            angular_vel = 0.1

            # 获取LLM预测的参数
            params = self.get_llm_action(image_path, linear_vel, angular_vel)

            # 更新DWA规划器参数
            self.update_dwa_params(params)

            # 执行规划和控制
            # ...

        def update_dwa_params(self, params: np.ndarray):
            """更新DWA参数 (通过ROS dynamic_reconfigure)"""
            print(f"Updating DWA params: {params}")
            # 实际实现中，这里会调用ROS的dynamic_reconfigure

    # 使用示例
    env = MockDWAEnv()
    env.step()


def example_compare_algorithms():
    """对比不同算法的示例"""
    print("=" * 60)
    print("Example 4: Compare Algorithms")
    print("=" * 60)

    client = QwenClient(qwen_url="http://localhost:5000")

    # 获取支持的算法
    algorithms_info = client.list_algorithms()
    print(f"Supported algorithms: {algorithms_info.get('algorithms', [])}")

    image_path = "/path/to/scene.png"

    # 对比不同算法
    for alg in ['DWA', 'TEB', 'MPPI']:
        print(f"\n--- {alg} ---")
        result = client.infer_from_path(
            image_path=image_path,
            algorithm=alg
        )
        if result:
            print(f"Time: {result['inference_time']:.3f}s")
            print(f"Params: {result['parameters']}")


if __name__ == "__main__":
    # 运行示例
    # example_basic_usage()
    # example_auto_start()
    example_integration_with_env()
    # example_compare_algorithms()