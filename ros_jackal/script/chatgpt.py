import json
import os
from openai import OpenAI
import base64
import re
import time
import random
import fcntl  # 文件锁

PROMPT_TEMPLATE = (
    "You are a Clearpath Jackal Robot, the length is 0.508 m, and the width is 0.430 m. "
    "The robot primarily moves along the purple global path. Your task is to predict {number} {algorithm} planner parameters "
    "based on the given navigation scene image. The predicted parameters should help traditional planners "
    "achieve faster, safer robot navigation by improving path-following and obstacle-avoidance. "
    "Your current linear velocity is {linear_vel} (linear_vel), and your angular velocity is {angular_vel} (angular_vel)\n"
    "SCENE UNDERSTANDING: "
    "- The green line on the robot represents its current direction of movement (x-axis). "
    "- The blue line on the robot represents the y-axis. "
    "- Grid spacing: 1 meter. "
    "- Red points: Hokuyo laser scan data (obstacles). "
    "- Purple line: Global path to follow. "
    "- Yellow rectangle: Robot's current position and footprint\n"
    "- Task: Navigate safely along the path while avoiding obstacles. "
    "OUTPUT FORMAT: The output must be in strict JSON format with exactly the following fields:\n"
    "{output_format}"
)

ALGORITHM_PARAMS = {
    "DWA": {
        "max_vel_x": {"range": [0.2, 2.0], "type": "float", "description": "Forward velocity (m/s)"},
        "max_vel_theta": {"range": [0.314, 3.14], "type": "float", "description": "Angular velocity (rad/s)"},
        "vx_samples": {"range": [4, 12], "type": "int", "description": "Number of linear velocity samples"},
        "vtheta_samples": {"range": [8, 40], "type": "int", "description": "Number of angular velocity samples"},
        "path_distance_bias": {"range": [0.1, 1.5], "type": "float", "description": "Path following weight"},
        "goal_distance_bias": {"range": [0.1, 2.0], "type": "float", "description": "Goal seeking weight"},
        "inflation_radius": {"range": [0.1, 0.6], "type": "float", "description": "Inflation radius (m)"}
    },
    "TEB": {
        "max_vel_x": {"range": [0.2, 2.0], "type": "float", "description": "Forward velocity (m/s)"},
        "max_vel_x_backwards": {"range": [0.1, 0.7], "type": "float", "description": "Backward velocity (m/s)"},
        "max_vel_theta": {"range": [0.314, 3.14], "type": "float", "description": "Angular velocity (rad/s)"},
        "dt_ref": {"range": [0.1, 0.35], "type": "float", "description": "Desired temporal resolution (s)"},
        "min_obstacle_dist": {"range": [0.05, 0.2], "type": "float", "description": "Minimum distance to obstacles (m)"},
        "inflation_dist": {"range": [0.01, 0.2], "type": "float", "description": "Inflation distance (m)"},
        "inflation_radius": {"range": [0.1, 0.6], "type": "float", "description": "Inflation radius (m)"}
    },
    "MPPI": {
        "max_vel_x": {"range": [-0.5, 2.0], "type": "float", "description": "Forward velocity (m/s)"},
        "max_vel_theta": {"range": [0.314, 3.14], "type": "float", "description": "Angular velocity (rad/s)"},
        "nr_pairs": {"range": [400, 800], "type": "int", "description": "Number of rollout pairs"},
        "nr_steps": {"range": [20, 40], "type": "int", "description": "Number of prediction steps"},
        "linear_stddev": {"range": [0.05, 0.15], "type": "float", "description": "Linear velocity standard deviation"},
        "angular_stddev": {"range": [0.02, 0.15], "type": "float", "description": "Angular velocity standard deviation"},
        "lambda": {"range": [0.5, 5.0], "type": "float", "description": "Temperature parameter"},
        "inflation_radius": {"range": [0.1, 0.6], "type": "float", "description": "Inflation radius (m)"}
    },
    "DDP": {
        "max_vel_x": {"range": [0.0, 2.0], "type": "float", "description": "Forward velocity (m/s)"},
        "max_vel_theta": {"range": [0.314, 3.14], "type": "float", "description": "Angular velocity (rad/s)"},
        "nr_pairs": {"range": [400, 800], "type": "int", "description": "Number of rollout pairs"},
        "distance": {"range": [0.01, 0.2], "type": "float", "description": "Distance threshold (m)"},
        "robot_radius": {"range": [0.01, 0.05], "type": "float", "description": "Robot radius (m)"},
        "inflation_radius": {"range": [0.1, 0.6], "type": "float", "description": "Inflation radius (m)"}
    }
}

def generate_output_format(param_config):

    lines = ["{"]
    for param_name, param_info in param_config.items():
        param_type = "<int>" if param_info["type"] == "int" else "<float>"
        range_str = f"{param_info['range'][0]}–{param_info['range'][1]}"
        line = f'  "{param_name}": {param_type},  // {param_info["description"]}, range: {range_str}'
        lines.append(line)
    lines[-1] = lines[-1].rstrip(',')
    lines.append("}")
    return "\n".join(lines)


class ChatgptEvaluator:
    def __init__(self, api_key = None, model = "gpt-4o", img_dir = None, alg = "DWA", init_params = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("需要提供 API key")

        self.img_dir = img_dir
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.img_id = 0
        self.algorithm = alg
        self.init_params = init_params

        self.param_config = ALGORITHM_PARAMS[self.algorithm]
        self.output_format = generate_output_format(self.param_config)
        self.param_order = list(self.param_config.keys())

        # API调用锁文件路径
        self.lock_file_path = "/tmp/openai_api_lock"


    def encode_image(self, image_path):

        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def get_current_image_path(self):

        image_name = f"VLM_{self.img_id:06d}.png"  # 格式化为 6 位数字
        return os.path.join(self.img_dir, image_name)

    def build_prompt(self, linear_vel, angular_vel):

        prompt = PROMPT_TEMPLATE.format(
            number=len(self.param_config),
            algorithm=self.algorithm,
            linear_vel=round(linear_vel, 4),
            angular_vel=round(angular_vel, 4),
            output_format=self.output_format
        )
        return prompt

    def parse_result_to_array(self, result):
        """将 VLM 结果转换为参数数组"""
        try:
            # 去除 markdown 标记
            cleaned = result.strip()
            if cleaned.startswith('```'):
                match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', cleaned, re.DOTALL)
                if match:
                    cleaned = match.group(1)
                else:
                    cleaned = re.sub(r'^```(?:json)?|```$', '', cleaned, flags=re.MULTILINE).strip()

            # 解析 JSON
            params = json.loads(cleaned)

            # 按顺序提取值
            param_array = [params[key] for key in self.param_order if key in params]

            return param_array

        except Exception as e:
            print(f"[ERROR] Parse failed: {e}")
            return self.init_params

    def evaluate_single(self, linear_vel = 0.0, angular_vel = 0.0):

        try:

            image_path = self.get_current_image_path()

            if not os.path.exists(image_path):
                print(f"[WARNING] Image not found: {image_path}")
                self.img_id += 1
                return None

            prompt = self.build_prompt(linear_vel, angular_vel)

            base64_image = self.encode_image(image_path)

            messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                    },
                    {"type": "text", "text": prompt}
                ]
            }]

            # 使用文件锁确保同一时间只有一个进程调用API
            lock_file = open(self.lock_file_path, 'w')

            try:
                # 尝试获取锁（最多等待30秒）
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

                # 获得锁后，随机等待0-1秒避免完全同步
                time.sleep(random.uniform(0.1, 0.5))

                # 重试机制：处理速率限制
                max_retries = 5
                base_wait = 2.0

                for attempt in range(max_retries):
                    try:
                        response = self.client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            max_tokens=500,
                            temperature=0.0
                        )

                        result = response.choices[0].message.content
                        self.img_id += 1
                        param_array = self.parse_result_to_array(result)

                        # API调用成功，添加小延迟避免下次调用太快
                        time.sleep(0.5)
                        return param_array

                    except Exception as e:
                        error_str = str(e)

                        # 检查是否是速率限制错误
                        if "rate_limit" in error_str.lower() or "429" in error_str:
                            if attempt < max_retries - 1:
                                # 指数退避 + 随机抖动
                                wait_time = base_wait * (2 ** attempt) + random.uniform(0, 1)
                                print(f"[RATE LIMIT] Attempt {attempt + 1}/{max_retries} failed. "
                                      f"Waiting {wait_time:.2f}s before retry...")
                                time.sleep(wait_time)
                            else:
                                print(f"[ERROR] Max retries reached. Using default params.")
                                self.img_id += 1
                                return self.init_params
                        else:
                            print(f"[ERROR] API call failed: {type(e).__name__}: {e}")
                            import traceback
                            traceback.print_exc()
                            self.img_id += 1
                            return self.init_params

            finally:

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()

        except Exception as e:
            print(f"[ERROR] VLM evaluation failed: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            self.img_id += 1
            return self.init_params

    def reset(self):
        # clean
        self.img_id = 0
