"""
Planner-specific configurations
每个 planner 的参数数量、名称和范围
范围来自 data_processing/csv_to_json/csv_to_json.py 的 ALGORITHM_PARAMS
"""

PLANNER_CONFIGS = {
    "dwa": {
        "num_params": 9,
        "param_names": [
            "max_vel_x",
            "max_vel_theta",
            "vx_samples",
            "vtheta_samples",
            "path_distance_bias",
            "goal_distance_bias",
            "inflation_radius",
            "next_linear_vel",
            "next_angular_vel"
        ],
        "param_ranges": [
            [0.2, 2.0],      # max_vel_x
            [0.314, 3.14],   # max_vel_theta
            [4, 12],         # vx_samples
            [8, 40],         # vtheta_samples
            [0.1, 1.5],      # path_distance_bias
            [0.1, 2.0],      # goal_distance_bias
            [0.1, 0.6],      # inflation_radius
            [-0.5, 2.0],     # next_linear_vel
            [-3.14, 3.14]    # next_angular_vel
        ],
        "param_weights": [
            1.0,   # max_vel_x (重要: 影响速度和效率)
            1.0,   # max_vel_theta (重要: 影响转向灵活性)
            0.75,   # vx_samples (不太重要: 采样数量)
            0.75,   # vtheta_samples (不太重要: 采样数量)
            0.75,   # path_distance_bias (中等: 路径跟踪)
            0.75,   # goal_distance_bias (中等: 目标导向)
            1.25,   # inflation_radius (重要: 安全性)
            1.0,   # next_linear_vel (非常重要: 输出目标)
            1.0    # next_angular_vel (非常重要: 输出目标)
        ]
    },
    "teb": {
        "num_params": 9,
        "param_names": [
            "max_vel_x",
            "max_vel_x_backwards",
            "max_vel_theta",
            "dt_ref",
            "min_obstacle_dist",
            "inflation_dist",
            "inflation_radius",
            "next_linear_vel",
            "next_angular_vel"
        ],
        "param_ranges": [
            [0.2, 2.0],      # max_vel_x
            [0.1, 0.7],      # max_vel_x_backwards
            [0.314, 3.14],   # max_vel_theta
            [0.1, 0.35],     # dt_ref
            [0.05, 0.2],     # min_obstacle_dist
            [0.01, 0.2],     # inflation_dist
            [0.1, 0.6],      # inflation_radius
            [-0.5, 2.0],     # next_linear_vel
            [-3.14, 3.14]    # next_angular_vel
        ],
        "param_weights": [
            1.25,   # max_vel_x (重要)
            1.0,   # max_vel_x_backwards (重要)
            0.75,   # max_vel_theta (重要)
            0.75,   # dt_ref (中等: 时间参考)
            1.0,   # min_obstacle_dist (重要: 安全性)
            1.25,   # inflation_dist (重要: 安全性)
            1.25,   # inflation_radius (重要: 安全性)
            1.0,   # next_linear_vel (非常重要: 输出目标)
            1.0    # next_angular_vel (非常重要: 输出目标)
        ]
    },
    "mppi": {
        "num_params": 10,
        "param_names": [
            "max_vel_x",
            "max_vel_theta",
            "nr_pairs_",
            "nr_steps_",
            "linear_stddev",
            "angular_stddev",
            "lambda",
            "inflation_radius",
            "next_linear_vel",
            "next_angular_vel"
        ],
        "param_ranges": [
            [-0.5, 2.0],     # max_vel_x (可以倒车)
            [0.314, 3.14],   # max_vel_theta
            [400, 800],      # nr_pairs_
            [20, 40],        # nr_steps_
            [0.05, 0.15],    # linear_stddev
            [0.02, 0.15],    # angular_stddev
            [0.5, 5.0],      # lambda
            [0.1, 0.6],      # inflation_radius
            [-0.5, 2.0],     # next_linear_vel
            [-3.14, 3.14]    # next_angular_vel
        ],
        "param_weights": [
            1.0,   # max_vel_x (重要)
            1.0,   # max_vel_theta (重要)
            0.75,   # nr_pairs_ (不太重要: 采样数量)
            0.75,   # nr_steps_ (不太重要: 采样步数)
            1.0,   # linear_stddev (中等: 探索)
            1.0,   # angular_stddev (中等: 探索)
            0.75,   # lambda (中等: 温度参数)
            1.25,   # inflation_radius (重要: 安全性)
            1.0,   # next_linear_vel (非常重要: 输出目标)
            1.0    # next_angular_vel (非常重要: 输出目标)
        ]
    },
    "ddp": {
        "num_params": 8,
        "param_names": [
            "max_vel_x",
            "max_vel_theta",
            "nr_pairs_",
            "distance",
            "robot_radius",
            "inflation_radius",
            "next_linear_vel",
            "next_angular_vel"
        ],
        "param_ranges": [
            [0.0, 2.0],      # max_vel_x
            [0.314, 3.14],   # max_vel_theta
            [400, 800],      # nr_pairs_
            [0.01, 0.2],     # distance
            [0.01, 0.05],    # robot_radius
            [0.1, 0.6],      # inflation_radius
            [-0.5, 2.0],     # next_linear_vel
            [-3.14, 3.14]    # next_angular_vel
        ],
        "param_weights": [
            1.0,   # max_vel_x (重要: 影响速度)
            1.0,   # max_vel_theta (重要: 影响转向)
            0.75,   # nr_pairs_ (不太重要: 采样数量)
            0.75,   # distance (重要: 轨迹距离)
            1.25,   # robot_radius (重要: 安全性)
            1.25,   # inflation_radius (重要: 安全性)
            1.0,   # next_linear_vel (非常重要: 输出目标)
            1.0    # next_angular_vel (非常重要: 输出目标)
        ]
    }
}

# 向后兼容
PLANNER_PARAMS = PLANNER_CONFIGS

def get_num_params(planner_name: str) -> int:
    """根据 planner 名称获取参数数量"""
    planner_name = planner_name.lower()
    if planner_name not in PLANNER_CONFIGS:
        raise ValueError(f"Unknown planner: {planner_name}. Supported: {list(PLANNER_CONFIGS.keys())}")
    return PLANNER_CONFIGS[planner_name]["num_params"]


def get_param_names(planner_name: str) -> list:
    """根据 planner 名称获取参数名称列表"""
    planner_name = planner_name.lower()
    if planner_name not in PLANNER_CONFIGS:
        raise ValueError(f"Unknown planner: {planner_name}. Supported: {list(PLANNER_CONFIGS.keys())}")
    return PLANNER_CONFIGS[planner_name]["param_names"]


def get_param_ranges(planner_name: str) -> list:
    """根据 planner 名称获取参数范围列表"""
    planner_name = planner_name.lower()
    if planner_name not in PLANNER_CONFIGS:
        raise ValueError(f"Unknown planner: {planner_name}. Supported: {list(PLANNER_CONFIGS.keys())}")
    return PLANNER_CONFIGS[planner_name]["param_ranges"]


def get_param_weights(planner_name: str) -> list:
    """根据 planner 名称获取参数权重列表"""
    planner_name = planner_name.lower()
    if planner_name not in PLANNER_CONFIGS:
        raise ValueError(f"Unknown planner: {planner_name}. Supported: {list(PLANNER_CONFIGS.keys())}")
    return PLANNER_CONFIGS[planner_name].get("param_weights", None)


if __name__ == "__main__":
    print("Planner Configurations:")
    print("=" * 60)
    for planner, config in PLANNER_PARAMS.items():
        print(f"{planner.upper()}: {config['num_params']} parameters")
        for i, name in enumerate(config['param_names'], 1):
            print(f"  {i}. {name}")
        print()
