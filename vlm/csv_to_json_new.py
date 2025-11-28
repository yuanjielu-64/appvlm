import csv
import json
import os.path
from pathlib import Path
import argparse
import pandas as pd

# 原始 prompt 模板（仅把速度位置留成占位符）
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

# "{{   \"max_vel_x\": <float>,        // Forward velocity (m/s), range: 0.2–2   "
# "\"max_vel_theta\": <float>,    // Angular velocity (rad/s), range: 0.314–3.14   "
# "\"vx_samples\": <float>,       // Number of linear velocity samples, integer, range: 4–12   "
# "\"path_distance_bias\": <float>, // Path following weight, range: 0.1–1.5   "
# "\"goal_distance_bias\": <float>,  // Goal seeking weight, range: 0.1–2   "
# "\"final_inflation\": <float> //  increase or decrease the inflation radius, range:  [-0.1, 0.1] }}"

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

# =====================================================================================

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


def get_row_from_trajectory(data_trajectory):
    df = pd.read_csv(data_trajectory)

    numeric_cols = df.select_dtypes(include=['float64', 'float32']).columns
    df[numeric_cols] = df[numeric_cols].round(4)

    df = df.sort_values(by=['nav_metric', 'Time'], ascending=[False, True])
    df = df.reset_index(drop=True)

    duplicates = df['Start_frame_id'].duplicated(keep=False)
    if duplicates.any():
        dup_values = df[duplicates]['Start_frame_id'].unique()
        print(f"Warning: Duplicate Start_frame_id found: {dup_values}")
        df = df.drop_duplicates(subset='Start_frame_id', keep='last')
        df = df.reset_index(drop=True)

    total = len(df)
    idx_40 = int(total * 0.4)
    idx_60 = int(total * 0.6)
    idx_80 = int(total * 0.8)

    top_40 = df.iloc[:idx_40]
    sample_40_60 = df.iloc[idx_40:idx_60].sample(frac=0.5)
    sample_60_80 = df.iloc[idx_60:idx_80].sample(frac=0.2)
    sample_80_100 = df.iloc[idx_80:].sample(frac=0.05)

    result_df = pd.concat([top_40, sample_40_60, sample_60_80, sample_80_100])
    return result_df

def csv_to_json(input_csv_path, output_json_path, data_trajectory, param_config, alg):
    data = []

    df_filtered = get_row_from_trajectory(data_trajectory)
    csv_path = Path(input_csv_path)
    parent_dir = csv_path.parent
    actor_name = csv_path.parent.name

    df_full = pd.read_csv(input_csv_path)
    numeric_cols = df_full.select_dtypes(include=['float64', 'float32']).columns
    df_full[numeric_cols] = df_full[numeric_cols].round(4)
    df_full = df_full.drop_duplicates(subset=['Method', 'img_label'], keep='last')
    df_full = df_full.reset_index(drop=True)

    output_format = generate_output_format(param_config)

    grouped_filtered = df_filtered.groupby('Method')

    for method, group in grouped_filtered:

        df_method = df_full[df_full['Method'] == method].sort_values('img_label').reset_index(drop=True)

        for _, traj_row in group.iterrows():
            start = int(traj_row['Start_frame_id'])
            done = int(traj_row['Done_frame_id'])

            segment = df_method[(df_method['img_label'] >= start) & (df_method['img_label'] <= done)]

            for idx in range(len(segment) - 1):
                row = segment.iloc[idx]
                img_label = int(row["img_label"])
                sample_id = f"{method}_{img_label:06d}"

                image_filename = f"{method}_{img_label:06d}.png"
                image_rel_path = str(Path(actor_name) / image_filename)

                if not (parent_dir / image_filename).exists():
                    continue

                prompt_text = PROMPT_TEMPLATE.format(
                    algorithm=alg,
                    number=len(param_config),
                    linear_vel=row["linear_vel"],
                    angular_vel=row["angular_vel"],
                    output_format=output_format
                )
                human_value = "<image>\n" + prompt_text

                answer_obj = {}
                for param_name, param_info in param_config.items():
                    if param_name == "inflation_radius":
                        param_name = "final_inflation"

                    if param_name in row:
                        value = row[param_name]
                        if param_info["type"] == "int":
                            answer_obj[param_name] = int(value)
                        else:
                            answer_obj[param_name] = float(value)

                gpt_value = json.dumps(answer_obj, ensure_ascii=False)

                entry = {
                    "id": sample_id,
                    "image": image_rel_path,
                    "conversations": [
                        {"from": "human", "value": human_value},
                        {"from": "gpt", "value": gpt_value},
                    ],
                }
                data.append(entry)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"Converted {len(data)} rows -> {output_json_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Convert CSV to JSON for VLM training')
    parser.add_argument('--alg', default="dwa")
    parser.add_argument('--root_dir', default="../ros_jackal/buffer/dwa_heurstic")
    parser.add_argument('--csv_name', default="data.csv")
    parser.add_argument('--trajectory_name', default="data_trajectory.csv")
    args = parser.parse_args()

    alg_upper = args.alg.upper()

    if alg_upper not in ALGORITHM_PARAMS:
        raise ValueError(f"Unknown method: {args.alg}")

    param_config = ALGORITHM_PARAMS[alg_upper]

    root_dir = Path(args.root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"Root dir not found: {root_dir}")

    actor_dirs = sorted([p for p in root_dir.iterdir() if p.is_dir() and p.name.startswith("actor_")])

    for actor_dir in actor_dirs:
        csv_files = [actor_dir / args.csv_name] if (actor_dir / args.csv_name).exists() else []
        data_files = actor_dir / args.trajectory_name

        if not csv_files or not data_files.exists():
            print(f"[SKIP] {actor_dir}")
            continue

        for csv_file in csv_files:
            output_json = actor_dir / f"{actor_dir.name}.json"
            csv_to_json(str(csv_file), str(output_json), str(data_files), param_config, alg_upper)
