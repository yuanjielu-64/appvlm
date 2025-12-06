import csv
import json
from pathlib import Path

# ===== 路径自行修改 =====
# 统一根目录，下面有 actor_0, actor_1, ... actor_299
root_dir = Path("/scratch/bwang25/appvlm/buffer/dwa_heurstic")
# 每个子目录中的 csv 文件名。如果不确定可设为 None 用 glob 找 *.csv
csv_name = "data.csv"
data_trajectory = "data_trajectory.csv"
# =======================

# 原始 prompt 模板（仅把速度位置留成占位符）
PROMPT_TEMPLATE = (
    "You are a Clearpath Jackal Robot, the length is 0.508 m, and the width is 0.430 m. "
    "The robot primarily moves along the purple global path. Your task is to predict six DWA planner parameters "
    "based on the given navigation scene image. The predicted parameters should help traditional planners "
    "(e.g., DWA, TEB) achieve faster, safer robot navigation by improving path-following and obstacle-avoidance. "
    "Your current linear velocity is {linear_vel} (linear_vel), and your angular velocity is {angular_vel} (angular_vel)\n"
    "SCENE UNDERSTANDING: "
    "- The blue line on the robot represents its current direction of movement (x-axis). "
    "- The green line on the robot represents the y-axis. "
    "- The blue square represents the global goal. "
    "- The green square represents the local goal. "
    "- Grid spacing: 1 meter. "
    "- Red points: Hokuyo laser scan data (obstacles). "
    "- Purple line: Global path to follow. "
    "- Yellow object: Robot current position. "
    "- Task: Navigate safely along the path while avoiding obstacles. "
    "OUTPUT FORMAT: The output must be in strict JSON format with exactly six fields: "
    "{{   \"max_vel_x\": <float>,        // Forward velocity (m/s), range: 0.2–2   "
    "\"max_vel_theta\": <float>,    // Angular velocity (rad/s), range: 0.314–3.14   "
    "\"vx_samples\": <float>,       // Number of linear velocity samples, integer, range: 4–12   "
    "\"path_distance_bias\": <float>, // Path following weight, range: 0.1–1.5   "
    "\"goal_distance_bias\": <float>,  // Goal seeking weight, range: 0.1–2   "
    "\"final_inflation\": <float> //  increase or decrease the inflation radius, range:  [-0.1, 0.1] }}"
)


def csv_to_json(input_csv_path, output_json_path):
    data = []

    csv_path = Path(input_csv_path)
    actor_name = csv_path.parent.name  # e.g., actor_0

    # 先按 Method 分组并按 img_label 排序，方便取“下一帧”的动作
    grouped_rows = {}
    with open(input_csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row.get("Method", "").strip()
            if not method:
                print(f"[SKIP] Missing Method for img_label={row.get('img_label')}")
                continue
            grouped_rows.setdefault(method, []).append(row)

    for method, rows in grouped_rows.items():
        rows_sorted = sorted(rows, key=lambda r: int(r["img_label"]))
        for idx, row in enumerate(rows_sorted[:-1]):  # 最后一帧没有下一帧动作，舍弃
            next_row = rows_sorted[idx + 1]

            img_label = int(row["img_label"])
            sample_id = f"{method}_{img_label:06d}"

            image_filename = f"{method}_{img_label:06d}.png"
            image_rel_path = str(Path(actor_name) / image_filename)

            linear_vel = row["linear_vel"]
            angular_vel = row["angular_vel"]
            prompt_text = PROMPT_TEMPLATE.format(
                linear_vel=linear_vel,
                angular_vel=angular_vel
            )
            human_value = "<image>\n" + prompt_text

            # 动作取自“下一帧”数据
            answer_obj = {
                "max_vel_x": float(next_row["max_vel_x"]),
                "max_vel_theta": float(next_row["max_vel_theta"]),
                "vx_samples": int(next_row["vx_samples"]),
                "path_distance_bias": float(next_row["path_distance_bias"]),
                "goal_distance_bias": float(next_row["goal_distance_bias"]),
                "final_inflation": float(next_row["final_inflation"]),
            }
            gpt_value = json.dumps(answer_obj, ensure_ascii=False)

            entry = {
                "id": sample_id,
                "image": image_rel_path,
                "conversations": [
                    {
                        "from": "human",
                        "value": human_value,
                    },
                    {
                        "from": "gpt",
                        "value": gpt_value,
                    },
                ],
            }
            data.append(entry)

    # 写成一个大的 list 到 .json 里
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"Converted {len(data)} rows from CSV to JSON -> {output_json_path}")


if __name__ == "__main__":
    # 遍历 root_dir 下的所有 actor_* 文件夹
    if not root_dir.exists():
        raise FileNotFoundError(f"Root dir not found: {root_dir}")

    actor_dirs = sorted([p for p in root_dir.iterdir() if p.is_dir() and p.name.startswith("actor_")])
    if not actor_dirs:
        raise RuntimeError(f"No actor_* folders found under {root_dir}")

    for actor_dir in actor_dirs:
        if csv_name:
            csv_files = [actor_dir / csv_name] if (actor_dir / csv_name).exists() else []
        else:
            csv_files = sorted(actor_dir.glob("*.csv"))

        if not csv_files:
            print(f"[SKIP] No CSV found in {actor_dir}")
            continue

        for csv_file in csv_files:
            output_json = actor_dir / f"{actor_dir.name}.json"
            csv_to_json(str(csv_file), str(output_json))
