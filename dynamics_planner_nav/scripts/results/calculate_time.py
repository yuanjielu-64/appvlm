import numpy as np


def calculate_average_time(filename):
    data = np.loadtxt(filename)

    success_flag = data[:, 1]  # 第2列：1=成功，0=失败
    collision_flag = data[:, 2]  # 第3列：collision标志
    timeout_flag = data[:, 3]  # 第4列：timeout标志
    time_col = data[:, 4]  # 第5列：时间
    nav_metric_col = data[:, -1]  # 最后一列：导航指标

    # 失败的设为50秒，成功的用实际时间
    processed_time = np.where(success_flag == 0, 50.0, time_col)

    # 计算各种统计值
    average_time = np.mean(processed_time)
    total_count = len(data)
    success_count = np.sum(success_flag == 1)
    fail_count = np.sum(success_flag == 0)
    collision_count = np.sum(collision_flag == 1)
    timeout_count = np.sum(timeout_flag == 1)

    success_rate = (success_count / total_count) * 100
    collision_rate = (collision_count / total_count) * 100
    timeout_rate = (timeout_count / total_count) * 100
    avg_nav_metric = np.mean(nav_metric_col)

    # 输出结果
    print(f"总数据点: {total_count}")
    print(f"成功次数: {success_count}")
    print(f"失败次数: {fail_count}")
    print(f"碰撞次数: {collision_count}")
    print(f"超时次数: {timeout_count}")
    print(f"成功率: {success_rate:.2f}%")
    print(f"碰撞率: {collision_rate:.2f}%")
    print(f"超时率: {timeout_rate:.2f}%")
    print(f"平均时间: {average_time:.4f}秒")
    print(f"导航指标平均值: {avg_nav_metric:.4f}")

    return {
        'average_time': average_time,
        'success_rate': success_rate,
        'collision_rate': collision_rate,
        'timeout_rate': timeout_rate,
        'avg_nav_metric': avg_nav_metric,
        'total_count': total_count,
        'success_count': success_count,
        'fail_count': fail_count,
        'collision_count': collision_count,
        'timeout_count': timeout_count
    }


# # 使用示例
# result = calculate_average_time('out.txt')
# print(
#     f"\n总结: 成功率 {result['success_rate']:.2f}%, 碰撞率 {result['collision_rate']:.2f}%, 超时率 {result['timeout_rate']:.2f}%, 平均时间 {result['average_time']:.2f}s, 导航指标 {result['avg_nav_metric']:.3f}")


def calculate_scores_for_environments(filename, env_ids):
    data = np.loadtxt(filename)


    env_id_col = data[:, 0]  # 第1列是环境ID
    mask = np.isin(env_id_col, env_ids)
    filtered_data = data[mask]

    if len(filtered_data) == 0:
        print("没有找到匹配的环境ID数据")
        return None


    success_flag = filtered_data[:, 1]  # 成功标志
    collision_flag = filtered_data[:, 2]  # 碰撞标志
    timeout_flag = filtered_data[:, 3]  # 超时标志
    time_col = filtered_data[:, 4]  # 时间
    nav_metric_col = filtered_data[:, -1]  # 导航指标

    # 失败的设为50秒
    processed_time = np.where(success_flag == 0, 50.0, time_col)

    # 计算统计值
    total_count = len(filtered_data)
    success_count = np.sum(success_flag == 1)
    fail_count = np.sum(success_flag == 0)
    collision_count = np.sum(collision_flag == 1)
    timeout_count = np.sum(timeout_flag == 1)

    success_rate = (success_count / total_count) * 100
    collision_rate = (collision_count / total_count) * 100
    timeout_rate = (timeout_count / total_count) * 100
    average_time = np.mean(processed_time)
    avg_nav_metric = np.mean(nav_metric_col)

    # 输出结果
    print(f"指定环境数量: {len(env_ids)}")
    print(f"找到的数据点: {total_count}")
    print(f"成功次数: {success_count}")
    print(f"失败次数: {fail_count}")
    print(f"碰撞次数: {collision_count}")
    print(f"超时次数: {timeout_count}")
    print(f"成功率: {success_rate:.2f}%")
    print(f"碰撞率: {collision_rate:.2f}%")
    print(f"超时率: {timeout_rate:.2f}%")
    print(f"平均时间: {average_time:.4f}秒")
    print(f"导航指标平均值: {avg_nav_metric:.4f}")

    return {
        'success_rate': success_rate,
        'collision_rate': collision_rate,
        'timeout_rate': timeout_rate,
        'average_time': average_time,
        'avg_nav_metric': avg_nav_metric,
        'total_count': total_count
    }


# 你指定的环境ID数组
#target_envs = [163, 278, 58, 275, 123, 282, 294, 19, 283, 264, 245, 176, 271, 299, 225, 197, 111, 284, 99, 2, 209, 243, 182, 219, 260, 276, 285, 265, 117, 288, 255, 280, 287, 168, 138, 231, 180, 207, 222, 214, 273, 281, 254, 137, 85, 216, 292, 262, 199, 167]
target_envs = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 50, 51, 52, 53, 54, 55, 56, 57, 60, 61, 62, 63, 64, 65, 66, 67, 68, 70, 71, 72, 73, 74, 75, 76, 77, 79, 80, 81, 82, 83, 84, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 113, 114, 115, 116, 117, 120, 122, 124, 125, 126, 127, 128, 129, 130, 131, 133, 134, 135, 136, 137, 139, 141, 142, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 158, 159, 160, 161, 164, 165, 166, 167, 168, 169, 170, 173, 174, 176, 177, 178, 179, 182, 183, 184, 187, 188, 189, 190, 191, 193, 194, 195, 196, 198, 199, 200, 201, 202, 203, 205, 207, 208, 213, 214, 218, 219, 220, 222, 223, 224, 225, 226, 228, 232, 233, 235, 236, 237, 238, 239, 242, 243, 246, 247, 248, 249, 251, 253, 255, 257, 258, 259, 261, 262, 263, 264, 267, 268, 269, 270, 272, 273, 274, 276, 278, 281, 283, 284, 285, 288, 290, 291, 292, 296, 297, 299]
# 计算指定环境的分数

result = calculate_scores_for_environments('out_ddp_v=1.5_ddp', target_envs)

if result:
    print(f"\n指定环境总结:")
    print(f"成功率: {result['success_rate']:.2f}%")
    print(f"碰撞率: {result['collision_rate']:.2f}%")
    print(f"超时率: {result['timeout_rate']:.2f}%")
    print(f"平均时间: {result['average_time']:.2f}秒")
    print(f"导航指标: {result['avg_nav_metric']:.3f}")