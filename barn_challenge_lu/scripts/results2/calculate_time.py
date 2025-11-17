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


# 使用示例
result = calculate_average_time('dwa_1.0.txt')
print(
    f"\n总结: 成功率 {result['success_rate']:.2f}%, 碰撞率 {result['collision_rate']:.2f}%, 超时率 {result['timeout_rate']:.2f}%, 平均时间 {result['average_time']:.2f}s, 导航指标 {result['avg_nav_metric']:.3f}")