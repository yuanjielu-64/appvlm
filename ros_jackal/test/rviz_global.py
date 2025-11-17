import rospy
import cv2
import numpy as np
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Path
import tf
import math
import time


class SceneCapture:
    def __init__(self, save_dir, target_resolution=0.01, laser_size_m=0.05,
                 robot_width=0.430, robot_length=0.508, path_width=0.01,
                 crop_size_m=10.0):
        self.save_dir = save_dir
        self.target_resolution = target_resolution
        self.laser_size_m = laser_size_m
        self.robot_width = robot_width
        self.robot_length = robot_length
        self.path_width = path_width
        self.crop_size_m = crop_size_m
        self.frame_id = 0
        self.costmap = None
        self.laser = None
        self.global_path = None
        self.tf_listener = tf.TransformListener()

        self.total_time = 0.0
        self.frame_count = 0

        rospy.Subscriber('/move_base/global_costmap/costmap',
                         OccupancyGrid, self.costmap_callback)
        rospy.Subscriber('/front/scan',
                         LaserScan, self.laser_callback)
        rospy.Subscriber('/move_base/NavfnROS/plan',
                         Path, self.path_callback)

        rospy.Timer(rospy.Duration(0.1), self.save_frame)

    def costmap_callback(self, msg):
        self.costmap = msg

    def laser_callback(self, msg):
        self.laser = msg

    def path_callback(self, msg):
        self.global_path = msg

    def draw_robot_rectangle(self, img, center_x, center_y, yaw):
        """画一个黄色矩形代表机器人"""
        robot_width_px = int(self.robot_width / self.target_resolution)
        robot_length_px = int(self.robot_length / self.target_resolution)

        half_w = robot_width_px / 2
        half_l = robot_length_px / 2

        corners = np.array([
            [-half_l, -half_w],
            [half_l, -half_w],
            [half_l, half_w],
            [-half_l, half_w]
        ], dtype=np.float32)

        yaw_adjusted = yaw + math.pi / 2

        rotation_matrix = np.array([
            [math.cos(yaw_adjusted), -math.sin(yaw_adjusted)],
            [math.sin(yaw_adjusted), math.cos(yaw_adjusted)]
        ])

        rotated_corners = corners @ rotation_matrix.T
        rotated_corners[:, 1] = -rotated_corners[:, 1]
        rotated_corners[:, 0] += center_x
        rotated_corners[:, 1] += center_y

        points = rotated_corners.astype(np.int32)

        cv2.fillPoly(img, [points], (0, 255, 255))
        cv2.polylines(img, [points], True, (0, 0, 0), 2)

    def draw_global_path(self, img, robot_center):
        """绘制全局路径（简化版，直接在 base_link 坐标系）"""
        if self.global_path is None or len(self.global_path.poses) == 0:
            return

        path_width_px = max(1, int(self.path_width / self.target_resolution))

        try:
            (trans, rot) = self.tf_listener.lookupTransform(
                'odom', 'base_link', rospy.Time(0))
            robot_x_odom = trans[0]
            robot_y_odom = trans[1]

            from tf.transformations import euler_from_quaternion
            _, _, robot_yaw = euler_from_quaternion(rot)
        except Exception as e:
            rospy.logwarn(f"Path TF lookup failed: {e}")
            return

        # 旋转矩阵（odom 到 base_link）
        cos_yaw = math.cos(-robot_yaw)
        sin_yaw = math.sin(-robot_yaw)

        points = []
        for pose_stamped in self.global_path.poses:
            # 路径点在 odom 中的位置
            px_odom = pose_stamped.pose.position.x
            py_odom = pose_stamped.pose.position.y

            # 相对于机器人
            rel_x_odom = px_odom - robot_x_odom
            rel_y_odom = py_odom - robot_y_odom

            # 转换到 base_link 坐标系
            rel_x_base = rel_x_odom * cos_yaw - rel_y_odom * sin_yaw
            rel_y_base = rel_x_odom * sin_yaw + rel_y_odom * cos_yaw

            # 转换到图像坐标
            img_x = int(robot_center + rel_x_base / self.target_resolution)
            img_y = int(robot_center - rel_y_base / self.target_resolution)

            points.append((img_x, img_y))

        # 绘制路径
        for i in range(len(points) - 1):
            cv2.line(img, points[i], points[i + 1], (255, 0, 255),
                     path_width_px, lineType=cv2.LINE_AA)

    def sample_rotated_region(self, data, width, height, resolution,
                              robot_x, robot_y, yaw, crop_size_m):
        """
        从 costmap 中采样一个旋转的正方形区域（aligned with base_link）
        优化版：使用向量化操作
        """
        crop_size_px = int(crop_size_m / self.target_resolution)

        # 创建输出图像
        output = np.full((crop_size_px, crop_size_px), -1, dtype=np.int8)

        # 旋转矩阵
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        # === 向量化版本：一次性计算所有像素 ===
        # 创建网格坐标
        out_x_indices = np.arange(crop_size_px)
        out_y_indices = np.arange(crop_size_px)
        out_x_grid, out_y_grid = np.meshgrid(out_x_indices, out_y_indices)

        # 转换到 base_link 坐标系（米）
        local_x = (out_x_grid - crop_size_px / 2) * self.target_resolution
        local_y = (crop_size_px / 2 - out_y_grid) * self.target_resolution

        # 转换到 odom 坐标系
        odom_x = robot_x + local_x * cos_yaw - local_y * sin_yaw
        odom_y = robot_y + local_x * sin_yaw + local_y * cos_yaw

        # 转换到 costmap 像素坐标
        map_px = ((odom_x - self.costmap.info.origin.position.x) / resolution).astype(np.int32)
        map_py = ((odom_y - self.costmap.info.origin.position.y) / resolution).astype(np.int32)

        # 边界检查
        valid_mask = (map_px >= 0) & (map_px < width) & (map_py >= 0) & (map_py < height)

        # 采样（只采样有效区域）
        output[valid_mask] = data[map_py[valid_mask], map_px[valid_mask]]

        return output

    def save_frame(self, event):
        if self.costmap is None or self.laser is None:
            return

        start_time = time.time()

        width = self.costmap.info.width
        height = self.costmap.info.height
        resolution = self.costmap.info.resolution

        # 获取机器人位置和朝向
        try:
            (trans, rot) = self.tf_listener.lookupTransform(
                self.costmap.header.frame_id, 'base_link', rospy.Time(0))
            robot_x = trans[0]
            robot_y = trans[1]
            from tf.transformations import euler_from_quaternion
            _, _, yaw = euler_from_quaternion(rot)
        except Exception as e:
            rospy.logwarn(f"TF lookup failed: {e}")
            return

        data = np.array(self.costmap.data).reshape((height, width))

        sample_start = time.time()
        data_aligned = self.sample_rotated_region(
            data, width, height, resolution, robot_x, robot_y, yaw, self.crop_size_m)
        sample_time = (time.time() - sample_start) * 1000

        crop_size_px = data_aligned.shape[0]

        img = np.zeros((crop_size_px, crop_size_px, 3), dtype=np.uint8)
        img[data_aligned == -1] = [128, 128, 128]
        img[data_aligned == 0] = [205, 205, 205]
        img[(data_aligned > 0) & (data_aligned < 99)] = [255, 180, 120]
        img[data_aligned >= 99] = [40, 40, 40]

        # 机器人在图像中心
        robot_center = crop_size_px // 2

        # 对齐到网格
        grid_spacing = int(1.0 / self.target_resolution)
        robot_px_aligned = round(robot_center / grid_spacing) * grid_spacing
        robot_py_aligned = round(robot_center / grid_spacing) * grid_spacing

        # === 绘制网格 ===
        max_offset = crop_size_px // grid_spacing + 1

        for i in range(-max_offset, max_offset + 1):
            x = robot_px_aligned + i * grid_spacing
            if 0 <= x < crop_size_px:
                cv2.line(img, (x, 0), (x, crop_size_px - 1), (120, 120, 120), 1)

        for i in range(-max_offset, max_offset + 1):
            y = robot_py_aligned + i * grid_spacing
            if 0 <= y < crop_size_px:
                cv2.line(img, (0, y), (crop_size_px - 1, y), (120, 120, 120), 1)

        # === 绘制激光点 ===
        laser_radius = max(1, int(self.laser_size_m / self.target_resolution))
        angle = self.laser.angle_min

        for r in self.laser.ranges:
            if r < self.laser.range_min or r > self.laser.range_max or np.isinf(r):
                angle += self.laser.angle_increment
                continue

            x = r * math.cos(angle)
            y = r * math.sin(angle)

            px = int(robot_px_aligned + x / self.target_resolution)
            py = int(robot_py_aligned - y / self.target_resolution)

            if 0 <= px < crop_size_px and 0 <= py < crop_size_px:
                cv2.circle(img, (px, py), laser_radius, (0, 0, 255), -1)

            angle += self.laser.angle_increment

        # === 绘制 Global Path ===
        self.draw_global_path(img, robot_center)

        # === 绘制坐标轴 ===
        axis_length = int(1.0 / self.target_resolution)
        axis_width = max(3, int(0.05 / self.target_resolution))

        cv2.line(img, (robot_px_aligned, robot_py_aligned),
                 (robot_px_aligned + axis_length, robot_py_aligned),
                 (0, 0, 255), axis_width, lineType=cv2.LINE_AA)

        cv2.line(img, (robot_px_aligned, robot_py_aligned),
                 (robot_px_aligned, robot_py_aligned - axis_length),
                 (0, 255, 0), axis_width, lineType=cv2.LINE_AA)

        # === 画机器人 ===
        self.draw_robot_rectangle(img, robot_px_aligned, robot_py_aligned, -math.pi / 2)

        cv2.circle(img, (robot_px_aligned, robot_py_aligned), 5, (255, 255, 255), -1)

        # 保存
        cv2.imwrite(f'{self.save_dir}/frame_{self.frame_id:06d}.png', img)

        elapsed_time = time.time() - start_time
        self.total_time += elapsed_time
        self.frame_count += 1

        rospy.loginfo(f"帧 {self.frame_id} | 总:{elapsed_time * 1000:.1f}ms | 采样:{sample_time:.1f}ms")

        self.frame_id += 1


# 使用
rospy.init_node('scene_capture')
capture = SceneCapture('/home/yuanjielu/Desktop',
                       target_resolution=0.01,
                       laser_size_m=0.05,
                       robot_width=0.430,
                       robot_length=0.508,
                       path_width=0.01,
                       crop_size_m=10.0)
rospy.spin()