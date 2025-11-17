import rospy
import cv2
import numpy as np
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
import tf
import math
import time
from nav_msgs.msg import Path

class SceneCapture:
    def __init__(self, save_dir, target_resolution=0.05, laser_size_m=0.05,
                 robot_width=0.430, robot_length=0.508, path_width=0.01):
        self.save_dir = save_dir
        self.target_resolution = target_resolution
        self.laser_size_m = laser_size_m
        self.robot_width = robot_width
        self.robot_length = robot_length
        self.path_width = path_width
        self.frame_id = 0
        self.costmap = None
        self.laser = None
        self.tf_listener = tf.TransformListener()
        self.global_path = None

        self.total_time = 0.0
        self.frame_count = 0

        rospy.Subscriber('/move_base/local_costmap/costmap',
                         OccupancyGrid, self.costmap_callback)
        rospy.Subscriber('/front/scan',
                         LaserScan, self.laser_callback)

        rospy.Subscriber('/move_base/NavfnROS/plan',  # 或者你的global path话题
                         Path, self.path_callback)

        rospy.Timer(rospy.Duration(0.1), self.save_frame)

    def costmap_callback(self, msg):
        self.costmap = msg

    def laser_callback(self, msg):
        self.laser = msg

    def path_callback(self, msg):
        self.global_path = msg

    def draw_global_path(self, img, robot_px_canvas, robot_py_canvas, yaw,
                         rotation_angle, diagonal):
        """绘制全局路径"""
        if self.global_path is None or len(self.global_path.poses) == 0:
            return

        path_width_px = max(1, int(self.path_width / self.target_resolution))

        # 获取机器人当前位置
        try:
            (trans, rot) = self.tf_listener.lookupTransform(
                self.global_path.header.frame_id, 'base_link', rospy.Time(0))
            robot_x = trans[0]
            robot_y = trans[1]
        except:
            return

        # 将路径点转换到图像坐标
        points = []
        for pose_stamped in self.global_path.poses:
            px = pose_stamped.pose.position.x
            py = pose_stamped.pose.position.y

            # 相对于机器人的位置
            rel_x = px - robot_x
            rel_y = py - robot_y

            # 转换到像素坐标（旋转前）
            img_x = robot_px_canvas + rel_x / self.target_resolution
            img_y = robot_py_canvas - rel_y / self.target_resolution

            points.append([img_x, img_y])

        if len(points) < 2:
            return

        # 应用旋转变换
        points = np.array(points, dtype=np.float32)
        ones = np.ones((points.shape[0], 1))
        points_homogeneous = np.hstack([points, ones])

        # 获取旋转矩阵
        M = cv2.getRotationMatrix2D((robot_px_canvas, robot_py_canvas), rotation_angle, 1.0)

        # 变换所有点
        transformed_points = (M @ points_homogeneous.T).T
        transformed_points = transformed_points.astype(np.int32)

        # 绘制路径
        for i in range(len(transformed_points) - 1):
            pt1 = tuple(transformed_points[i])
            pt2 = tuple(transformed_points[i + 1])
            cv2.line(img, pt1, pt2, (255, 0, 255), path_width_px, lineType=cv2.LINE_AA)  # 紫色

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

    def draw_global_path(self, img, robot_px_canvas, robot_py_canvas, yaw,
                         rotation_angle, diagonal):
        """绘制全局路径"""
        if self.global_path is None or len(self.global_path.poses) == 0:
            return

        path_width_px = max(1, int(self.path_width / self.target_resolution))

        # 获取机器人在 odom 中的位置
        try:
            (trans, rot) = self.tf_listener.lookupTransform(
                'odom', 'base_link', rospy.Time(0))
            robot_x_odom = trans[0]
            robot_y_odom = trans[1]
        except Exception as e:
            rospy.logwarn(f"TF lookup failed: {e}")
            return

        # 将路径点转换到图像坐标
        points = []
        for pose_stamped in self.global_path.poses:
            px_odom = pose_stamped.pose.position.x
            py_odom = pose_stamped.pose.position.y

            rel_x = px_odom - robot_x_odom
            rel_y = py_odom - robot_y_odom

            # 旋转180度：X 和 Y 都取反
            img_x = robot_px_canvas - rel_x / self.target_resolution
            img_y = robot_py_canvas + rel_y / self.target_resolution

            points.append([img_x, img_y])

        if len(points) < 2:
            return

        # 应用旋转变换
        points = np.array(points, dtype=np.float32)
        ones = np.ones((points.shape[0], 1))
        points_homogeneous = np.hstack([points, ones])

        M = cv2.getRotationMatrix2D((robot_px_canvas, robot_py_canvas), rotation_angle, 1.0)

        transformed_points = (M @ points_homogeneous.T).T
        transformed_points = transformed_points.astype(np.int32)

        # 绘制路径
        for i in range(len(transformed_points) - 1):
            pt1 = tuple(transformed_points[i])
            pt2 = tuple(transformed_points[i + 1])
            cv2.line(img, pt1, pt2, (255, 0, 255), path_width_px, lineType=cv2.LINE_AA)

    def save_frame(self, event):
        if self.costmap is None or self.laser is None:
            return

        start_time = time.time()

        width = self.costmap.info.width
        height = self.costmap.info.height
        resolution = self.costmap.info.resolution

        # 获取朝向
        try:
            (trans, rot) = self.tf_listener.lookupTransform(
                'odom', 'base_link', rospy.Time(0))
            from tf.transformations import euler_from_quaternion
            _, _, yaw = euler_from_quaternion(rot)
        except:
            yaw = 0.0

        scale_factor = resolution / self.target_resolution
        target_width = int(width * scale_factor)
        target_height = int(height * scale_factor)

        robot_px = width // 2
        robot_py = height // 2

        # 绘制 costmap
        data = np.array(self.costmap.data).reshape((height, width))
        img = np.zeros((height, width, 3), dtype=np.uint8)

        img[data == -1] = [128, 128, 128]
        img[data == 0] = [205, 205, 205]
        img[(data > 0) & (data < 99)] = [255, 180, 120]
        img[data >= 99] = [40, 40, 40]

        img = cv2.flip(img, 0)
        robot_py_flipped = height - 1 - robot_py

        img_large = cv2.resize(img, (target_width, target_height),
                               interpolation=cv2.INTER_NEAREST)

        robot_px_scaled = int(robot_px * scale_factor)
        robot_py_scaled = int(robot_py_flipped * scale_factor)

        # 旋转整个图像
        rotation_angle = -math.degrees(yaw) - 90

        # 使用 local costmap 的实际大小
        diagonal = int(math.sqrt(target_width ** 2 + target_height ** 2)) + 100
        canvas = np.full((diagonal, diagonal, 3), 128, dtype=np.uint8)

        x_offset = (diagonal - target_width) // 2
        y_offset = (diagonal - target_height) // 2
        canvas[y_offset:y_offset + target_height, x_offset:x_offset + target_width] = img_large

        robot_px_canvas = robot_px_scaled + x_offset
        robot_py_canvas = robot_py_scaled + y_offset

        # === 绘制全局路径 ===
        self.draw_global_path(canvas, robot_px_canvas, robot_py_canvas, yaw,
                              rotation_angle, diagonal)

        M = cv2.getRotationMatrix2D((robot_px_canvas, robot_py_canvas), rotation_angle, 1.0)
        img_rotated = cv2.warpAffine(canvas, M, (diagonal, diagonal))

        # === 裁剪回 local costmap 的原始大小（取较小边）===
        crop_size = min(target_width, target_height)

        x1 = robot_px_canvas - crop_size // 2
        y1 = robot_py_canvas - crop_size // 2
        x2 = x1 + crop_size
        y2 = y1 + crop_size

        img_final = img_rotated[y1:y2, x1:x2]

        robot_center = crop_size // 2

        grid_spacing = int(1.0 / self.target_resolution)
        robot_px_aligned = round(robot_center / grid_spacing) * grid_spacing
        robot_py_aligned = round(robot_center / grid_spacing) * grid_spacing

        # 绘制网格
        max_offset = crop_size // grid_spacing + 1

        for i in range(-max_offset, max_offset + 1):
            x = robot_px_aligned + i * grid_spacing
            if 0 <= x < crop_size:
                cv2.line(img_final, (x, 0), (x, crop_size - 1), (120, 120, 120), 1)

        for i in range(-max_offset, max_offset + 1):
            y = robot_py_aligned + i * grid_spacing
            if 0 <= y < crop_size:
                cv2.line(img_final, (0, y), (crop_size - 1, y), (120, 120, 120), 1)

        # 绘制激光点
        laser_radius = max(1, int(self.laser_size_m / self.target_resolution))
        angle = self.laser.angle_min - math.pi / 2

        for r in self.laser.ranges:
            if r < self.laser.range_min or r > self.laser.range_max or np.isinf(r):
                angle += self.laser.angle_increment
                continue

            x = r * math.cos(angle)
            y = r * math.sin(angle)
            px = int(robot_px_aligned + x / self.target_resolution)
            py = int(robot_py_aligned - y / self.target_resolution)

            if 0 <= px < crop_size and 0 <= py < crop_size:
                cv2.circle(img_final, (px, py), laser_radius, (0, 0, 255), -1)

            angle += self.laser.angle_increment

        # 绘制坐标轴
        axis_length = int(1.0 / self.target_resolution)
        axis_width = max(3, int(0.05 / self.target_resolution))

        cv2.line(img_final, (robot_px_aligned, robot_py_aligned),
                 (robot_px_aligned, robot_py_aligned + axis_length),
                 (0, 0, 255), axis_width, lineType=cv2.LINE_AA)

        cv2.line(img_final, (robot_px_aligned, robot_py_aligned),
                 (robot_px_aligned + axis_length, robot_py_aligned),
                 (0, 255, 0), axis_width, lineType=cv2.LINE_AA)

        self.draw_robot_rectangle(img_final, robot_px_aligned, robot_py_aligned, -math.pi)

        cv2.circle(img_final, (robot_px_aligned, robot_py_aligned), 5, (255, 255, 255), -1)

        # 整体逆时针旋转90°
        img_final = cv2.rotate(img_final, cv2.ROTATE_90_COUNTERCLOCKWISE)

        cv2.imwrite(f'{self.save_dir}/frame_{self.frame_id:06d}.png', img_final)

        elapsed_time = time.time() - start_time
        self.total_time += elapsed_time
        self.frame_count += 1

        rospy.loginfo(f"帧 {self.frame_id} | {elapsed_time * 1000:.2f}ms")

        self.frame_id += 1


# 使用
rospy.init_node('scene_capture')
capture = SceneCapture('/home/yuanjielu/Desktop',
                       target_resolution=0.01,
                       laser_size_m=0.05,
                       robot_width=0.430,
                       robot_length=0.508,
                        path_width=0.01)
rospy.spin()