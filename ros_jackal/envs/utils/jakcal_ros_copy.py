import numpy as np
import cv2
import math
import time
import os
import pandas as pd
from numpy.array_api import trunc
from sympy import false

try:
    import rospy
    from std_msgs.msg import Bool, Float64
    from sensor_msgs.msg import LaserScan
    from nav_msgs.msg import Odometry, OccupancyGrid, Path
    from std_msgs.msg import Float64MultiArray
    from geometry_msgs.msg import PointStamped
    from visualization_msgs.msg import Marker
except ModuleNotFoundError:
    pass


class JackalRos:
    def __init__(self, init_position, goal_position, use_move_base=False, img_dir=None):

        self.robot_state = {
            'x': 0.0,
            'y': 0.0,
            'z': 0.0,
            'theta': 0.0,
            'velocity': 0.0,
            'angular_velocity': 0.0
        }

        self.last_robot_state ={
            'x': 0.0,
            'y': 0.0,
            'z': 0.0,
            'theta': 0.0,
            'velocity': 0.0,
            'angular_velocity': 0.0
        }

        self.duration = 0

        self.start = False
        self.use_move_base = use_move_base
        self.init_postion = init_position
        self.local_goal = [0, 4]
        self.global_goal = goal_position
        self.laser_data = None
        self.bad_vel = 0
        self.vel_counter = 0
        self.img_dir = img_dir

        self.is_colliding = False
        self.collision_count = 0
        self.collision_start_time = None
        self.last_collision_duration = None

        self.last_action = [0.5, 1.57, 6, 20, 0.75, 1, 0.3]

        self.target_resolution = 0.01
        self.laser_size_m = 0.05
        self.robot_width = 0.430
        self.robot_length = 0.508
        self.path_width = 0.01
        self.crop_size_m = 8.0
        self.frame_id = 0
        self.costmap = None
        self.global_path = None

        self.front_half_width = 30
        self.side_min_deg = 30
        self.side_max_deg = 135

        self.global_path_history = []

        self.front_thresholds = [0.25, 0.50, 0.75]
        self.side_thresholds = [0.2, 0.40, 0.6]

        self.v = 2.0
        self.omega_eps = 0.5
        self.last_inflation = 0.31

        self.max_vel_x = {
            "safe": 2.0,
            "medium_safe": 1.5,
            "unsafe": 0.75,
            "very_unsafe": 0.25,
            "unknown": 0.6
        }

        self.min_vel_x = {
            "safe": 2.0,
            "medium_safe": 1.5,
            "unsafe": 0.75,
            "very_unsafe": 0.25,
            "unknown": 0.6
        }

        self.angular_policy = {
            "safe": 2.0,
            "medium": 1.5,
            "unsafe": 0.6,
            "very_unsafe": 0.0,
            "unknown": 1.5
        }

        self._setup_subscribers()
        self._setup_publisher()

        rospy.loginfo("JackalRos initialized")

    def _setup_subscribers(self):
        self._laser_sub = rospy.Subscriber(
            "/front/scan", LaserScan, self._laser_callback, queue_size=1)

        self._odom_sub = rospy.Subscriber(
            "/odometry/filtered", Odometry, self._odometry_callback, queue_size=1)

        self._collision_sub = rospy.Subscriber(
            '/collision', Bool, self._collision_callback)

        self._get_local_goal_sub = rospy.Subscriber(
            "/local_goal", Marker, self.local_goal_callback)

        self._get_global_goal_sub = rospy.Subscriber(
            "/global_goal", Marker, self.global_goal_callback)

        self._get_local_goal_movebase_sub = rospy.Subscriber(
            "/move_base/TrajectoryPlannerROS/local_plan",
            Path, self._local_goal_movebase_callback, queue_size=1)

        if self.img_dir is not None:
            rospy.sleep(2.0)

            self._costmap_sub = rospy.Subscriber(
                '/move_base/local_costmap/costmap',
                OccupancyGrid, self._costmap_callback, queue_size=1)

            self._path_sub = rospy.Subscriber(
                '/move_base/NavfnROS/plan',
                Path, self._path_callback, queue_size=1)

    def _setup_publisher(self):
        self._dynamics_pub = rospy.Publisher('/dy_dt', Float64MultiArray, queue_size=1)
        self._params_pub = rospy.Publisher('/ddp_params', Float64MultiArray, queue_size=1)
        self._local_goal_pub = rospy.Publisher('/current_local_goal', PointStamped, queue_size=1)
        self._global_goal_pub = rospy.Publisher('/current_global_goal', PointStamped, queue_size=1)

    def _collision_callback(self, msg):
        current_time = rospy.get_time()
        if msg.data:
            if not self.is_colliding:
                self.collision_count += 1
                self.collision_start_time = current_time
                self.is_colliding = True
        else:
            if self.is_colliding:
                duration = current_time - self.collision_start_time
                self.last_collision_duration = duration
                self.is_colliding = False

    def _laser_callback(self, msg):
        self.laser_data = msg

    def _odometry_callback(self, msg):

        q1 = msg.pose.pose.orientation.x
        q2 = msg.pose.pose.orientation.y
        q3 = msg.pose.pose.orientation.z
        q0 = msg.pose.pose.orientation.w

        self.robot_state['x'] = msg.pose.pose.position.x
        self.robot_state['y'] = msg.pose.pose.position.y
        self.robot_state['z'] = msg.pose.pose.position.z
        self.robot_state['theta'] = math.atan2(
            2 * (q0 * q3 + q1 * q2),
            1 - 2 * (q2 * q2 + q3 * q3))
        self.robot_state['velocity'] = msg.twist.twist.linear.x
        self.robot_state['angular_velocity'] = msg.twist.twist.angular.z

        if not self.start:
            if self.robot_state['velocity'] >= 0.1:
                self.start = True
                self.start_time = rospy.get_time()
        else:
            if self.robot_state['velocity'] <= 0.05:
                self.bad_vel += 1
            self.vel_counter += 1

    def _local_goal_movebase_callback(self, msg: Path):
        if not msg.poses:
            return

        last = msg.poses[-1].pose.position
        self.local_goal = np.array([last.x, last.y], dtype=float)
        if self.use_move_base:
            self.publish_goals()

    def _costmap_callback(self, msg):
        self.costmap = msg

    def _path_callback(self, msg):
        self.global_path = msg

    def local_goal_callback(self, msg):
        if len(msg.points) > 0:
            self.local_goal = np.array([msg.points[0].x, msg.points[0].y])

    def global_goal_callback(self, msg):
        if len(msg.points) > 0:
            self.global_goal = np.array([msg.points[0].x, msg.points[0].y])

    def set_params(self, p, indices=[0, 1]):

        msg = Float64MultiArray()
        if p is None:
            msg.data = []
        else:
            params_list = p.tolist() if hasattr(p, 'tolist') else list(p) if isinstance(p, (list, tuple)) else [
                float(p)]
            msg.data = [params_list[i] for i in indices if i < len(params_list)]

        self._params_pub.publish(msg)

    def set_dynamics_equation(self, action):
        msg = Float64MultiArray()
        if action is None or len(action) == 0:
            msg.data = []
        else:
            msg.data = action.tolist() if hasattr(action, 'tolist') else list(action)
        self._dynamics_pub.publish(msg)

    def get_global_path(self):
        if self.global_path is None or len(self.global_path.poses) == 0:
            return False
        else:
            return True

    def get_collision(self):
        return self.collision_count, self.last_collision_duration

    def get_laser_scan(self):
        return self.laser_data

    def get_robot_state(self):
        return np.array([
            self.robot_state['x'],
            self.robot_state['y'],
            self.robot_state['theta'],
            self.robot_state['velocity'],
            self.robot_state['angular_velocity']
        ])

    def get_local_goal(self):
        return self.local_goal

    def get_global_goal(self):
        return self.global_goal

    def get_cmd_vel(self):
        return self.cmd_vel_data

    def get_bad_vel(self):
        return [self.bad_vel, self.vel_counter]

    def publish_goals(self):
        local_msg = PointStamped()
        local_msg.header.stamp = rospy.Time.now()
        local_msg.header.frame_id = "odom"
        local_msg.point.x = self.local_goal[0]
        local_msg.point.y = self.local_goal[1]
        local_msg.point.z = 0.0
        self._local_goal_pub.publish(local_msg)

        global_msg = PointStamped()
        global_msg.header.stamp = rospy.Time.now()
        global_msg.header.frame_id = "odom"
        global_msg.point.x = self.global_goal[0]
        global_msg.point.y = self.global_goal[1]
        global_msg.point.z = 0.0
        self._global_goal_pub.publish(global_msg)

    def _draw_goals(self, img, robot_center, robot_x, robot_y, robot_yaw, crop_size_px):
        cos_yaw = math.cos(-robot_yaw)
        sin_yaw = math.sin(-robot_yaw)

        half_size = int(0.25 / 2 / self.target_resolution)

        if self.local_goal is not None:
            local_x, local_y = self.local_goal[0], self.local_goal[1]

            rel_x = local_x - robot_x
            rel_y = local_y - robot_y

            rel_x_base = rel_x * cos_yaw - rel_y * sin_yaw
            rel_y_base = rel_x * sin_yaw + rel_y * cos_yaw

            img_x = int(robot_center + rel_x_base / self.target_resolution)
            img_y = int(robot_center - rel_y_base / self.target_resolution)

            if 0 <= img_x < crop_size_px and 0 <= img_y < crop_size_px:
                cv2.rectangle(img,
                              (img_x - half_size, img_y - half_size),
                              (img_x + half_size, img_y + half_size),
                              (0, 255, 0), -1)  # -1表示填充

        if self.global_goal is not None:
            global_x, global_y = self.global_goal[0], self.global_goal[1]

            rel_x = global_x - robot_x
            rel_y = global_y - robot_y

            rel_x_base = rel_x * cos_yaw - rel_y * sin_yaw
            rel_y_base = rel_x * sin_yaw + rel_y * cos_yaw

            img_x = int(robot_center + rel_x_base / self.target_resolution)
            img_y = int(robot_center - rel_y_base / self.target_resolution)


            if 0 <= img_x < crop_size_px and 0 <= img_y < crop_size_px:

                cv2.rectangle(img,
                              (img_x - half_size, img_y - half_size),
                              (img_x + half_size, img_y + half_size),
                              (255, 0, 0), -1)  # -1表示填充

    def _draw_robot_rectangle(self, img, center_x, center_y, yaw):

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

    def _draw_global_path(self, img, robot_center, robot_x, robot_y, robot_yaw):

        if self.global_path is None or len(self.global_path.poses) == 0:
            return

        path_width_px = max(1, int(self.path_width / self.target_resolution))

        cos_yaw = math.cos(-robot_yaw)
        sin_yaw = math.sin(-robot_yaw)

        points = []
        for pose_stamped in self.global_path.poses:
            px_odom = pose_stamped.pose.position.x
            py_odom = pose_stamped.pose.position.y

            rel_x_odom = px_odom - robot_x
            rel_y_odom = py_odom - robot_y

            rel_x_base = rel_x_odom * cos_yaw - rel_y_odom * sin_yaw
            rel_y_base = rel_x_odom * sin_yaw + rel_y_odom * cos_yaw

            img_x = int(robot_center + rel_x_base / self.target_resolution)
            img_y = int(robot_center - rel_y_base / self.target_resolution)

            points.append((img_x, img_y))

        for i in range(len(points) - 1):
            cv2.line(img, points[i], points[i + 1], (255, 0, 255),
                     path_width_px, lineType=cv2.LINE_AA)

    def _sample_rotated_region(self, data, width, height, resolution,
                               robot_x, robot_y, yaw, crop_size_m):

        crop_size_px = int(crop_size_m / self.target_resolution)
        output = np.full((crop_size_px, crop_size_px), -1, dtype=np.int8)

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        out_x_indices = np.arange(crop_size_px)
        out_y_indices = np.arange(crop_size_px)
        out_x_grid, out_y_grid = np.meshgrid(out_x_indices, out_y_indices)

        local_x = (out_x_grid - crop_size_px / 2) * self.target_resolution
        local_y = (crop_size_px / 2 - out_y_grid) * self.target_resolution

        odom_x = robot_x + local_x * cos_yaw - local_y * sin_yaw
        odom_y = robot_y + local_x * sin_yaw + local_y * cos_yaw

        map_px = ((odom_x - self.costmap.info.origin.position.x) / resolution).astype(np.int32)
        map_py = ((odom_y - self.costmap.info.origin.position.y) / resolution).astype(np.int32)

        valid_mask = (map_px >= 0) & (map_px < width) & (map_py >= 0) & (map_py < height)
        output[valid_mask] = data[map_py[valid_mask], map_px[valid_mask]]

        return output

    def save_frame(self):

        if self.costmap is None or self.laser_data is None:
            return

        start_time = time.time()

        width = self.costmap.info.width
        height = self.costmap.info.height
        resolution = self.costmap.info.resolution

        robot_x = self.robot_state['x']
        robot_y = self.robot_state['y']
        yaw = self.robot_state['theta']

        # data = np.array(self.costmap.data).reshape((height, width))

        # data_aligned = self._sample_rotated_region(
        #     data, width, height, resolution, robot_x, robot_y, yaw, self.crop_size_m)

        # crop_size_px = data_aligned.shape[0]

        crop_size_px = int(self.crop_size_m / self.target_resolution)

        # img = np.zeros((crop_size_px, crop_size_px, 3), dtype=np.uint8)
        # img[data_aligned == -1] = [128, 128, 128]
        # img[data_aligned == 0] = [205, 205, 205]
        # img[(data_aligned > 0) & (data_aligned < 99)] = [255, 180, 120]
        # img[data_aligned >= 99] = [40, 40, 40]

        img = np.ones((crop_size_px, crop_size_px, 3), dtype=np.uint8) * 205

        robot_center = crop_size_px // 2

        grid_spacing = int(1.0 / self.target_resolution)
        max_offset = crop_size_px // grid_spacing + 1

        for i in range(-max_offset, max_offset + 1):
            x = robot_center + i * grid_spacing
            if 0 <= x < crop_size_px:
                cv2.line(img, (x, 0), (x, crop_size_px - 1), (120, 120, 120), 1)

        for i in range(-max_offset, max_offset + 1):
            y = robot_center + i * grid_spacing
            if 0 <= y < crop_size_px:
                cv2.line(img, (0, y), (crop_size_px - 1, y), (120, 120, 120), 1)

        laser_radius = max(1, int(self.laser_size_m / self.target_resolution))
        angle = self.laser_data.angle_min

        for r in self.laser_data.ranges:
            if r < self.laser_data.range_min or r > self.laser_data.range_max or np.isinf(r):
                angle += self.laser_data.angle_increment
                continue

            x = r * math.cos(angle)
            y = r * math.sin(angle)

            px = int(robot_center + x / self.target_resolution)
            py = int(robot_center - y / self.target_resolution)

            if 0 <= px < crop_size_px and 0 <= py < crop_size_px:
                cv2.circle(img, (px, py), laser_radius, (0, 0, 255), -1)

            angle += self.laser_data.angle_increment

        self._draw_global_path(img, robot_center, robot_x, robot_y, yaw)

        self._draw_goals(img, robot_center, robot_x, robot_y, yaw, crop_size_px)

        axis_length = int(1.0 / self.target_resolution)
        axis_width = max(3, int(0.05 / self.target_resolution))

        cv2.line(img, (robot_center, robot_center),
                 (robot_center + axis_length, robot_center),
                 (0, 0, 255), axis_width, lineType=cv2.LINE_AA)

        cv2.line(img, (robot_center, robot_center),
                 (robot_center, robot_center - axis_length),
                 (0, 255, 0), axis_width, lineType=cv2.LINE_AA)

        self._draw_robot_rectangle(img, robot_center, robot_center, -math.pi / 2)
        cv2.circle(img, (robot_center, robot_center), 5, (255, 255, 255), -1)

        crop_left_m = 2.0
        crop_top_m = 2.0
        crop_bottom_m = 2.0
        crop_right_m = 0.0

        px = lambda m: int(m / self.target_resolution)
        L, T, B, R = px(crop_left_m), px(crop_top_m), px(crop_bottom_m), px(crop_right_m)

        final_size_px = int(self.crop_size_m / self.target_resolution)
        half = final_size_px // 2

        x0 = robot_center - half + L
        x1 = robot_center + half - R
        y0 = robot_center - half + T
        y1 = robot_center + half - B

        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(img.shape[1], x1)
        y1 = min(img.shape[0], y1)

        img = img[y0:y1, x0:x1]

        cv2.imwrite(f'{self.img_dir}/frame_{self.frame_id:06d}.png', img)

        self.frame_id += 1

    def save_info(self, action):

        # param_list=['TrajectoryPlannerROS/max_vel_x',
        #             'TrajectoryPlannerROS/max_vel_theta',
        #             'TrajectoryPlannerROS/vx_samples',
        #             'TrajectoryPlannerROS/vtheta_samples',
        #             'TrajectoryPlannerROS/path_distance_bias',
        #             'TrajectoryPlannerROS/goal_distance_bias',
        #             'inflation_radius'],

        csv_path = os.path.join(self.img_dir, "data.csv")

        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)

        self.v = self.assess_safety()

        delta_inflation = self.assess_inflation()

        data = {
            "img_label": [self.frame_id],
            "linear_vel": self.robot_state["velocity"],
            "angular_vel": self.robot_state["angular_velocity"],


            "max_vel_x": [self.v],
            "max_vel_theta":[action[1]],
            "vx_samples":[action[2]],
            "path_distance_bias": [action[3]],
            "goal_distance_bias": [action[4]],
            "delta_inflation": [0.0]
        }

        df_new = pd.DataFrame(data)

        if self.frame_id == 0 or not os.path.exists(csv_path):
            df_new.to_csv(csv_path, mode='w', header=True, index=False)
        else:
            df_new.to_csv(csv_path, mode='a', header=False, index=False)

        self.last_inflation = action[6]

    def assess_inflation(self):

        if self.global_path is None or len(self.global_path.poses) == 0:
            return -0.01

        current_path = [[p.pose.position.x, p.pose.position.y]
                        for p in self.global_path.poses]

        if len(self.global_path_history) == 0:
            self.global_path_history.append(current_path)
            return 0.0


    def to_deg(self, rad):
        return rad * 180.0 / math.pi

    def classify(self, d, th):
        """th = [very_unsafe, unsafe, medium, safe]"""
        if math.isinf(d) or math.isnan(d):
            return "unknown"
        if d < th[0]:
            return "very_unsafe"
        if d < th[1]:
            return "unsafe"
        if d < th[2]:
            return "medium_safe"
        return "safe"

    def _rect_edge_radius(self, angles_rad):
        """
        向量化计算：从 (0,0) 出发，沿每个角度 ang 的方向，到达车体矩形外缘的距离 t(ang)。
        矩形：x ∈ [-a,a], y ∈ [-b,b]，a=L/2, b=W/2
        """
        a = self.robot_length * 0.5
        b = self.robot_width  * 0.5

        ang = np.asarray(angles_rad, dtype=float)
        ca = np.cos(ang)
        sa = np.sin(ang)

        INF = np.inf
        with np.errstate(divide='ignore', invalid='ignore'):
            # 与 x = +a 相交（仅当沿 +x 有分量时）
            t_xpos = np.where(ca > 1e-9, a / ca, INF)
            y_at_xpos = t_xpos * sa
            t_xpos = np.where(np.abs(y_at_xpos) <= b, t_xpos, INF)

            # 与 x = -a 相交（仅当沿 -x 有分量时；t 要为正）
            t_xneg = np.where(ca < -1e-9, -a / ca, INF)
            y_at_xneg = t_xneg * sa
            t_xneg = np.where(np.abs(y_at_xneg) <= b, t_xneg, INF)

            # 与 y = +b 相交（仅当沿 +y 有分量时）
            t_ypos = np.where(sa > 1e-9, b / sa, INF)
            x_at_ypos = t_ypos * ca
            t_ypos = np.where(np.abs(x_at_ypos) <= a, t_ypos, INF)

            # 与 y = -b 相交（仅当沿 -y 有分量时）
            t_yneg = np.where(sa < -1e-9, -b / sa, INF)
            x_at_yneg = t_yneg * ca
            t_yneg = np.where(np.abs(x_at_yneg) <= a, t_yneg, INF)

        # 取四种可能交点中 t>0 的最小值
        t = np.minimum.reduce([t_xpos, t_xneg, t_ypos, t_yneg])
        return t

    def _sector_min_deg(self, scan, deg_min, deg_max):
        """
        返回 [deg_min, deg_max] 扇区内的最小“清距”（障碍距离 - 车体外缘距离），单位：米。
        若跨越 ±180° 自动拆段。
        """
        n = len(scan.ranges)
        if n == 0:
            return float('inf')

        angles = scan.angle_min + np.arange(n) * scan.angle_increment
        ang_deg = np.degrees(angles)
        ranges = np.asarray(scan.ranges, dtype=float)

        def _min_clear_in_closed_interval(a_min, a_max):
            mask = (ang_deg >= a_min) & (ang_deg <= a_max)
            if not np.any(mask):
                return float('inf')
            vals = ranges[mask]
            angs = angles[mask]
            # 过滤无效量测
            valid = np.isfinite(vals)
            if not np.any(valid):
                return float('inf')
            vals = vals[valid]
            angs = angs[valid]
            # 计算到车体矩形外缘的半径，并求清距
            t_robot = self._rect_edge_radius(angs)
            clearance = vals - t_robot
            # 清距 < 0 说明已侵入（算作 0，更保守）
            clearance = np.maximum(clearance, 0.0)
            return float(np.min(clearance)) if clearance.size > 0 else float('inf')

        # 不跨界
        if deg_min <= deg_max:
            return _min_clear_in_closed_interval(deg_min, deg_max)
        # 跨界（如 170 到 -170）
        d1 = _min_clear_in_closed_interval(deg_min, 180.0)
        d2 = _min_clear_in_closed_interval(-180.0, deg_max)
        return min(d1, d2)

    # --- 计算四个扇区的最小距离 ---
    def _compute_sector_minima(self, scan):
        half = float(self.front_half_width)  # 默认 45°
        # 前方 ±half
        d_front = self._sector_min_deg(scan, -half, +half)
        # 左侧 45~135
        d_left  = self._sector_min_deg(scan, float(self.side_min_deg), float(self.side_max_deg))
        # 右侧 -45~-135
        d_right = self._sector_min_deg(scan, -float(self.side_max_deg), -float(self.side_min_deg))

        return {"front": d_front, "left": d_left, "right": d_right}


    def _label_sector(self, dist, sector):

        if sector == "front":
            th = self.front_thresholds
        elif sector in ("left", "right"):
            th = self.side_thresholds

        return self.classify(dist, th)

    def assess_safety(self):

        scan = self.laser_data
        if scan is None:
            return 1.5

        d = self._compute_sector_minima(scan)

        omega = self.robot_state['angular_velocity']

        safety_dist = self._compute_safety_distance(d, omega)

        v_max = self._dist_to_velocity(safety_dist)

        return v_max

    def _compute_safety_distance(self, d, omega):
        """
        Compute weighted minimum distance based on turning direction

        Args:
            d: dictionary with keys "front", "left", "right"
            omega: angular velocity (positive = left turn)

        Returns:
            weighted minimum distance in meters
        """
        eps = self.omega_eps

        # Dynamic weighting based on turning direction
        if omega > eps:  # Turning left
            weights = {"front": 1.0, "left": 1.5, "right": 0.75}
        elif omega < -eps:  # Turning right
            weights = {"front": 1.0, "left": 0.75, "right": 1.5}
        else:  # Going straight
            weights = {"front": 1, "left": 0.5, "right": 0.5}

        # Compute weighted distances
        weighted_distances = [d[s] / weights[s] for s in ["front", "left", "right"]]

        return min(weighted_distances)

    def _dist_to_velocity(self, dist):
        """
        Sigmoid mapping: distance → velocity

        Args:
            dist: safety distance in meters

        Returns:
            v_max: maximum velocity in m/s [0.2, 2.0]
        """
        # If distance >= d_max, always return v_max

        self.v_min = 0.5
        self.v_max = 2.0
        self.sigmoid_k = 4.0
        self.sigmoid_d_mid = 0.75
        self.sigmoid_d_max = 1.75

        if dist >= self.sigmoid_d_max:
            return self.v_max

        # Sigmoid mapping
        v = self.v_min + (self.v_max - self.v_min) / \
            (1.0 + np.exp(-self.sigmoid_k * (dist - self.sigmoid_d_mid)))

        return float(np.clip(v, self.v_min, self.v_max))

    def reset(self, init_params):
        self.is_colliding = False
        self.collision_count = 0
        self.collision_start_time = None
        self.last_collision_duration = None
        self.bad_vel = 0
        self.vel_counter = 0
        self.start = False
        self.start_time = 0
        self.last_action = init_params