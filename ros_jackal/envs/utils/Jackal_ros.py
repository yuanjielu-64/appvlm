from dataclasses import dataclass
import math, os, time
from typing import Any, Tuple
from collections import deque
import csv
import numpy as np
import cv2
import pandas as pd
import tf
import os
import copy
from enum import Enum

from scipy.signal import savgol_filter

try:
    from PIL import Image as PILImage
except ImportError:
    PILImage = None

try:
    import rospy
    from std_msgs.msg import Bool, Float64, Float64MultiArray
    from sensor_msgs.msg import LaserScan
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import Odometry, OccupancyGrid, Path
    from geometry_msgs.msg import PointStamped, Twist

    import sensor_msgs.point_cloud2 as pc2
    from sensor_msgs.msg import PointCloud2
    from std_msgs.msg import Header

    from visualization_msgs.msg import Marker
    from std_msgs.msg import Header
except ModuleNotFoundError:
    rospy = None

PLANNER_PARAMS = {
    "DWA": ["max_vel_x", "max_vel_theta", "vx_samples", "vtheta_samples",
            "path_distance_bias", "goal_distance_bias", "inflation"],

    "TEB": ["max_vel_x", "max_vel_theta", "min_obstacle_dist",
            "weight_kinematics", "weight_obstacle", "inflation"],

    "MPPI": ["num_samples", "horizon_length", "temperature",
             "max_vel_x", "inflation"],

    "DDP": ["iterations", "horizon", "max_vel_x",
            "regularization", "inflation"]
}


class RobotMode(Enum):
    BASELINE_RL_HB = "rl_hb"
    BASELINE_CHATGPT = "chatgpt"
    NORMAL = "normal"
    RECOVERY = "recovery_behavior"


@dataclass
class Cfg:
    target_res: float = 0.01
    robot_w: float = 0.430
    robot_l: float = 0.508
    path_w: float = 0.01
    crop_size_m: float = 8.0
    laser_size_m: float = 0.05
    last_inflation: float = 0.31

    # sectors & thresholds
    front_half_deg: int = 30
    side_min_deg: int = 30
    side_max_deg: int = 135
    front_th: tuple = (0.25, 0.50, 0.75)  # very_unsafe, unsafe, medium
    side_th: tuple = (0.20, 0.40, 0.60)

    # velocity mapping
    v_min: float = 0.5
    v_max: float = 2.0
    sig_k: float = 4.0
    sig_d_mid: float = 0.75
    sig_d_max: float = 1.75
    omega_eps: float = 0.5

    crop_left_m: float = 2.0
    crop_top_m: float = 2.0
    crop_bottom_m: float = 2.0
    crop_right_m: float = 0.0

    omega_max: float = 2.0
    curvature_sample_dist: float = 0.3
    curvature_gain: float = 0.8
    min_safe_dist: float = 0.3
    reference_speed: float = 1.0
    turning_curvature_thresh: float = 0.1
    side_clearance_ratio: float = 1.3

    infl_max_hi: float = 0.40
    infl_max_lo: float = 0.10
    vel_window: int = 10
    backtrack_alpha: float = 0.6
    vel_near_zero: float = 0.03


@dataclass
class RobotState:
    x: float = 0.0;
    y: float = 0.0;
    z: float = 0.0
    theta: float = 0.0;
    v: float = 0.0;
    w: float = 0.0
    mode: RobotMode = RobotMode.NORMAL

    def get_robot_state(self):
        return np.array([self.x, self.y, self.theta, self.v, self.w])


# ================== 3) 安全评估组件 ==================
class SafetyAssessor:

    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self._scan_cache = None
        self.d = {"front": 2, "left": 2, "right": 2}

    def _ensure_scan_cache(self, scan: LaserScan):
        if self._scan_cache is not None:
            return
        n = len(scan.ranges)
        angles = scan.angle_min + np.arange(n) * scan.angle_increment
        ang_deg = np.degrees(angles)
        c = self.cfg
        m_front = (ang_deg >= -c.front_half_deg) & (ang_deg <= c.front_half_deg)
        m_left = (ang_deg >= c.side_min_deg) & (ang_deg <= c.side_max_deg)
        m_right = (ang_deg <= -c.side_min_deg) & (ang_deg >= -c.side_max_deg)
        self._scan_cache = dict(angles=angles, m_front=m_front, m_left=m_left, m_right=m_right)

    def _rect_edge_radius(self, ang_rad: np.ndarray) -> np.ndarray:

        a = self.cfg.robot_l * 0.5
        b = self.cfg.robot_w * 0.5
        ca, sa = np.cos(ang_rad), np.sin(ang_rad)
        INF = np.inf
        with np.errstate(divide='ignore', invalid='ignore'):
            txp = np.where(ca > 1e-9, a / ca, INF);
            txp = np.where(np.abs(txp * sa) <= b, txp, INF)
            txn = np.where(ca < -1e-9, -a / ca, INF);
            txn = np.where(np.abs(txn * sa) <= b, txn, INF)
            typ = np.where(sa > 1e-9, b / sa, INF);
            typ = np.where(np.abs(typ * ca) <= a, typ, INF)
            tyn = np.where(sa < -1e-9, -b / sa, INF);
            tyn = np.where(np.abs(tyn * ca) <= a, tyn, INF)
        return np.minimum.reduce([txp, txn, typ, tyn])

    def _sector_min_masked(self, ranges, angles, mask) -> float:
        if not np.any(mask): return float('inf')
        vals = np.asarray(ranges, dtype=float)[mask]
        angs = angles[mask]
        valid = np.isfinite(vals)
        if not np.any(valid): return float('inf')
        vals = vals[valid];
        angs = angs[valid]
        t_robot = self._rect_edge_radius(angs)
        clr = np.maximum(vals - t_robot, 0.0)
        return float(clr.min()) if clr.size else float('inf')

    def _compute_sector_minima(self, scan: LaserScan):
        self._ensure_scan_cache(scan)
        c = self._scan_cache
        r = np.asarray(scan.ranges, dtype=float)
        return {
            "front": self._sector_min_masked(r, c["angles"], c["m_front"]),
            "left": self._sector_min_masked(r, c["angles"], c["m_left"]),
            "right": self._sector_min_masked(r, c["angles"], c["m_right"]),
        }

    def _weighted_clearance(self, d, omega):
        eps = self.cfg.omega_eps
        if omega > eps:
            weights = {"front": 1.0, "left": 1.5, "right": 0.75}
        elif omega < -eps:
            weights = {"front": 1.0, "left": 0.75, "right": 1.5}
        else:
            weights = {"front": 1.0, "left": 0.5, "right": 0.5}
        return min(d[s] / weights[s] for s in ("front", "left", "right"))

    def _dist_to_velocity(self, dist: float) -> float:
        c = self.cfg
        if dist >= c.sig_d_max: return c.v_max
        v = c.v_min + (c.v_max - c.v_min) / (1.0 + np.exp(-c.sig_k * (dist - c.sig_d_mid)))
        return float(np.clip(v, c.v_min, c.v_max))

    def assess(self, scan: LaserScan, state: RobotState) -> float:
        self.d = self._compute_sector_minima(scan)
        safety_d = self._weighted_clearance(self.d, state.w)
        return self._dist_to_velocity(safety_d)

    def assessAngular(self, state):
        v_max = 1.5
        a_lat_max = 1.0
        omega0 = 2.0
        v_floor = 0.2
        eps = 1e-6

        v = float(abs(state.v))

        cap_linear = omega0 * max(0.0, 1.0 - v / max(v_max, eps))

        cap_phys = a_lat_max / max(v, v_floor)

        cap = min(cap_linear, cap_phys)

        if not np.isfinite(cap) or cap < 0.0:
            cap = omega0

        cap = float(np.clip(cap, 0.0, getattr(self.cfg, "omega_max", omega0)))

        alpha = 0.3
        if getattr(self, "_cap_ema", None) is None:
            self._cap_ema = cap
        else:
            self._cap_ema = alpha * cap + (1 - alpha) * self._cap_ema
        cap = self._cap_ema

        return (-cap, cap)

    def reset(self):
        self._scan_cache = None


# ================== 4) 画图/存图组件 ==================
class FrameDrawer:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.frame_id = 0
        self.img_name = None  # 初始化为None，首次generate_img()时会设置
        self.img = None
        self.img_PIL = None  # PIL Image格式，用于pickle保存

    def _draw_robot(self, img, cx, cy, yaw):
        c = self.cfg
        w_px = int(c.robot_w / c.target_res)
        l_px = int(c.robot_l / c.target_res)
        half_w, half_l = w_px / 2, l_px / 2
        corners = np.array([[-half_l, -half_w], [half_l, -half_w], [half_l, half_w], [-half_l, half_w]],
                           dtype=np.float32)
        yaw_adj = yaw + math.pi / 2
        R = np.array([[math.cos(yaw_adj), -math.sin(yaw_adj)], [math.sin(yaw_adj), math.cos(yaw_adj)]])
        P = (corners @ R.T)
        P[:, 1] = -P[:, 1]
        P[:, 0] += cx
        P[:, 1] += cy
        pts = P.astype(np.int32)
        cv2.fillPoly(img, [pts], (0, 255, 255))
        cv2.polylines(img, [pts], True, (0, 0, 0), 2)

    def _draw_path(self, img, robot_center, robot_x, robot_y, robot_yaw, path: Path):
        if path is None or len(path.poses) == 0: return
        c = self.cfg
        path_w = max(1, int(c.path_w / c.target_res))
        cos_y, sin_y = math.cos(-robot_yaw), math.sin(-robot_yaw)
        pts = []
        for ps in path.poses:
            rx = ps.pose.position.x - robot_x
            ry = ps.pose.position.y - robot_y
            bx = rx * cos_y - ry * sin_y
            by = rx * sin_y + ry * cos_y
            ix = int(robot_center + bx / c.target_res)
            iy = int(robot_center - by / c.target_res)
            pts.append((ix, iy))
        for i in range(len(pts) - 1):
            cv2.line(img, pts[i], pts[i + 1], (0, 0, 0), path_w, lineType=cv2.LINE_AA)

    def _draw_goals(self, img, robot_center, robot_x, robot_y, robot_yaw, local_goal, global_goal):
        c = self.cfg
        cos_y, sin_y = math.cos(-robot_yaw), math.sin(-robot_yaw)
        half = int(0.25 / 2 / c.target_res)

        def draw_square(pt, color):
            x, y = pt
            cv2.rectangle(img, (x - half, y - half), (x + half, y + half), color, -1)

        for goal, color in ((local_goal, (0, 255, 0)), (global_goal, (255, 0, 0))):
            if goal is None: continue
            gx, gy = goal
            rx, ry = gx - robot_x, gy - robot_y
            bx = rx * cos_y - ry * sin_y
            by = rx * sin_y + ry * cos_y
            ix = int(robot_center + bx / c.target_res)
            iy = int(robot_center - by / c.target_res)
            h = img.shape[0]
            if 0 <= ix < h and 0 <= iy < h: draw_square((ix, iy), color)

    def _draw_grid(self, img, center):
        c = self.cfg
        step = int(1.0 / c.target_res)
        n = img.shape[0]
        max_off = n // step + 1
        for i in range(-max_off, max_off + 1):
            x = center + i * step
            y = center + i * step
            if 0 <= x < n: cv2.line(img, (x, 0), (x, n - 1), (120, 120, 120), 1)
            if 0 <= y < n: cv2.line(img, (0, y), (n - 1, y), (120, 120, 120), 1)

    def _draw_save_points(self, img, center, robot_x, robot_y, robot_yaw, save_points):
        """
        绘制save_points (从odom坐标系转换到图像坐标系)

        参数:
            img: 图像
            center: 图像中心坐标
            robot_x, robot_y, robot_yaw: 机器人在odom坐标系下的位姿
            save_points: (N, 3) 数组, [x_odom, y_odom, z]
        """
        c = self.cfg

        cos_y = math.cos(-robot_yaw)
        sin_y = math.sin(-robot_yaw)

        for point in save_points:
            # 1. 从odom坐标系转到机器人坐标系
            x_odom = point[0]
            y_odom = point[1]
            z = point[2]  # z值可以用来区分不同类型的点

            # 相对于机器人的位置
            rx = x_odom - robot_x
            ry = y_odom - robot_y

            # 旋转到机器人坐标系
            bx = rx * cos_y - ry * sin_y
            by = rx * sin_y + ry * cos_y

            # 2. 转换到图像坐标系
            ix = int(center + bx / c.target_res)
            iy = int(center - by / c.target_res)  # 注意y轴反向

            # 3. 检查是否在图像范围内
            if 0 <= ix < img.shape[1] and 0 <= iy < img.shape[0]:
                # 根据z值决定颜色和大小
                if abs(z - 0.3) < 0.01:
                    # 路径点中心 (z=0.3) - 黄色,较大
                    color = (0, 255, 255)  # BGR: 黄色
                    radius = 3
                else:
                    # 检测点 (z=0.0) - 绿色,较小
                    color = (0, 255, 0)  # BGR: 绿色
                    radius = 2

                cv2.circle(img, (ix, iy), radius, color, -1)

    def save_frame(self, img_dir, state: RobotState, scan: LaserScan, global_path: Path,
                   local_goal, global_goal, save_points, alg):

        if scan is None: return False

        c = self.cfg
        n = int(c.crop_size_m / c.target_res)
        img = np.ones((n, n, 3), dtype=np.uint8) * 205
        center = n // 2

        # grid + laser
        self._draw_grid(img, center)
        r_px = max(1, int(c.laser_size_m / c.target_res))
        ang = scan.angle_min
        for r in scan.ranges:
            if r < scan.range_min or r > scan.range_max or np.isinf(r):
                ang += scan.angle_increment;
                continue
            x = r * math.cos(ang);
            y = r * math.sin(ang)
            px = int(center + x / c.target_res);
            py = int(center - y / c.target_res)
            if 0 <= px < n and 0 <= py < n: cv2.circle(img, (px, py), r_px, (0, 0, 255), -1)
            ang += scan.angle_increment

        # path + goals + robot axes
        self._draw_path(img, center, state.x, state.y, state.theta, global_path)
        # self._draw_goals(img, center, state.x, state.y, state.theta, local_goal, global_goal)

        # if save_points is not None and len(save_points) > 0:
        #     self._draw_save_points(img, center, state.x, state.y, state.theta, save_points)

        axis_len = int(1.0 / c.target_res);
        axis_w = max(3, int(0.05 / c.target_res))
        cv2.line(img, (center, center), (center + axis_len, center), (0, 255, 0), axis_w)  # x-axis
        cv2.line(img, (center, center), (center, center - axis_len), (255, 0, 0), axis_w)  # y-axis
        self._draw_robot(img, center, center, -math.pi / 2)

        px = lambda m: int(m / c.target_res)
        L = px(c.crop_left_m)
        T = px(c.crop_top_m)
        B = px(c.crop_bottom_m)
        R = px(c.crop_right_m)

        H, W = img.shape[:2]

        x0 = max(0, L)
        y0 = max(0, T)
        x1 = min(W, W - R)
        y1 = min(H, H - B)

        if x1 > x0 and y1 > y0:
            img = img[y0:y1, x0:x1]

        self.img_dir = img_dir
        self.alg = alg
        self.img = copy.deepcopy(img)

        # 同时生成PIL Image格式（用于pickle保存）
        if PILImage is not None:
            # OpenCV使用BGR格式，PIL使用RGB格式，需要转换
            img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            self.img_PIL = PILImage.fromarray(img_rgb)
        else:
            self.img_PIL = None

        return True

    def generate_img(self):
        if self.img is not None:
            cv2.imwrite(os.path.join(self.img_dir, f"{self.alg}_{self.frame_id:06d}.png"), self.img)
            self.frame_id += 1
            self.img_name = f"{self.alg}_{self.frame_id:06d}.png"
            return True

        return False

    def get_current_img_as_pil(self):
        """
        获取当前图像的PIL Image对象（用于保存到pickle）

        Returns:
            PIL.Image 或 None
        """
        if self.img is None:
            return None

        if PILImage is not None:
            # OpenCV使用BGR格式，PIL使用RGB格式，需要转换
            img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            pil_img = PILImage.fromarray(img_rgb)
            return pil_img
        else:
            print("Warning: PIL not available, cannot convert to PIL Image")
            return None


class GlobalFunc:

    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.test_pub = None
        self.test_pub2 = None
        self.save_points = None
        self.inflation_history = []
        self._vel_hist = deque(maxlen=self.cfg.vel_window)

        self._last_cap = cfg.infl_max_hi
        self._ema_final = None
        self.dists = []

    def reset(self):
        self.save_points = None
        self.inflation_history = []
        self._vel_hist.clear()
        self._last_cap = self.cfg.infl_max_hi
        self._ema_final = None
        self.dists = []

    def assessDWA(self, scan_odom, global_path, state, inflation_radius):

        self._vel_hist.append(float(state.v))

        hist = list(self._vel_hist)

        if len(hist) < 3:
            return inflation_radius

        lo = float(getattr(self.cfg, "infl_max_lo", 0.20))
        hi = self._dyn_inflation_cap()

        ir = float(np.clip(inflation_radius, lo, hi))
        path = self.get_points_from_path(global_path, state)
        self.dists = self.get_corners_points(path, scan_odom, state)

        if self.dists is None or np.size(self.dists) == 0:
            target_raw = ir
        else:
            d_min = float(np.min(self.dists))
            if state.v <= 0.0:
                target_raw = max(lo, ir - np.random.uniform(0.02, 0.15))
            else:
                if 0.05 < d_min <= 0.10:
                    target_raw = ir
                elif 0.01 < d_min <= 0.05:
                    target_raw = ir
                else:
                    target_raw = ir

        target_raw = float(np.clip(target_raw, lo, hi))
        current = min(target_raw, hi)

        if self._ema_final is None:
            self._ema_final = current

        alpha = 0.3
        smoothed = alpha * current + (1 - alpha) * self._ema_final
        self._ema_final = smoothed

        return float(np.clip(smoothed, lo, hi))

    def assessDDP(self, scan_odom, global_path, state, inflation_radius):

        self._vel_hist.append(float(state.v))

        hist = list(self._vel_hist)

        if len(hist) < 3:
            return inflation_radius

        lo = float(getattr(self.cfg, "infl_max_lo", 0.20))
        hi = self._dyn_inflation_cap()

        ir = float(np.clip(inflation_radius, lo, hi))
        path = self.get_points_from_path(global_path, state)
        self.dists = self.get_corners_points(path, scan_odom, state)

        if self.dists is None or np.size(self.dists) == 0:
            target_raw = ir
        else:
            d_min = float(np.min(self.dists))
            if state.v <= 0.1:
                target_raw = max(lo, ir - np.random.uniform(0.02, 0.15))
            else:
                if d_min <= 0:
                    target_raw = ir + np.random.uniform(0.05, 0.15)
                elif 0 < d_min < 0.05:
                    target_raw = ir + np.random.uniform(0.01, 0.04)
                elif d_min < 0.10:
                    target_raw = ir + np.random.uniform(0.01, 0.02)
                else:
                    target_raw = ir

        target_raw = float(np.clip(target_raw, lo, hi))
        current = min(target_raw, hi)

        if self._ema_final is None:
            self._ema_final = current

        alpha = 0.3
        smoothed = alpha * current + (1 - alpha) * self._ema_final
        self._ema_final = smoothed

        return float(np.clip(smoothed, lo, hi))

    def assessTEB(self, scan_odom, global_path, state, inflation_radius):

        self._vel_hist.append(float(state.v))

        hist = list(self._vel_hist)

        if len(hist) < 3:
            return inflation_radius

        lo = float(getattr(self.cfg, "infl_max_lo", 0.20))
        hi = self._dyn_inflation_cap()

        ir = float(np.clip(inflation_radius, lo, hi))
        path = self.get_points_from_path(global_path, state)
        self.dists = self.get_corners_points(path, scan_odom, state)

        if self.dists is None or np.size(self.dists) == 0:
            target_raw = ir
        else:
            d_min = float(np.min(self.dists))
            if state.v <= 0.1:
                target_raw = max(lo, ir - np.random.uniform(0.02, 0.15))
            else:
                if d_min <= 0:
                    target_raw = ir + np.random.uniform(0.05, 0.15)
                elif 0 < d_min < 0.05:
                    target_raw = ir + np.random.uniform(0.01, 0.04)
                elif d_min < 0.10:
                    target_raw = ir + np.random.uniform(0.01, 0.02)
                else:
                    target_raw = ir

        target_raw = float(np.clip(target_raw, lo, hi))
        current = min(target_raw, hi)

        if self._ema_final is None:
            self._ema_final = current

        alpha = 0.3
        smoothed = alpha * current + (1 - alpha) * self._ema_final
        self._ema_final = smoothed

        return float(np.clip(smoothed, lo, hi))

    def assessMPPI(self, scan_odom, global_path, state, inflation_radius):

        self._vel_hist.append(float(state.v))

        hist = list(self._vel_hist)

        if len(hist) < 3:
            return inflation_radius

        lo = float(getattr(self.cfg, "infl_max_lo", 0.20))
        hi = self._dyn_inflation_cap()

        ir = float(np.clip(inflation_radius, lo, hi))
        path = self.get_points_from_path(global_path, state)
        self.dists = self.get_corners_points(path, scan_odom, state)

        if self.dists is None or np.size(self.dists) == 0:
            target_raw = ir
        else:
            d_min = float(np.min(self.dists))
            if state.v <= 0.1:
                target_raw = max(lo, ir - np.random.uniform(0.02, 0.15))
            else:
                if d_min <= 0:
                    target_raw = ir + np.random.uniform(0.05, 0.15)
                elif 0 < d_min < 0.05:
                    target_raw = ir + np.random.uniform(0.01, 0.04)
                elif d_min < 0.10:
                    target_raw = ir + np.random.uniform(0.01, 0.02)
                else:
                    target_raw = ir

        target_raw = float(np.clip(target_raw, lo, hi))
        current = min(target_raw, hi)

        if self._ema_final is None:
            self._ema_final = current

        alpha = 0.3
        smoothed = alpha * current + (1 - alpha) * self._ema_final
        self._ema_final = smoothed

        return float(np.clip(smoothed, lo, hi))

    def _update_vel_hist(self, v: float):

        self._vel_hist.append(float(v))

    def _dyn_inflation_cap(self) -> float:
        hist = list(self._vel_hist)
        hi = getattr(self.cfg, "infl_max_hi", 0.40)
        lo = getattr(self.cfg, "infl_max_lo", 0.10)

        neg_count = sum(1 for v in hist if v < 0)
        low_speed = sum(1 for v in hist if abs(v) < 0.1)
        N = len(hist)

        consecutive_neg = 0
        max_consecutive_neg = 0
        for v in hist:
            if v < 0.0:
                consecutive_neg += 1
                max_consecutive_neg = max(max_consecutive_neg, consecutive_neg)
            else:
                consecutive_neg = 0

        consecutive_low = 0
        max_consecutive_low = 0
        for v in hist:
            if abs(v) < 0.1:
                consecutive_low += 1
                max_consecutive_low = max(max_consecutive_low, consecutive_low)
            else:
                consecutive_low = 0

        target = hi

        if low_speed / N > 0.5:
            target -= 0.10

        if max_consecutive_low >= 3:
            target -= 0.05
        if max_consecutive_low >= 5:
            target -= 0.05
        if max_consecutive_low >= 7:
            target -= 0.05

        if neg_count >= 1:
            target -= 0.10

        if max_consecutive_neg >= 2:
            target -= 0.05
        if max_consecutive_neg >= 3:
            target -= 0.05

        target = float(np.clip(target, lo, hi))

        max_change_up = 0.03  # 恢复时：每次最多增加 0.03
        max_change_down = 0.08  # 降低时：每次最多减少 0.08

        if target > self._last_cap:
            # 恢复：慢慢增加
            cap = min(self._last_cap + max_change_up, target)
        else:
            # 降低：快速降低
            cap = max(self._last_cap - max_change_down, target)

        # 更新历史
        self._last_cap = cap

        return cap

    def get_points_from_path(self, global_path, state):

        if global_path is None or len(global_path.poses) < 2:
            return np.empty((0, 3), dtype=float)

        P = np.array([[ps.pose.position.x, ps.pose.position.y] for ps in global_path.poses], dtype=float)
        N = len(P)

        idxs = [0]
        i = 0
        while i < N - 1:
            step = np.random.randint(1, 3 + 1)
            i = min(i + step, N - 1)
            idxs.append(i)

        S = P[idxs]

        M = len(S)
        theta = np.empty(M, dtype=float)
        for k in range(M):
            if k < M - 1:
                dx, dy = S[k + 1, 0] - S[k, 0], S[k + 1, 1] - S[k, 1]
            else:
                dx, dy = S[k, 0] - S[k - 1, 0], S[k, 1] - S[k - 1, 1]
            if dx == 0.0 and dy == 0.0:

                theta[k] = theta[k - 1] if k > 0 else 0.0
            else:
                theta[k] = math.atan2(dy, dx)

        path_points = []
        check_distance = 3

        for pose in np.column_stack([S, theta])[:]:
            px, py, theta = pose[0], pose[1], pose[2]
            dist_from_robot = math.sqrt((px - state.x) ** 2 + (py - state.y) ** 2)

            if dist_from_robot < check_distance:
                path_points.append([px, py, theta])

        return np.array(path_points)

    def get_corners_points(self, path, scan_odom, state):
        """
        计算路径点的6个关键点到障碍物的最小距离
        8个点: 4个角点 + 4个侧面中点
        """
        if len(path) == 0:
            return 0.0

        x = path[:, 0]  # (N,)
        y = path[:, 1]  # (N,)
        theta = path[:, 2]  # (N,)

        half_l = self.cfg.robot_l / 2
        half_w = self.cfg.robot_w / 2

        corners_local = np.array([

            [half_l, half_w],  # 0: 前左角
            [half_l, -half_w],  # 1: 前右角

            [half_l / 3, -half_w],  # 2: 右侧前1/3
            [-half_l / 3, -half_w],  # 3: 右侧后1/3

            [-half_l, -half_w],  # 4: 后右角
            [-half_l, half_w],  # 5: 后左角

            [-half_l / 3, half_w],  # 6: 左侧后1/3
            [half_l / 3, half_w]  # 7: 左侧前1/3
        ])  # shape: (8, 2)

        cos_t = np.cos(theta)  # (N,)
        sin_t = np.sin(theta)  # (N,)

        R = np.zeros((len(path), 2, 2))
        R[:, 0, 0] = cos_t
        R[:, 0, 1] = -sin_t
        R[:, 1, 0] = sin_t
        R[:, 1, 1] = cos_t

        corners_odom = np.einsum('ij,nkj->nik', corners_local, R)
        # shape: (N, 8, 2) - N个路径点,每个6个关键点

        corners_odom[:, :, 0] += x[:, np.newaxis]
        corners_odom[:, :, 1] += y[:, np.newaxis]
        # 现在 corners_odom[i, j] = 第i个路径点的第j个角点在odom坐标系下的坐标

        diff = corners_odom[:, :, np.newaxis, :] - scan_odom[np.newaxis, np.newaxis, :, :]
        distances = np.linalg.norm(diff, axis=3)  # (N, 8, M)
        # distances[i, j, k] = 第i个路径点的第j个角点到第k个障碍物的距离

        closest_obs_idx_per_corner = distances.argmin(axis=2)  # (N, 8)
        # closest_ob_idx_per_corner[i, j] = 第i个路径点的第j个角点最近的障碍物索引

        min_distance_per_corner = distances.min(axis=2)  # shape: (N, 8)
        # min_distance_per_corner[i, j] = 第i个路径点的第j个角点到最近障碍物的距离

        left_idx = np.array([0, 5, 6, 7])
        right_idx = np.array([1, 2, 3, 4])

        d_left = np.min(min_distance_per_corner[:, left_idx], axis=1)
        d_right = np.min(min_distance_per_corner[:, right_idx], axis=1)

        avg_distance_per_path_point = 0.5 * (d_left + d_right)

        most_dangerous_path_idx = np.argmin(avg_distance_per_path_point)

        if self.test_pub is not None:
            self._visualize_most_dangerous_path_point(
                corners_odom[most_dangerous_path_idx],  # (8, 2) 最危险路径点的8个角点
                closest_obs_idx_per_corner[most_dangerous_path_idx],  # (8,) 对应的最近障碍物索引
                min_distance_per_corner[most_dangerous_path_idx],  # (8,) 各角点到障碍物的距离
                scan_odom,
                path[most_dangerous_path_idx],  # (3,) 路径点本身 [x, y, theta]
                path,
                avg_distance_per_path_point[most_dangerous_path_idx]  # 该路径点的最小距离
            )

        return avg_distance_per_path_point

    def _visualize_most_dangerous_path_point(self, corners, closest_obs_indices,
                                             corner_distances, scan_odom,
                                             path_point, path, min_distance):
        """
        可视化距离障碍物最近的路径点(最危险的点)

        参数:
            corners: (6, 2) 最危险路径点的6个检测点坐标
            closest_obs_indices: (6,) 每个检测点对应的最近障碍物索引
            corner_distances: (6,) 每个检测点到其最近障碍物的距离
            scan_odom: (M, 2) 所有障碍物点
            path_point: (3,) 路径点本身 [x, y, theta]
            min_distance: float 该路径点的最小距离(6个点中的最小值)
        """

        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "odom"

        # ========== 1. 发布6个检测点 (绿色,z=0) ========== ✅

        path_centers = np.column_stack([
            path[:, 0],  # x坐标
            path[:, 1],  # y坐标
            np.ones(len(path)) * 0.3  # z=0.3 (稍高)
        ]).astype(np.float32)  # shape: (N, 3)

        corners_3d = np.column_stack([
            corners,
            np.zeros(len(corners))
        ]).astype(np.float32)

        all_points = np.vstack([path_centers, corners_3d])  # shape: (N*9, 3)

        self.save_points = all_points

        cloud_corners = pc2.create_cloud_xyz32(header, all_points)
        self.test_pub.publish(cloud_corners)

        # ========== 2. 发布对应的最近障碍物 (红色,z=0.5) ========== ✅
        if hasattr(self, 'test_pub2') and self.test_pub2 is not None:
            closest_obstacles = scan_odom[closest_obs_indices]  # (8, 2)

            obstacles_3d = np.column_stack([
                closest_obstacles,
                np.ones(len(closest_obstacles)) * 0.5
            ]).astype(np.float32)

            cloud_obstacles = pc2.create_cloud_xyz32(header, obstacles_3d)
            self.test_pub2.publish(cloud_obstacles)

    def _visualize_corners_and_obstacles(self, corners_odom, closest_obstacle_idx, scan_odom):
        """
        可视化辅助函数: 显示角点和最近的障碍物
        """
        # 准备数据
        N = len(corners_odom)
        corners_flat = corners_odom.reshape(-1, 2)  # (N*4, 2) 展平所有角点
        obs_idx_flat = closest_obstacle_idx.reshape(-1)  # (N*4,) 展平障碍物索引
        closest_obstacles = scan_odom[obs_idx_flat]  # (N*4, 2) 对应的最近障碍物坐标

        # 发布1: 所有角点 (绿色点云)
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "odom"

        corners_3d = np.column_stack([
            corners_flat,
            np.zeros(len(corners_flat))  # z=0
        ]).astype(np.float32)

        cloud_corners = pc2.create_cloud_xyz32(header, corners_3d)
        self.test_pub.publish(cloud_corners)

        # 发布2: 每个角点对应的最近障碍物 (红色点云)
        if hasattr(self, 'test_pub2') and self.test_pub2 is not None:
            obstacles_3d = np.column_stack([
                closest_obstacles,
                np.ones(len(closest_obstacles)) * 0.5  # z=0.5 用于区分
            ]).astype(np.float32)

            cloud_obstacles = pc2.create_cloud_xyz32(header, obstacles_3d)
            self.test_pub2.publish(cloud_obstacles)


class JackalRos:
    def __init__(self, init_position, goal_position, use_move_base=False, img_dir=None, world_path=None, id=0,
                 cfg: Cfg = Cfg(), use_vlm=False, data_mode='auto', save_image=True, algorithm_name='Unknown'):
        self.cfg = cfg
        self.state = RobotState()
        self.last_state = RobotState()
        self.reference_state = RobotState()
        self.use_move_base = use_move_base
        self.img_dir = img_dir
        self.planner_name = self._get_planner_name()
        self._csv_path = None
        self.local_goal = np.array([0.0, 4.0], dtype=float)
        self.start_position = init_position[:2]
        self.global_goal = np.array(goal_position[:2], dtype=float) if goal_position is not None else None
        self.scan: LaserScan | None = None
        self.costmap: OccupancyGrid | None = None
        self.global_path: Path | None = None
        self.WORLD_PATH = world_path
        self.id = id

        self.use_vlm = use_vlm
        self.data_mode = data_mode
        self.save_image = save_image
        self.algorithm_name = algorithm_name

        # Set CSV file names based on data_mode
        if data_mode == 'manual':
            self._csv_path = os.path.join(img_dir, "data_fix.csv") if img_dir else None
            self._trajectory_path = os.path.join(img_dir, "data_trajectory_fix.csv") if img_dir else None
        else:  # auto mode
            self._csv_path = os.path.join(img_dir, "data.csv") if img_dir else None
            self._trajectory_path = os.path.join(img_dir, "data_trajectory.csv") if img_dir else None
        self._csv_header_written = os.path.exists(self._csv_path) if self._csv_path else False

        self.start = False
        self.bad_vel = 0;
        self.vel_counter = 0
        self.is_colliding = False;
        self.collision_count = 0
        self.collision_start_time = None;
        self.last_collision_duration = None
        self.last_collision_time = None
        self.should_abort = False

        self.path_curvature = 0.0
        self.local_plan: Path | None = None

        self.max_collision_duration = 1

        self.row = None

        self.start_frame_id = 0
        self.end_frame_id = 0

        self.iteration = 0

        self.prev_goal_dist = None
        self.min_goal_dist = None
        self.no_progress_count = 0
        self.last_save_dist = None

        self.obs_odom = None

        time.sleep(1)

        self.safety = SafetyAssessor(cfg)
        self.globalFunc = GlobalFunc(cfg)
        self.drawer = FrameDrawer(cfg)

        if rospy:
            self._setup_subscribers()
            self._setup_publishers()
            self.globalFunc.test_pub = self.test
            self.globalFunc.test_pub2 = self.test2

    # ---------- ROS: subs/pubs & callbacks ----------
    def _setup_subscribers(self):
        self._laser_sub = rospy.Subscriber("/front/scan", LaserScan, self._on_laser, queue_size=1)
        self._odom_sub = rospy.Subscriber("/odometry/filtered", Odometry, self._on_odom, queue_size=1)
        self._coll_sub = rospy.Subscriber("/collision", Bool, self._on_collision)
        self._lgoal_sub = rospy.Subscriber("/local_goal", Marker, self._on_local_goal)
        self._ggoal_sub = rospy.Subscriber("/global_goal", Marker, self._on_global_goal)
        self._lplan_sub = rospy.Subscriber("/move_base/TrajectoryPlannerROS/local_plan", Path, self._on_local_plan,
                                           queue_size=1)

        if self.img_dir:
            time.sleep(2.0)
            self._costmap_sub = rospy.Subscriber("/move_base/local_costmap/costmap", OccupancyGrid, self._on_costmap,
                                                 queue_size=1)
            self._path_sub = rospy.Subscriber("/move_base/NavfnROS/plan", Path, self._on_global_path, queue_size=1)

    def _setup_publishers(self):
        self.pub_dy = rospy.Publisher('/dy_dt', Float64MultiArray, queue_size=1)
        self.pub_param = rospy.Publisher('/params', Float64MultiArray, queue_size=1)
        self.pub_lg = rospy.Publisher('/current_local_goal', PointStamped, queue_size=1)
        self.pub_gg = rospy.Publisher('/current_global_goal', PointStamped, queue_size=1)
        self.pub_smooth = rospy.Publisher("/smooth_global_path", Path, queue_size=1, latch=True)
        self.pub_scan_odom = rospy.Publisher("/scan_odom", PointCloud2, queue_size=1, latch=True)
        self.test = rospy.Publisher("/test", PointCloud2, queue_size=1, latch=True)
        self.test2 = rospy.Publisher("/test2", PointCloud2, queue_size=1, latch=True)

    def _on_laser(self, msg: LaserScan):
        self.scan = msg
        pose_odom = (self.state.x, self.state.y, self.state.theta)
        self.obs_odom = self.scan_to_odom(msg, pose_odom, voxel=0.05)

        self.publish_scan_odom(stamp=msg.header.stamp)

    def _on_costmap(self, msg: OccupancyGrid):
        self.costmap = msg

    def _on_global_path(self, msg: Path):

        if len(msg.poses) < 5:
            self.global_path = msg
            self.pub_smooth.publish(msg)
            return

        path_xy = np.array([[p.pose.position.x, p.pose.position.y] for p in msg.poses], dtype=float)

        smooth_xy = self.smooth_path_savgol(
            path_xy,
            step=0.05,
            win_base=11,
            win_strong=17,
            order=2,
            corridor_r=0.6,
            kappa_lo=0.3,
            kappa_hi=0.8,
            keep_ends=True
        )

        out = Path()
        out.header = msg.header
        out.poses = []
        for x, y in smooth_xy:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            out.poses.append(ps)

        self.global_path = out

        self.pub_smooth.publish(out)

    def _on_local_plan(self, msg: Path):
        self.local_plan = msg

        if msg.poses:
            p = msg.poses[-1].pose.position
            self.local_goal = np.array([p.x, p.y], dtype=float)

            self.path_curvature = self._compute_path_curvature(msg)

            if self.use_move_base:
                self.publish_goals()

    def _on_local_goal(self, msg: Marker):
        if msg.points:
            p = msg.points[0]
            self.local_goal = np.array([p.x, p.y], dtype=float)

    def _on_global_goal(self, msg: Marker):
        if msg.points:
            p = msg.points[0]
            self.global_goal = np.array([p.x, p.y], dtype=float)

    def _on_collision(self, msg: Bool):
        current_time = rospy.get_time()

        if msg.data:
            self.last_collision_time = current_time

            if not self.is_colliding:
                self.is_colliding = True
                self.collision_start_time = current_time
                self.collision_count += 1

        if self.is_colliding and self.last_collision_time is not None:
            time_since_last = current_time - self.last_collision_time
            if time_since_last > 0.25:
                print(time_since_last)
                self.is_colliding = False
                self.last_collision_duration = current_time - self.collision_start_time
                self.collision_start_time = None

        if self.is_colliding and self.collision_start_time is not None:
            duration = current_time - self.collision_start_time
            if duration >= self.max_collision_duration:
                self.should_abort = True
                self.last_collision_duration = duration

    def _on_odom(self, msg: Odometry):
        q1, q2, q3, q0 = msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w
        theta = math.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
        self.state.x = msg.pose.pose.position.x
        self.state.y = msg.pose.pose.position.y
        self.state.z = msg.pose.pose.position.z
        self.state.theta = theta
        self.state.v = msg.twist.twist.linear.x
        self.state.w = msg.twist.twist.angular.z
        if not self.start and self.state.v >= 0.1:
            self.start = True;
            self.start_time = rospy.get_time() if rospy else time.time()
        elif self.start:
            if self.state.v <= 0.05: self.bad_vel += 1
            self.vel_counter += 1

    def get_bad_vel(self):
        return [self.bad_vel, self.vel_counter]

    def publish_scan_odom(self, stamp):
        pts = self.obs_odom
        if pts is None or len(pts) == 0:
            return
        header = Header()
        header.stamp = stamp
        header.frame_id = "odom"

        pts3 = np.column_stack([pts, np.zeros((pts.shape[0],), dtype=np.float32)]).astype(np.float32)

        cloud_msg = pc2.create_cloud_xyz32(header, pts3.astype(np.float32))
        self.pub_scan_odom.publish(cloud_msg)

    def assess_safety(self) -> float:
        if self.scan is None: return self.cfg.v_min
        return self.safety.assess(self.scan, self.state)

    def assess_vel_angular(self, vel_theta) -> [float, float]:
        if self.scan is None: return (vel_theta, -vel_theta)
        return self.safety.assessAngular(self.state)

    def assess_inflation(self, inflation, planner) -> float:
        if self.global_path is None: return inflation - 0.2
        if planner == 'DWA':
            return self.globalFunc.assessDWA(self.obs_odom, self.global_path, self.state, inflation)
        elif planner == 'DDP':
            return self.globalFunc.assessDDP(self.obs_odom, self.global_path, self.state, inflation)
        elif planner == 'TEB':
            return self.globalFunc.assessTEB(self.obs_odom, self.global_path, self.state, inflation)
        elif planner == 'MPPI':
            return self.globalFunc.assessMPPI(self.obs_odom, self.global_path, self.state, inflation)

    def _odd(self, n):
        n = int(max(n, 3))
        return n if n % 2 == 1 else n - 1

    def smooth_path_savgol(self, path_xy, step=0.10, win_base=11, win_strong=17, order=2,
                           corridor_r=0.6,
                           kappa_lo=0.3, kappa_hi=0.8,
                           keep_ends=True):

        P0 = np.asarray(path_xy, float)

        if len(P0) < 5:
            return P0

        d = np.linalg.norm(np.diff(P0, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(d)])
        u = np.arange(0.0, s[-1] + 1e-9, step)
        X = np.interp(u, s, P0[:, 0])
        Y = np.interp(u, s, P0[:, 1])
        P = np.c_[X, Y]
        n = len(P)

        wb = self._odd(min(win_base, n))
        ws = self._odd(min(win_strong, n))
        Xb = savgol_filter(P[:, 0], wb, order, mode='interp')
        Yb = savgol_filter(P[:, 1], wb, order, mode='interp')
        Pb = np.c_[Xb, Yb]

        Xs = savgol_filter(P[:, 0], ws, order, mode='interp')
        Ys = savgol_filter(P[:, 1], ws, order, mode='interp')
        Ps = np.c_[Xs, Ys]

        if n < 3:
            Pmix = Pb
        else:
            A = P[:-2]
            B = P[1:-1]
            C = P[2:]
            v1 = B - A
            v2 = C - B
            n1 = np.linalg.norm(v1, axis=1) + 1e-9
            n2 = np.linalg.norm(v2, axis=1) + 1e-9
            cosang = np.clip((v1 * v2).sum(1) / (n1 * n2), -1.0, 1.0)
            ang = np.arccos(cosang)
            seg = (n1 + n2) * 0.5
            kappa = ang / (seg + 1e-9)
            kappa = np.r_[0.0, kappa, 0.0]

            alpha = (kappa - kappa_lo) / (kappa_hi - kappa_lo + 1e-9)
            alpha = np.clip(alpha, 0.0, 1.0)[:, None]

            Pmix = (1.0 - alpha) * Pb + alpha * Ps

        if keep_ends:
            Pmix[0] = P[0]
            Pmix[-1] = P[-1]

        if corridor_r and corridor_r > 0:
            delta = Pmix - P
            norm = np.linalg.norm(delta, axis=1)
            mask = norm > corridor_r
            if np.any(mask):
                scale = corridor_r / (norm[mask] + 1e-9)
                Pmix[mask] = P[mask] + delta[mask] * scale[:, None]

        return Pmix

    def scan_to_odom(self, scan, pose_odom, voxel=0.05):
        ang = scan.angle_min + np.arange(len(scan.ranges)) * scan.angle_increment
        r = np.asarray(scan.ranges, dtype=float)
        m = np.isfinite(r) & (r >= scan.range_min) & (r <= scan.range_max)
        if not np.any(m):
            return np.empty((0, 2))

        xb = r[m] * np.cos(ang[m])
        yb = r[m] * np.sin(ang[m])
        pts_base = np.c_[xb, yb]

        x, y, th = pose_odom
        c, s = math.cos(th), math.sin(th)
        R = np.array([[c, -s], [s, c]], dtype=float)
        t = np.array([x, y], dtype=float)
        pts_odom = pts_base @ R.T + t

        if voxel and voxel > 0:
            keys = np.round(pts_odom / voxel).astype(np.int32)
            _, idx = np.unique(keys, axis=0, return_index=True)
            pts_odom = pts_odom[np.sort(idx)]

        return pts_odom

    def publish_goals(self):
        if not rospy: return

        def pub_point(pub, xy):
            msg = PointStamped();
            msg.header.stamp = rospy.Time.now();
            msg.header.frame_id = "odom"
            msg.point.x, msg.point.y, msg.point.z = float(xy[0]), float(xy[1]), 0.0
            pub.publish(msg)

        pub_point(self.pub_lg, self.local_goal)
        if self.global_goal is not None: pub_point(self.pub_gg, self.global_goal)

    def set_params(self, v):
        msg = Float64MultiArray()

        if v is None:
            msg.data = []
        else:

            if hasattr(v, 'tolist'):
                msg.data = v.tolist()
            elif isinstance(v, (list, tuple)):
                msg.data = list(v)
            else:
                msg.data = [float(v)]

        self.pub_param.publish(msg)

    def get_collision(self):
        return self.is_colliding

    def set_dynamics_equation(self, action):
        if not rospy: return
        msg = Float64MultiArray()
        msg.data = [] if not action else (action.tolist() if hasattr(action, 'tolist') else list(action))
        self.pub_dy.publish(msg)

    def save_frame(self):

        if not self.save_image: return False

        if self.img_dir is None or self.scan is None: return False

        if not os.path.exists(self.img_dir): os.makedirs(self.img_dir, exist_ok=True)

        # 根据 algorithm_name 确定标签
        if self.algorithm_name == 'APPLR':
            alg = "RL"
        elif self.algorithm_name == 'Heurstic_based':
            alg = "HB"
        elif self.algorithm_name in ['Qwen', 'ChatGPT', 'IL']:
            alg = "VLM"
        elif self.algorithm_name in ['FTRL']:
            alg = "FTRL"
        else:
            alg = "Unknown"

        # Add "_fix" suffix for manual mode
        if self.data_mode == 'manual':
            alg = f"{alg}_fix"

        return self.drawer.save_frame(self.img_dir, self.state, self.scan, self.global_path, self.local_goal,
                                      self.global_goal, self.globalFunc.save_points, alg)

    def save_info(self, action, start, done, info):
        if not self.img_dir:
            return

        if self.algorithm_name == 'APPLR':
            alg = "RL"
        elif self.algorithm_name == 'Heurstic_based':
            alg = "HB"
        elif self.algorithm_name in ['Qwen', 'ChatGPT', 'IL']:
            alg = "VLM"
        elif self.algorithm_name == 'FTRL':
            alg = "FTRL"
        else:
            alg = "Unknown"

        if alg in ['RL', 'HB']:
            if start:
                self.start_frame_id = self.drawer.frame_id

            if done:
                self._save_trajectory_summary(alg, info)
                return

            self.row = self._build_row_data(alg, action)

            if not self.should_save_frame():
                return

            if self.save_image:
                result = self.drawer.generate_img()
                self._write_to_csv(result)

        elif alg in ['FTRL']:
            if start:
                self.save_frame()
                self.start_frame_id = self.drawer.frame_id
                self.drawer.img_name = f"{alg}_{self.drawer.frame_id:06d}.png"
                if self.save_image:
                    self.drawer.generate_img()
            elif done:
                self.save_frame()
                if self.save_image:
                    self.drawer.generate_img()
                return
            else:
                self.save_frame()
                if self.save_image:
                    self.drawer.generate_img()

        else:  # VLM
            if start:
                self.save_frame()
                self.start_frame_id = self.drawer.frame_id
                if self.save_image:
                    self.drawer.generate_img()
            elif done:
                self._save_trajectory_summary(alg, info)
                if self.save_image:
                    self._clean_imgs()
                return
            else:
                if self.save_image:
                    self.save_frame()
                    self.drawer.generate_img()

    def should_save_frame(self):

        if self.is_colliding == True:
            return False

        delta_theta = abs(self.state.theta - self.reference_state.theta)
        if delta_theta > math.pi:
            delta_theta = 2 * math.pi - delta_theta

        gx, gy = float(self.global_goal[0]), float(self.global_goal[1])

        current_goal_dist = math.hypot(self.state.x - gx, self.state.y - gy)
        reference_goal_dist = math.hypot(self.reference_state.x - gx,
                                         self.reference_state.y - gy)

        cumulative_progress = reference_goal_dist - current_goal_dist

        if self.min_goal_dist is None:
            self.min_goal_dist = current_goal_dist
            closer_to_goal = False
        else:
            closer_to_goal = (self.min_goal_dist - current_goal_dist) > 0.1
            if current_goal_dist < self.min_goal_dist:
                self.min_goal_dist = current_goal_dist

        is_turning = delta_theta > 0.2
        has_velocity = abs(self.state.v) >= 0.1

        has_cumulative_progress = cumulative_progress > 0.15

        if closer_to_goal:
            self.reference_state = copy.deepcopy(self.state)
            self.no_progress_count = 0
            return True

        if has_cumulative_progress:
            self.reference_state = copy.deepcopy(self.state)
            self.no_progress_count = 0
            return True

        if is_turning or has_velocity:
            self.no_progress_count += 1
            if self.no_progress_count % 3 == 0:
                self.reference_state = copy.deepcopy(self.state)
                self.no_progress_count = 0
                return True
            return False

        self.no_progress_count += 1

        if self.no_progress_count > 4:
            if np.random.random() < 0.05:
                return True
            return False

        return True

    def reset(self, init_params):
        self.is_colliding = False;
        self.collision_count = 0
        self.collision_start_time = None;
        self.last_collision_duration = None
        self.bad_vel = 0;
        self.vel_counter = 0;
        self.start = False
        self.start_time = 0;
        self.last_action = init_params
        self.last_collision_time = None
        self.should_abort = False
        self.path_curvature = 0.0
        self.prev_goal_dist = None
        self.min_goal_dist = None
        self.no_progress_count = 0
        self.last_save_dist = None
        self.local_plan = None
        self.row = None
        self.obs_odom = None

        self.teb_fail = 0
        self.dwa_fail = 0

        self.globalFunc.reset()
        self.safety.reset()

    def _clean_imgs(self):
        for frame_id in range(self.start_frame_id, self.drawer.frame_id):

            img_name = f"VLM_{frame_id:06d}.png"
            img_path = os.path.join(self.img_dir, img_name)

            if os.path.exists(img_path):
                try:
                    os.remove(img_path)
                except Exception as e:
                    if rospy:
                        rospy.logwarn(f"Failed to delete {img_name}: {e}")

    def _compute_goal_metrics(self):
        gx, gy = float(self.global_goal[0]), float(self.global_goal[1])
        goal_dist = math.hypot(self.state.x - gx, self.state.y - gy)

        delta_goal = 0.0 if self.prev_goal_dist is None else self.prev_goal_dist - goal_dist
        moved_toward = 1 if delta_goal > 0.05 else (-1 if delta_goal < -0.05 else 0)

        if self.min_goal_dist is None or goal_dist < self.min_goal_dist:
            self.min_goal_dist = goal_dist

        self.prev_goal_dist = goal_dist
        return float(goal_dist), float(delta_goal), int(moved_toward)

    def _build_row_data(self, alg, action):

        goal_dist, delta_goal, moved_toward = self._compute_goal_metrics()

        base_data = {
            "Method": alg,
            "img_label": self.drawer.frame_id,
            "linear_vel": self.last_state.v,
            "angular_vel": self.last_state.w,
            "pre_inflation": self._safe_get(action, len(action) - 1, 0.0),
            "goal_dist": goal_dist,
            "delta_goal_dist": delta_goal,
            "moved_toward_goal": moved_toward,
            "next_linear_vel:": self.state.v,
            "next_angular_vel": self.state.w
        }

        planner_params = self._extract_planner_params(alg, action)

        return {**base_data, **planner_params}

    def _extract_planner_params(self, alg, action):
        if self.planner_name == "DWA":
            if alg == "RL":
                return {
                    "max_vel_x": self._safe_get(action, 0),
                    "max_vel_theta": self._safe_get(action, 1),
                    "vx_samples": int(self._safe_get(action, 2)),
                    "vtheta_samples": int(self._safe_get(action, 3)),
                    "path_distance_bias": self._safe_get(action, 4),
                    "goal_distance_bias": self._safe_get(action, 5),
                    "final_inflation": self._safe_get(action, 6),
                }
            else:
                v_max = self._safe_call(self.assess_safety, 1.5)
                _, max_w = self._safe_call(lambda: self.assess_vel_angular(action[1]), (-2, 2))
                inflation = self._safe_call(lambda: self.assess_inflation(action[6], 'DWA'), 0.3)

                vx = self._safe_get(action, 2, 12)
                vtheta = self._safe_get(action, 3, 40)

                ratio = vx / vtheta if vtheta != 0 else 1.0
                target = np.random.uniform(400, 600)
                vtheta = np.clip(np.sqrt(target / ratio), 10, 50)
                vx = np.clip(ratio * vtheta, 3, 50)

                if not math.isnan(vx):
                    vx = int(vx)
                else:
                    vx = 12

                if not math.isnan(vtheta):
                    vtheta = int(vtheta)
                else:
                    vtheta = 50

                return {
                    "max_vel_x": v_max,
                    "max_vel_theta": max_w if isinstance(max_w, float) else max_w[1],
                    "vx_samples": int(vx),
                    "vtheta_samples": int(vtheta),
                    "path_distance_bias": self._safe_get(action, 4, 0.05),
                    "goal_distance_bias": self._safe_get(action, 5, 0.2),
                    "final_inflation": inflation,
                }
        elif self.planner_name == "TEB":
            if alg == "RL":
                return {
                    "max_vel_x": self._safe_get(action, 0),
                    "max_vel_x_backwards": self._safe_get(action, 1),
                    "max_vel_theta": int(self._safe_get(action, 2)),
                    "dt_ref": int(self._safe_get(action, 3)),
                    "min_obstacle_dist": self._safe_get(action, 4),
                    "inflation_dist": self._safe_get(action, 5),
                    "final_inflation": self._safe_get(action, 6),
                }
            else:
                v_max = self._safe_call(self.assess_safety, 1.5)
                _, max_w = self._safe_call(lambda: self.assess_vel_angular(action[1]), (-2, 2))
                inflation = self._safe_call(lambda: self.assess_inflation(action[6], 'TEB'), 0.3)

                if self.state.v >= 1.5:
                    min_obstacle_dist = np.random.uniform(0.25, 0.35)
                    inflation_dist = np.random.uniform(0.30, 0.50)

                elif self.state.v >= 1.0:
                    min_obstacle_dist = np.random.uniform(0.18, 0.25)
                    inflation_dist = np.random.uniform(0.20, 0.35)

                else:
                    min_obstacle_dist = np.random.uniform(0.12, 0.18)
                    inflation_dist = np.random.uniform(0.15, 0.25)

                return {
                    "max_vel_x": v_max,
                    "max_vel_x_backwards": action[1],
                    "max_vel_theta": max_w if isinstance(max_w, float) else max_w[1],
                    "dt_ref": action[3],
                    "min_obstacle_dist": min_obstacle_dist,
                    "inflation_dist": inflation_dist,
                    "final_inflation": inflation,
                }
        elif self.planner_name == "DDP":
            if alg == "RL":

                PARAM_LIMITS = {
                    'max_vel_x': [0.0, 2.0],
                    'max_vel_theta': [0.314, 3.14],
                    'nr_pairs_': [400, 800],
                    'distance': [0.01, 0.4],
                    'robot_radius': [0.01, 0.15],
                    'inflation_radius': [0.1, 0.6],
                }

                return {
                    "max_vel_x": self._clip_param(action, 0, PARAM_LIMITS['max_vel_x']),
                    "max_vel_theta": self._clip_param(action, 1, PARAM_LIMITS['max_vel_theta']),
                    "nr_pairs_": int(self._clip_param(action, 2, PARAM_LIMITS['nr_pairs_'])),
                    "distance": self._clip_param(action, 3, PARAM_LIMITS['distance']),
                    "robot_radius": self._clip_param(action, 4, PARAM_LIMITS['robot_radius']),
                    "final_inflation": self._clip_param(action, 5, PARAM_LIMITS['inflation_radius']),
                }
            else:
                v_max = self._safe_call(self.assess_safety, 1.5)
                _, max_w = self._safe_call(lambda: self.assess_vel_angular(action[1]), (-2, 2))
                inflation = self._safe_call(lambda: self.assess_inflation(action[5], 'DDP'), 0.25)

                if self.state.v >= 1.5:
                    nr_pairs = max(action[2], 500)
                    distance = np.random.uniform(0.2, 0.4)
                    robot_radius = np.random.uniform(0.10, 0.20)
                elif self.state.v >= 1.0:
                    nr_pairs = max(action[2], 600)
                    distance = np.random.uniform(0.05, 0.15)
                    robot_radius = np.random.uniform(0.05, 0.10)
                else:
                    nr_pairs = max(action[2], 800)
                    distance = np.random.uniform(0.01, 0.05)
                    robot_radius = np.random.uniform(0.01, 0.05)

                return {
                    "max_vel_x": v_max,
                    "max_vel_theta": max_w if isinstance(max_w, float) else max_w[1],
                    "nr_pairs_": int(nr_pairs),
                    "distance": distance,
                    "robot_radius": robot_radius,
                    "final_inflation": inflation,
                }
        elif self.planner_name == "MPPI":
            if alg == "RL":
                return {
                    "max_vel_x": self._safe_get(action, 0),
                    "max_vel_theta": self._safe_get(action, 1),
                    "nr_pairs_": int(self._safe_get(action, 2)),
                    "nr_steps_": int(self._safe_get(action, 3)),
                    "linear_stddev": self._safe_get(action, 4),
                    "angular_stddev": self._safe_get(action, 5),
                    "lambda": self._safe_get(action, 6),
                    "final_inflation": self._safe_get(action, 7),
                }
            else:
                v_max = self._safe_call(self.assess_safety, 1.5)
                _, max_w = self._safe_call(lambda: self.assess_vel_angular(action[1]), (-2, 2))
                inflation = self._safe_call(lambda: self.assess_inflation(action[6], 'MPPI'), 0.3)

                if self.state.v >= 1.5:
                    nr_pairs = max(action[2], 500)
                    nr_steps_ = np.random.uniform(10, 20)

                elif self.state.v >= 1.0:
                    nr_pairs = max(action[2], 600)
                    nr_steps_ = np.random.uniform(20, 30)

                else:
                    nr_pairs = max(action[2], 800)
                    nr_steps_ = np.random.uniform(30, 40)

                return {
                    "max_vel_x": v_max,
                    "max_vel_theta": max_w if isinstance(max_w, float) else max_w[1],
                    "nr_pairs_": int(nr_pairs),
                    "nr_steps_": int(nr_steps_),
                    "linear_stddev": self._safe_get(action, 4),
                    "angular_stddev": self._safe_get(action, 5),
                    "lambda": self._safe_get(action, 6),
                    "final_inflation": inflation,
                }

    def _clip_param(self, action, idx, limits, default=None):

        if default is None:
            default = (limits[0] + limits[1]) / 2

        val = self._safe_get(action, idx, default)
        return float(np.clip(val, limits[0], limits[1]))

    def _write_to_csv(self, result):

        if not self.row:
            return

        if result == False:
            return

        write_header = not self._csv_header_written or not os.path.exists(self._csv_path)

        try:
            df = pd.DataFrame([self.row])
            df.to_csv(self._csv_path, mode='a', header=write_header, index=False)
            self._csv_header_written = True

        except Exception as e:
            if rospy:
                rospy.logwarn(f"CSV write failed: {e}")

    def _append_csv(self, path, data):

        os.makedirs(os.path.dirname(path), exist_ok=True)
        write_header = not os.path.exists(path)

        try:
            df = pd.DataFrame([data])
            df.to_csv(path, mode='a', header=write_header, index=False)
        except Exception as e:
            if rospy:
                rospy.logwarn(f"CSV append failed: {e}")

    def _safe_get(self, action, idx, default=0.0):
        try:
            val = float(action[idx])
            return default if np.isnan(val) else val
        except:
            return default

    def _safe_call(self, func, default):
        try:
            result = func()
            if isinstance(result, (list, tuple)):
                return result if not any(np.isnan(x) for x in result) else default
            return result if not np.isnan(result) else default
        except:
            return default

    def _save_trajectory_summary(self, alg, info):

        opt_time, nav_metric = self.get_score(
            self.start_position, self.global_goal,
            info['status'], info['time'], info['world']
        )

        if info['collision'] >= 1:
            nav_metric = 0.0

        summary = {
            "Method": alg,
            "Start_frame_id": self.start_frame_id,
            "Done_frame_id": self.drawer.frame_id - 1,
            "Collision": info['collision'],
            "Recovery": info['recovery'],
            "Smoothness": info['smoothness'],
            "Status": info['status'],
            "Time": info['time'],
            "World": info['world'],
            "optimal_time": opt_time,
            "nav_metric": nav_metric,
        }

        self._append_csv(self._trajectory_path, summary)

        if self.use_vlm == False:
            test_dir = f"test_{'rl' if alg == 'RL' else 'hb'}"
        else:
            test_dir = f"test_vlm"

        test_path = os.path.join(os.path.dirname(self.img_dir), test_dir, f"test_results_{self.id}.csv")
        self._append_csv(test_path,
                         {k: v for k, v in summary.items() if k != "Start_frame_id" and k != "Done_frame_id"})

        self.iteration += 2

    def _get_action_value(self, action, idx, default=None):

        if action is None:
            return default
        try:
            return float(action[idx])
        except (IndexError, TypeError, ValueError):
            return default

    def _get_planner_name(self):

        if not self.img_dir:
            return "unknown"

        parts = os.path.normpath(self.img_dir).split(os.sep)

        known_planners = {'dwa', 'teb', 'mppi', 'ddp'}

        for part in reversed(parts):

            if part.startswith('actor_'):
                continue

            prefix = part.split('_')[0].lower()

            if prefix in known_planners:
                return prefix.upper()

        return "UNKNOWN"

    def get_score(self, INIT_POSITION, GOAL_POSITION, status, time, world):

        if status == "success":
            success = True
        else:
            success = False

        world = int(world.split('_')[1].split('.')[0])

        path_file_name = os.path.join(self.WORLD_PATH, "path_files/", "path_%d.npy" % int(world))
        path_array = np.load(path_file_name)
        path_array = [self.path_coord_to_gazebo_coord(*p) for p in path_array]
        path_array = np.insert(path_array, 0, (INIT_POSITION[0], INIT_POSITION[1]), axis=0)
        path_array = np.insert(path_array, len(path_array),
                               (INIT_POSITION[0] + GOAL_POSITION[0], INIT_POSITION[1] + GOAL_POSITION[1]), axis=0)
        path_length = 0
        for p1, p2 in zip(path_array[:-1], path_array[1:]):
            path_length += self.compute_distance(p1, p2)

        optimal_time = path_length / 2
        actual_time = time
        nav_metric = int(success) * optimal_time / np.clip(actual_time, 2 * optimal_time, 8 * optimal_time)

        return optimal_time, nav_metric

    def compute_distance(self, p1, p2):
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    def path_coord_to_gazebo_coord(self, x, y):
        RADIUS = 0.075
        r_shift = -RADIUS - (30 * RADIUS * 2)
        c_shift = RADIUS + 5

        gazebo_x = x * (RADIUS * 2) + r_shift
        gazebo_y = y * (RADIUS * 2) + c_shift

        return (gazebo_x, gazebo_y)

    def _compute_path_curvature(self, path: Path, sample_dist=0.3) -> float:

        if path is None or len(path.poses) < 2:
            return 0.0

        def find_point_at_distance(start_idx, target_dist):
            cumulative = 0.0
            p_prev = path.poses[start_idx].pose.position

            for i in range(start_idx + 1, len(path.poses)):
                p_curr = path.poses[i].pose.position
                seg_dist = math.sqrt(
                    (p_curr.x - p_prev.x) ** 2 +
                    (p_curr.y - p_prev.y) ** 2
                )
                cumulative += seg_dist

                if cumulative >= target_dist:
                    return p_curr, i

                p_prev = p_curr

            return None, -1

        p0 = path.poses[0].pose.position
        p1, idx1 = find_point_at_distance(0, sample_dist)

        if p1 is None:
            return 0.0

        p2, _ = find_point_at_distance(idx1, sample_dist)

        if p2 is None:
            return 0.0

        dx1, dy1 = p1.x - p0.x, p1.y - p0.y
        dx2, dy2 = p2.x - p1.x, p2.y - p1.y

        d1 = math.sqrt(dx1 ** 2 + dy1 ** 2)
        d2 = math.sqrt(dx2 ** 2 + dy2 ** 2)

        if d1 < 0.01 or d2 < 0.01:
            return 0.0

        cross = dx1 * dy2 - dy1 * dx2
        return float(2.0 * cross / (d1 * d2 * (d1 + d2)))


