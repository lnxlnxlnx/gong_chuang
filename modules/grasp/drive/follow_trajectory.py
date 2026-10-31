"""扣除臂上相机运动后，从真实观测学习短时顺时针圆弧。"""

from collections import deque
import math

import numpy as np


class PlanarGeometry:
    """只用已有传动与半径标定，连接电机脉冲和水平平面。"""

    def __init__(self, config):
        self.ppr = config.pulses_per_revolution
        self.angular_scale = self.ppr * config.base_reduction / (2 * math.pi)
        self.radial_scale = self.ppr / config.require_mm_per_motor_rev("extension")
        axis = config.axes["extension"]
        if (
            axis.radius_reference_mm is None
            or axis.radius_reference_motor_degrees is None
        ):
            raise ValueError("连续跟随需要伸缩半径标定")
        self.reference_radius = axis.radius_reference_mm
        self.reference_pulses = axis.radius_reference_motor_degrees * self.ppr / 360

    def world(self, base, extension):
        radius = (
            self.reference_radius
            + (extension - self.reference_pulses) / self.radial_scale
        )
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("连续跟随的物料半径必须为正")
        angle = -base / self.angular_scale  # 电机 CW 为正，数学平面逆时针为正。
        return radius * math.cos(angle), radius * math.sin(angle)

    def joints(self, point, velocity, reference_base):
        x, y = point
        vx, vy = velocity
        radius = math.hypot(x, y)
        if radius <= 0:
            raise ValueError("物料轨迹不能穿过底座转轴")
        angle = -math.atan2(y, x)
        angle += (
            2
            * math.pi
            * round((reference_base / self.angular_scale - angle) / (2 * math.pi))
        )
        return (
            (
                angle * self.angular_scale,
                self.reference_pulses
                + (radius - self.reference_radius) * self.radial_scale,
            ),
            (
                -(x * vy - y * vx) / (radius * radius) * self.angular_scale,
                (x * vx + y * vy) / radius * self.radial_scale,
            ),
        )


def _estimate_velocity(points, *, reject_jitter=True):
    """用拟合残差排除检测抖动，保留有观测支持的真实速度。"""
    if len(points) < 4:
        return (0.0, 0.0), False
    values = np.array(points)
    times = values[:, 0] - values[:, 0].mean()
    centered = values[:, 1:] - values[:, 1:].mean(axis=0)
    time_energy = times @ times
    velocity = times @ centered / time_energy
    residual = centered - times[:, None] * velocity
    uncertainty = math.sqrt(
        float((residual * residual).sum()) / (len(points) - 2) / time_energy
    )
    threshold = max(1.0, 3 * uncertainty) if reject_jitter else 1.0
    stationary = math.hypot(*velocity) <= threshold
    return ((0.0, 0.0) if stationary else tuple(velocity)), stationary


class ClockwiseTrajectory:
    """窗口内的运动用于前馈；只有实测支持的顺时针圆弧能用于丢料预测。"""

    def __init__(self, *, prediction_seconds=1.5, stationary_seconds=0.5):
        if not math.isfinite(prediction_seconds) or prediction_seconds <= 0:
            raise ValueError("prediction_seconds 必须为正的有限数")
        self.prediction_seconds = prediction_seconds
        if not math.isfinite(stationary_seconds) or stationary_seconds <= 0:
            raise ValueError("stationary_seconds 必须为正的有限数")
        self.stationary_seconds = stationary_seconds
        self._stationary_points = deque()
        self.points = deque(maxlen=40)
        self.circle = None
        self.velocity = (0.0, 0.0)
        self.stationary = False

    def reset_stability(self):
        """丢料后重新积累停稳证据，保留运动历史供短时预测。"""
        self._stationary_points.clear()
        self.stationary = False

    def observe(self, stamp, point):
        if self.points and stamp <= self.points[-1][0]:
            raise ValueError("跟随观测的时间戳必须递增")
        if self.points and stamp - self.points[-1][0] > 0.35:
            self.points.clear()
            self.reset_stability()
        self.points.append((stamp, *point))
        self._stationary_points.append((stamp, *point))
        # 停稳窗口本身就是用户设置的时长，不能先等速度窗口结束再另等半秒。
        while (
            len(self._stationary_points) > 4
            and self._stationary_points[1][0] <= stamp - self.stationary_seconds
        ):
            self._stationary_points.popleft()
        while self.points and stamp - self.points[0][0] > 0.8:
            self.points.popleft()
        recent = [p for p in self.points if stamp - p[0] <= 0.4]
        if len(recent) < 4:
            recent = list(self.points)[-4:]
        self.velocity, _ = _estimate_velocity(recent)
        # 停盘前的减速也会产生拟合残差，不能把它当作检测抖动而提前放行。
        _, stationary = _estimate_velocity(
            list(self._stationary_points), reject_jitter=False,
        )
        self.stationary = (
            stationary
            and stamp - self._stationary_points[0][0] >= self.stationary_seconds
        )
        self.circle = None
        if len(self.points) < 4:
            return
        values = np.array(self.points)
        xy = values[:, 1:]
        origin = xy.mean(axis=0)
        centered = xy - origin
        if np.linalg.norm(xy[-1] - xy[0]) < 3:
            return
        matrix = np.column_stack((2 * centered, np.ones(len(xy))))
        solution, _, rank, singular = np.linalg.lstsq(
            matrix, (centered * centered).sum(axis=1), rcond=None
        )
        if rank < 3 or singular[-1] / singular[0] < 1e-4:
            return
        center = solution[:2] + origin
        radii = np.linalg.norm(xy - center, axis=1)
        radius = float(radii.mean())
        if radius < 5 or max(abs(radii - radius)) > 1.0:
            return
        angles = np.unwrap(np.arctan2(xy[:, 1] - center[1], xy[:, 0] - center[0]))
        times = values[:, 0] - stamp
        slope, _ = np.linalg.lstsq(
            np.column_stack((times, np.ones(len(times)))), angles, rcond=None
        )[0]
        # 不从噪声或反向观测捏造顺时针运动，也不沿用停盘前的旧角速度。
        if slope >= -0.01 or angles[0] - angles[-1] < 0.03:
            return
        speed = math.hypot(*self.velocity)
        if speed < 1 or abs(speed - radius * abs(slope)) > max(2, speed * 0.4):
            return
        self.circle = (center, radius, float(angles[-1]), float(slope))

    @property
    def last_stamp(self):
        return None if not self.points else self.points[-1][0]

    def predict(self, stamp, *, invisible=False):
        if not self.points or (
            invisible
            and (
                self.circle is None or stamp - self.last_stamp > self.prediction_seconds
            )
        ):
            return None
        dt = max(0.0, stamp - self.last_stamp)
        if self.circle is None:
            _, x, y = self.points[-1]
            return (
                (x + self.velocity[0] * dt, y + self.velocity[1] * dt),
                self.velocity,
            )
        center, radius, angle, omega = self.circle
        angle += omega * dt
        c, s = math.cos(angle), math.sin(angle)
        return (
            (center[0] + radius * c, center[1] + radius * s),
            (-radius * s * omega, radius * c * omega),
        )
