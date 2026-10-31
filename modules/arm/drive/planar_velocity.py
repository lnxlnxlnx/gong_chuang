"""Measured, leased speed control confined to the two planar arm motors."""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Mapping, TYPE_CHECKING

from modules.motor.drive.emm42 import Direction, Emm42Driver

if TYPE_CHECKING:
    from modules.arm.runtime import RobotArm

AXES = ("base", "extension")


@dataclass(frozen=True)
class PlanarSample:
    """Absolute motor pulses anchored to entry command odometry."""

    stamp: float
    base_pulses: float
    extension_pulses: float


class PlanarVelocitySession:
    """Serialize encoder exchanges and lease every speed inside motor limits.

    The driver retains its 30 ms silence after unacknowledged writes; this
    session adds no duplicate inter-frame wait.
    """

    def __init__(
        self,
        arm: RobotArm,
        *,
        max_rpm: int,
        motor_limits: Mapping[str, tuple[float, float]],
        acceleration: int = 200,
        command_timeout: float = 0.35,
    ):
        if (
            isinstance(max_rpm, bool)
            or not isinstance(max_rpm, int)
            or not 1 <= max_rpm <= Emm42Driver.MAX_RPM
        ):
            raise ValueError(
                "max_rpm must be a positive integer within the driver range"
            )
        if (
            isinstance(acceleration, bool)
            or not isinstance(acceleration, int)
            or not 0 <= acceleration <= 255
        ):
            raise ValueError("acceleration must be an integer in 0..255")
        if not math.isfinite(command_timeout) or command_timeout <= 0:
            raise ValueError("command_timeout must be finite and positive")
        if set(motor_limits) != set(AXES):
            raise ValueError("motor_limits must contain exactly base and extension")
        self._limits = {}
        for axis in AXES:
            low, high = motor_limits[axis]
            if not math.isfinite(low) or not math.isfinite(high) or low >= high:
                raise ValueError(
                    f"motor_limits[{axis}] must be finite increasing motor degrees"
                )
            position = arm.turns_state.of(axis).motor_degrees
            if not low <= position <= high:
                raise ValueError(
                    f"{axis} initial motor position {position} "
                    f"outside limits {(low, high)}"
                )
            self._limits[axis] = (float(low), float(high))
        self._arm = arm
        self._motors = {axis: arm._motors[axis] for axis in AXES}
        self._max_rpm = max_rpm
        self._acceleration = acceleration
        self._timeout = float(command_timeout)
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._closed = self._expired = False
        self._failure = self._deadline = None
        self._speeds = dict.fromkeys(AXES, 0)
        self._stopped = False
        self._history = deque(maxlen=512)
        self._axis_history = {axis: deque(maxlen=512) for axis in AXES}
        self._baseline = {}
        self._anchor = {axis: arm.turns_state.of(axis).pulses for axis in AXES}
        # 运动中读失败直接刹停，不再持锁重试三次；限速按这条实际故障路径留余量。
        # 两轴各一次读取、静默和刹停仍须全部计入，不能仅按正常串口耗时算。
        read_cost = sum(
            2 * float(getattr(m, "timeout", 0.1))
            + Emm42Driver.QUIET_TIMEOUT_SECONDS
            for m in self._motors.values()
        )
        stop_cost = sum(
            Emm42Driver.MAX_ATTEMPTS
            * (
                2 * float(getattr(m, "timeout", 0.1))
                + Emm42Driver.QUIET_TIMEOUT_SECONDS
                + Emm42Driver.RETRY_SETTLE_SECONDS
            )
            if getattr(m, "expect_ack", False)
            else Emm42Driver.QUIET_TIMEOUT_SECONDS + Emm42Driver.UNACKED_SETTLE_SECONDS
            for m in self._motors.values()
        )
        self._brake_seconds = read_cost + stop_cost + 0.1
        try:
            started = time.monotonic()
            for axis, motor in self._motors.items():
                self._baseline[axis] = motor.read_position()
                self._axis_history[axis].append((time.monotonic(), self._anchor[axis]))
            self._sample_started = started
            self._history.append(
                PlanarSample(
                    time.monotonic(), self._anchor["base"], self._anchor["extension"]
                )
            )
        except BaseException as error:
            self._fail(error)
            raise
        self._watchdog = threading.Thread(
            target=self._watch, name="arm-planar-lease", daemon=True
        )
        self._watchdog.start()

    def _require_live(self):
        if self._closed:
            raise RuntimeError("planar velocity session closed")
        if self._failure is not None:
            raise RuntimeError("planar velocity session failed") from self._failure
        if self._expired:
            raise RuntimeError("planar velocity command lease expired")

    def _read_sample(self):
        started = time.monotonic()
        measured = {}
        stamps = {}
        moving = any(self._speeds.values())
        for axis, motor in self._motors.items():
            read_started = time.monotonic()
            raw = motor.read_position(attempts=1) if moving else motor.read_position()
            # 两轴共用串口，不能把第一轴的测量错标为第二轴读完的时间。
            stamps[axis] = (read_started + time.monotonic()) / 2
            measured[axis] = (
                self._anchor[axis]
                + (raw - self._baseline[axis]) * self._arm.pulses_per_revolution / 65536
            )
        # Do not partially reconcile one axis when the other encoder fails.
        for axis in AXES:
            self._arm.turns_state.record(
                axis, round(measured[axis]) - self._arm.turns_state.of(axis).pulses
            )
            degrees = measured[axis] * 360 / self._arm.pulses_per_revolution
            low, high = self._limits[axis]
            if not low <= degrees <= high:
                raise RuntimeError(
                    f"{axis} measured motor position {degrees} "
                    f"outside limit {(low, high)}"
                )
            self._axis_history[axis].append((stamps[axis], measured[axis]))
        snapshot = PlanarSample(
            time.monotonic(), measured["base"], measured["extension"]
        )
        self._sample_started = started
        self._history.append(snapshot)
        return snapshot

    def sample(self) -> PlanarSample:
        """Read both encoders; failure brakes both axes and invalidates odometry."""
        with self._lock:
            self._require_live()
            try:
                return self._read_sample()
            except BaseException as error:
                self._fail(error)
                raise

    def position_at(self, stamp: float) -> tuple[float, float]:
        """Interpolate measured history only; never extrapolate camera latency."""
        with self._lock:
            self._require_live()
            positions = []
            for axis in AXES:
                history = self._axis_history[axis]
                if (
                    not math.isfinite(stamp)
                    or not history[0][0] <= stamp <= history[-1][0]
                ):
                    raise ValueError(
                        "timestamp outside measured planar position history"
                    )
                previous = history[0]
                for current in history:
                    if current[0] >= stamp:
                        ratio = (
                            0
                            if current[0] == previous[0]
                            else (stamp - previous[0]) / (current[0] - previous[0])
                        )
                        positions.append(
                            previous[1] + ratio * (current[1] - previous[1])
                        )
                        break
                    previous = current
            return tuple(positions)

    def set_velocity(self, base_rpm: float, extension_rpm: float) -> tuple[int, int]:
        """Refresh the lease and write only changed, safe integer motor speeds."""
        requested = (base_rpm, extension_rpm)
        if any(not math.isfinite(value) for value in requested):
            raise ValueError("planar rpm must be finite")
        with self._lock:
            self._require_live()
            now = time.monotonic()
            if self._deadline is not None and now >= self._deadline:
                self._expired = True
                self._stop_locked()
                raise RuntimeError("planar velocity command lease expired")
            # F6 反转仍有减速段：先刹停、实测对账，避免按新方向余量续租旧方向运动。
            if any(
                old * request < 0
                for old, request in zip(self._speeds.values(), requested)
            ):
                try:
                    self._stop_locked()
                except BaseException as error:
                    self._fail(error)
                    raise
                now = time.monotonic()
            snapshot = self._history[-1]
            age = max(0, now - self._sample_started)
            actual = []
            for axis, request, pulses in zip(
                AXES, requested, (snapshot.base_pulses, snapshot.extension_pulses)
            ):
                low, high = self._limits[axis]
                degrees = pulses * 360 / self._arm.pulses_per_revolution
                sign = 1 if request >= 0 else -1
                room = high - degrees if sign > 0 else degrees - low
                # Bound motion since the oldest read by the configured cap and
                # keep half a pulse for integer odometry quantization.
                room -= self._max_rpm * 6 * age + 180 / self._arm.pulses_per_revolution
                cap = max(
                    0, math.floor(room / (6 * (self._timeout + self._brake_seconds)))
                )
                # A lower F6 target still needs time to decelerate from the
                # previous speed. FE is safer than asking for a slow ramp when
                # the old speed has already consumed the braking allowance.
                old_braking_room = abs(self._speeds[axis]) * 6 * self._brake_seconds
                deceleration_cap = max(
                    0, math.floor((room - old_braking_room) / (6 * self._timeout))
                )
                cap = min(cap, deceleration_cap)
                actual.append(sign * int(min(abs(request), self._max_rpm, cap)))
            try:
                self._deadline = now + self._timeout
                self._stopped = False
                self._wake.set()
                for axis, rpm in zip(AXES, actual):
                    if rpm == self._speeds[axis]:
                        continue
                    if rpm == 0:
                        self._motors[axis].stop()
                    else:
                        self._motors[axis].set_speed(
                            Direction.CW if rpm > 0 else Direction.CCW,
                            abs(rpm),
                            self._acceleration,
                        )
                    self._speeds[axis] = rpm
                    if time.monotonic() >= self._deadline:
                        self._expired = True
                        raise RuntimeError(
                            "planar velocity command lease expired "
                            "during speed exchange"
                        )
            except BaseException as error:
                self._fail(error)
                raise
            return tuple(actual)

    def _brake_both(self):
        errors = []
        for axis, motor in self._motors.items():
            try:
                motor.stop()
                self._speeds[axis] = 0
            except BaseException as error:
                errors.append(error)
        if errors:
            self._arm._planar_position_valid = False
            raise RuntimeError(
                "planar stop failed: " + "; ".join(str(e) for e in errors)
            ) from errors[0]
        self._stopped = True

    def _fail(self, primary):
        self._failure = primary
        self._arm._planar_position_valid = False
        self._deadline = None
        self._wake.set()
        try:
            self._brake_both()
        except BaseException as cleanup:
            raise primary.with_traceback(primary.__traceback__) from cleanup

    def _stop_locked(self):
        self._deadline = None
        self._wake.set()
        if self._stopped:
            return
        try:
            self._brake_both()
        except BaseException as error:
            self._failure = error
            raise
        if self._failure is None:
            try:
                self._read_sample()
            except BaseException as error:
                self._failure = error
                self._arm._planar_position_valid = False
                raise

    def stop(self):
        """Attempt both brakes even if one fails, then reconcile measured motion."""
        with self._lock:
            if not self._closed:
                self._stop_locked()

    def _watch(self):
        while True:
            with self._lock:
                if self._closed or self._failure is not None or self._expired:
                    return
                delay = (
                    None
                    if self._deadline is None
                    else self._deadline - time.monotonic()
                )
                if delay is not None and delay <= 0:
                    self._expired = True
                    try:
                        self._stop_locked()
                    except BaseException as error:
                        self._failure = error
                    return
                self._wake.clear()
            self._wake.wait(delay)

    def close(self):
        """Brake and release exclusive arm access on every exit path."""
        try:
            with self._lock:
                if self._closed:
                    return
                try:
                    self._stop_locked()
                finally:
                    self._closed = True
                    self._wake.set()
                    if self._arm._planar_velocity_session is self:
                        self._arm._planar_velocity_session = None
        finally:
            if self._watchdog is not threading.current_thread():
                self._watchdog.join(timeout=1)

    def __enter__(self):
        self._require_live()
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.close()
        except BaseException as cleanup:
            if exc is not None:
                raise exc.with_traceback(traceback) from cleanup
            raise
        return False
