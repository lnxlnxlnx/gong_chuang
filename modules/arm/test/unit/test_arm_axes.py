"""一次多轴：同时启动（多机同步三步法）、并行安全上限、失败后的缓存清理、票据。

全部离线：串口是内存假件，等待只记账不真睡。
"""

from __future__ import annotations

import unittest

from modules.arm.config import ArmConfig, AxisConfig
from modules.arm.drive.kinematics import CylindricalPose
from modules.arm.runtime import (
    AXES_SAME_START,
    AXES_SERIAL,
    RobotArm,
)
from modules.motor.drive.emm42 import Direction
from modules.motor.drive.motion import (
    DEFAULT_INTER_FRAME_SECONDS,
    DEFAULT_PARALLEL_ROTATION_LIMIT_PULSES,
    estimate_move_seconds,
)
from modules.motion import CANCELLED, DONE, MotionTicket

# 独立协议样本：这些字面量故意不复用实现里的常量，字段写错仍然会挂。
POSITION_COMMAND = 0xFD
EMERGENCY_STOP_COMMAND = 0xFE
FRAME_TAIL = 0x6B
RELATIVE_POSITION = 0
BUFFERED_FOR_TRIGGER = 1
SAFE_TEST_RPM = 10
SAFE_TEST_ACCELERATION = 5
PULSES_PER_REVOLUTION = 3200

#: 多机同步触发帧：广播地址 + 0xFF 0x66。
SYNC_TRIGGER_FRAME = bytes((0x00, 0xFF, 0x66, FRAME_TAIL))

BASE_MOTOR_ID = 5
LIFT_MOTOR_ID = 6
EXTENSION_MOTOR_ID = 7


def fixture_config(**overrides) -> ArmConfig:
    """显式配置，避免这些用例跟着 modules/arm/config.json 的现场调参一起变。"""
    settings = dict(
        pulses_per_revolution=PULSES_PER_REVOLUTION,
        rpm=SAFE_TEST_RPM,
        acceleration=SAFE_TEST_ACCELERATION,
        default_pulses=800,
        max_pulses=5000,
        axes={
            "base": AxisConfig(motor_id=BASE_MOTOR_ID, reduction_ratio=6),
            "lift": AxisConfig(motor_id=LIFT_MOTOR_ID),
            "extension": AxisConfig(motor_id=EXTENSION_MOTOR_ID),
        },
    )
    settings.update(overrides)
    return ArmConfig(**settings)


def buffered_position_frame(
    *,
    motor_id: int,
    direction: Direction,
    pulses: int,
    rpm: int = SAFE_TEST_RPM,
    acceleration: int = SAFE_TEST_ACCELERATION,
) -> bytes:
    """一帧"缓存住、等广播触发"的相对位置命令（多机同步标志 = 01）。"""
    return (
        bytes((motor_id, POSITION_COMMAND, int(direction)))
        + rpm.to_bytes(2, "big")
        + bytes((acceleration,))
        + pulses.to_bytes(4, "big")
        + bytes((RELATIVE_POSITION, BUFFERED_FOR_TRIGGER, FRAME_TAIL))
    )


def emergency_stop_frame(*, motor_id: int) -> bytes:
    return bytes((motor_id, EMERGENCY_STOP_COMMAND, 0x98, 0, FRAME_TAIL))


class FakeClock:
    """假单调时钟，和 :class:`RecordingSleep` 一起推进——票据的等待是**轮询**到点的，
    不推进时钟就会一直等下去。"""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class RecordingSleep:
    """记录等待时长并推进假时钟，不真的睡。"""

    def __init__(self, clock: FakeClock) -> None:
        self.calls: list = []
        self._clock = clock

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
        self._clock.now += seconds


class FakeSerial:
    """内存总线：普通命令回一条 ACK，写入都记下来。"""

    def __init__(self) -> None:
        self._receive_buffer = bytearray()
        self.writes: list = []
        self.is_open = True

    def write(self, data: bytes) -> int:
        frame = bytes(data)
        self.writes.append(frame)
        self._receive_buffer.extend(bytes((frame[0], frame[1], 0x02, 0x6B)))
        return len(frame)

    def flush(self) -> None:
        pass

    def read(self, size: int) -> bytes:
        data = bytes(self._receive_buffer[:size])
        del self._receive_buffer[:size]
        return data

    def reset_input_buffer(self) -> None:
        """驱动每次发包前都会清缓冲：应答是 :meth:`write` 里当场产生的，这里没东西可清。"""
        self._receive_buffer.clear()

    def close(self) -> None:
        self.is_open = False


class FailingSerial(FakeSerial):
    """第 ``fail_on`` 次写入抛错：用来验证"缓存了却没触发"时的清理。

    装配时会先发三条使能帧，所以用例是在 ``make_arm`` 之后才 :meth:`fail_from`，
    让计数从动作帧开始。
    """

    def __init__(self) -> None:
        super().__init__()
        self.fail_on = None
        self.attempts = 0

    def fail_from(self, attempt: int) -> None:
        self.fail_on = attempt
        self.attempts = 0

    def write(self, data: bytes) -> int:
        self.attempts += 1
        if self.fail_on is not None and self.attempts == self.fail_on:
            raise RuntimeError("串口炸了")
        return super().write(data)


class ArmAxesTestCase(unittest.TestCase):
    def make_arm(self, serial=None, config=None, **settings):
        self.clock = FakeClock()
        self.sleep = RecordingSleep(self.clock)
        self.serial = serial if serial is not None else FakeSerial()
        self.arm = RobotArm._from_settings(
            config=config if config is not None else fixture_config(),
            port="/dev/ttyUSB0",
            bus=self.serial,
            sleep=self.sleep,
            clock=self.clock,
            **settings,
        )
        # 装配时已经使能三个轴（帧间也留了间隔）；这些用例只看之后的动作帧与等待。
        self.serial.writes.clear()
        self.sleep.calls.clear()
        return self.arm


class EnableFrameSpacingTests(unittest.TestCase):
    """三轴使能帧之间必须留间隔。

    5/6/7 里只有 5 号会回包、6/7 号不回，所以 ``expect_ack=False`` 的那两条之间没有
    任何自然的停顿；背靠背发帧会让帧在这台机器的 USB-TTL 上失效，驱动器回的那条
    ``地址 00 EE 6B``（错误命令）没人读走，下一个用总线的人就会拿它当自己的 ACK。
    """

    def test_enable_frames_are_spaced_by_the_inter_frame_gap(self) -> None:
        clock = FakeClock()
        sleep = RecordingSleep(clock)
        serial = FakeSerial()

        RobotArm._from_settings(
            config=fixture_config(),
            port="/dev/ttyUSB0",
            bus=serial,
            sleep=sleep,
            clock=clock,
        )

        self.assertEqual(
            [frame[0] for frame in serial.writes],
            [BASE_MOTOR_ID, LIFT_MOTOR_ID, EXTENSION_MOTOR_ID],
        )
        # 三条帧之间两个间隔，最后一条之后不再等。
        self.assertEqual(sleep.calls, [DEFAULT_INTER_FRAME_SECONDS] * 2)


class MoveAxesSameStartTests(ArmAxesTestCase):
    def test_buffers_every_axis_then_triggers_them_together(self) -> None:
        arm = self.make_arm()

        reports = arm.move_axes({"lift": 800, "extension": -400})

        self.assertEqual(
            self.serial.writes,
            [
                buffered_position_frame(
                    motor_id=LIFT_MOTOR_ID, direction=Direction.CW, pulses=800
                ),
                buffered_position_frame(
                    motor_id=EXTENSION_MOTOR_ID, direction=Direction.CCW, pulses=400
                ),
                SYNC_TRIGGER_FRAME,
            ],
        )
        self.assertEqual([item.motor_id for item in reports], [LIFT_MOTOR_ID, EXTENSION_MOTOR_ID])

    def test_frames_are_spaced_by_the_inter_frame_gap(self) -> None:
        """CH340 上背靠背两条位置帧会双双失效，所以帧间必须留间隔。"""
        arm = self.make_arm()

        arm.move_axes({"lift": 800, "extension": 400})

        # 两条帧之间留间隔，最后只等**最长**的那一根，不是各等一遍。
        self.assertEqual(self.sleep.calls[0], DEFAULT_INTER_FRAME_SECONDS)
        self.assertAlmostEqual(
            self.sleep.calls[-1],
            estimate_move_seconds(800, SAFE_TEST_RPM, PULSES_PER_REVOLUTION, SAFE_TEST_ACCELERATION),
            places=9,
        )
        self.assertEqual(len(self.sleep.calls), 2)

    def test_axes_are_sent_in_configuration_order_not_mapping_order(self) -> None:
        arm = self.make_arm()

        arm.move_axes({"extension": 100, "base": 200, "lift": 300})

        self.assertEqual(
            [frame[0] for frame in self.serial.writes],
            [BASE_MOTOR_ID, LIFT_MOTOR_ID, EXTENSION_MOTOR_ID, 0x00],
        )

    def test_zero_pulse_axes_are_not_sent_at_all(self) -> None:
        """协议不允许 0 位移：取 0 的轴一帧都不发，也不参与等待。"""
        arm = self.make_arm()

        reports = arm.move_axes({"lift": 800, "base": 0})

        self.assertEqual(len(reports), 1)
        self.assertEqual([frame[0] for frame in self.serial.writes], [LIFT_MOTOR_ID, 0x00])

    def test_no_moving_axis_sends_nothing_and_finishes_at_once(self) -> None:
        arm = self.make_arm()

        self.assertEqual(arm.move_axes({"lift": 0, "base": 0}), ())
        self.assertEqual(self.serial.writes, [])
        self.assertEqual(self.sleep.calls, [])

        ticket = arm.move_axes({}, wait=False)
        self.assertEqual(ticket.wait(), ())
        self.assertEqual(self.serial.writes, [])

    def test_every_axis_counts_its_pulses_by_the_shared_ledger(self) -> None:
        arm = self.make_arm()

        arm.move_axes({"lift": 800, "extension": -400})

        self.assertEqual(arm.turns_state.of("lift").pulses, 800)
        self.assertEqual(arm.turns_state.of("extension").pulses, -400)
        self.assertEqual(arm.turns_state.of("base").pulses, 0)

    def test_rejects_an_unknown_axis_before_sending_anything(self) -> None:
        arm = self.make_arm()

        with self.assertRaisesRegex(ValueError, "不是已知的轴"):
            arm.move_axes({"shoulder": 100})

        self.assertEqual(self.serial.writes, [])

    def test_rejects_pulses_that_are_not_integers(self) -> None:
        arm = self.make_arm()

        for value in (1.5, True, "100", None):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    arm.move_axes({"lift": value})
        self.assertEqual(self.serial.writes, [])

    def test_rejects_pulses_beyond_the_single_frame_cap(self) -> None:
        arm = self.make_arm()

        with self.assertRaises(ValueError):
            arm.move_axes({"lift": 5001})

        self.assertEqual(self.serial.writes, [])

    def test_rejects_targets_that_are_not_a_mapping(self) -> None:
        arm = self.make_arm()

        with self.assertRaisesRegex(ValueError, "映射"):
            arm.move_axes([("lift", 100)])

    def test_rejects_an_unknown_mode(self) -> None:
        arm = self.make_arm()

        with self.assertRaisesRegex(ValueError, "mode must be one of"):
            arm.move_axes({"lift": 100}, mode="turbo")


class MoveAxesSafetyTests(ArmAxesTestCase):
    def test_refuses_a_large_base_rotation_running_with_the_extension(self) -> None:
        """伸出状态下横扫会把整条臂扫过工作区：超限必须在**发帧之前**拒绝。"""
        arm = self.make_arm()

        with self.assertRaisesRegex(ValueError, AXES_SERIAL):
            arm.move_axes({"base": DEFAULT_PARALLEL_ROTATION_LIMIT_PULSES + 1, "extension": 100})

        self.assertEqual(self.serial.writes, [])

    def test_a_small_base_rotation_may_run_with_the_extension(self) -> None:
        arm = self.make_arm()

        arm.move_axes({"base": DEFAULT_PARALLEL_ROTATION_LIMIT_PULSES, "extension": 100})

        self.assertEqual(len(self.serial.writes), 3)

    def test_base_alone_may_rotate_any_distance(self) -> None:
        """没有伸缩参与时不存在"伸出状态下横扫"这回事。"""
        arm = self.make_arm()

        arm.move_axes({"base": 5000})

        self.assertEqual(self.serial.writes[-1], SYNC_TRIGGER_FRAME)

    def test_the_limit_can_be_tuned_at_construction(self) -> None:
        arm = self.make_arm(parallel_rotation_limit_pulses=0)

        with self.assertRaisesRegex(ValueError, AXES_SERIAL):
            arm.move_axes({"base": 1, "extension": 1})

    def test_a_frame_failure_really_clears_what_was_already_buffered(self) -> None:
        """缓存了却没触发是危险状态：下一次别人的广播会让这些轴突然动起来。"""
        serial = FailingSerial()
        arm = self.make_arm(serial=serial, config=fixture_config(max_pulses=5000))
        arm.pose_state.record_measurement(CylindricalPose(0.0, 250.0, 100.0))
        # 装配时发过三条使能帧；计数从这里开始，第 2 条动作帧（升降）发不出去。
        serial.fail_from(2)

        with self.assertRaisesRegex(RuntimeError, "串口炸了"):
            arm.move_axes({"base": 100, "lift": 200})

        self.assertEqual(
            serial.writes,
            [
                buffered_position_frame(
                    motor_id=BASE_MOTOR_ID, direction=Direction.CW, pulses=100
                ),
                # 先补一条触发把缓存消费掉，再停住已经起步的那根轴。
                SYNC_TRIGGER_FRAME,
                emergency_stop_frame(motor_id=BASE_MOTOR_ID),
            ],
        )
        self.assertFalse(arm.pose_state.is_known)


class MoveAxesSerialModeTests(ArmAxesTestCase):
    def test_serial_mode_walks_the_safety_corridor_one_axis_at_a_time(self) -> None:
        arm = self.make_arm()
        arm.pose_state.record_measurement(CylindricalPose(0.0, 250.0, 100.0))

        reports = arm.move_axes(
            {"base": 600, "lift": 800, "extension": -400}, mode=AXES_SERIAL
        )

        # 抬升 → 伸缩 → 底座：抬升在最前，底座在平面动作之后。
        self.assertEqual(
            [item.motor_id for item in reports],
            [LIFT_MOTOR_ID, EXTENSION_MOTOR_ID, BASE_MOTOR_ID],
        )
        self.assertEqual(
            [frame[0] for frame in self.serial.writes],
            [LIFT_MOTOR_ID, EXTENSION_MOTOR_ID, BASE_MOTOR_ID],
        )
        # 串行是**立即执行**（同步标志 0），不是缓存。
        self.assertEqual(self.serial.writes[0][-2], 0)

    def test_serial_mode_refuses_wait_false_because_they_contradict(self) -> None:
        arm = self.make_arm()

        with self.assertRaisesRegex(ValueError, "互相矛盾"):
            arm.move_axes({"lift": 800}, mode=AXES_SERIAL, wait=False)

    def test_the_default_mode_is_same_start(self) -> None:
        arm = self.make_arm()

        arm.move_axes({"lift": 100})
        buffered = arm.move_axes({"lift": 100}, mode=AXES_SAME_START)

        self.assertEqual(len(buffered), 1)
        self.assertEqual(self.serial.writes.count(SYNC_TRIGGER_FRAME), 2)


class MoveAxesTicketTests(ArmAxesTestCase):
    def test_wait_false_returns_a_ticket_and_marks_the_pose_unknown(self) -> None:
        arm = self.make_arm()
        arm.pose_state.record_measurement(CylindricalPose(0.0, 250.0, 100.0))

        ticket = arm.move_axes({"base": 3200, "lift": 400}, wait=False)

        self.assertIsInstance(ticket, MotionTicket)
        self.assertTrue(ticket.pending)
        # 只发了帧：什么时候到位不知道，位置不可信。
        self.assertFalse(arm.pose_state.is_known)
        self.assertEqual(ticket.subject, "arm:base+lift")
        # 整组等的是**最长**的那一根：底座 3200 脉冲比升降 400 脉冲慢得多。
        self.assertAlmostEqual(
            ticket.estimated_seconds,
            estimate_move_seconds(3200, SAFE_TEST_RPM, PULSES_PER_REVOLUTION, SAFE_TEST_ACCELERATION),
            places=9,
        )

    def test_waiting_the_ticket_finishes_the_group_and_advances_the_pose(self) -> None:
        arm = self.make_arm()
        arm.pose_state.record_measurement(CylindricalPose(0.0, 250.0, 100.0))
        ticket = arm.move_axes({"base": 3200}, wait=False)

        reports = ticket.wait()

        self.assertEqual(ticket.state, DONE)
        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0].motor_id, BASE_MOTOR_ID)
        # 3200 脉冲 = 电机一圈；底座减速比 6 → 输出轴 60°。
        self.assertTrue(arm.pose_state.is_known)
        self.assertAlmostEqual(arm.pose_state.pose.base_degrees, 60.0, places=9)

    def test_the_ticket_only_sleeps_once_for_the_longest_axis(self) -> None:
        arm = self.make_arm()
        ticket = arm.move_axes({"base": 4000, "lift": 800}, wait=False)
        before = list(self.sleep.calls)

        ticket.wait()

        # 建票据不睡；等到点只睡一次（估算模式下票据自己睡的那一段）。
        self.assertEqual(before, [DEFAULT_INTER_FRAME_SECONDS])
        self.assertAlmostEqual(
            sum(self.sleep.calls) - sum(before),
            estimate_move_seconds(4000, SAFE_TEST_RPM, PULSES_PER_REVOLUTION, SAFE_TEST_ACCELERATION),
            places=6,
        )

    def test_cancelling_the_ticket_stops_every_axis_of_the_group(self) -> None:
        arm = self.make_arm()
        ticket = arm.move_axes({"base": 100, "lift": 200}, wait=False)

        self.assertTrue(ticket.cancel())

        self.assertEqual(ticket.state, CANCELLED)
        self.assertEqual(
            self.serial.writes[-2:],
            [emergency_stop_frame(motor_id=BASE_MOTOR_ID), emergency_stop_frame(motor_id=LIFT_MOTOR_ID)],
        )
        # 取消不收尾：动作没走完，推进位姿会记错位置。
        self.assertFalse(arm.pose_state.is_known)

    def test_a_ticket_waited_after_close_does_not_resurrect_the_pose(self) -> None:
        """会话关了之后位姿就该保持未知：迟到的收尾不能把它"复活"。"""
        arm = self.make_arm()
        arm.pose_state.record_measurement(CylindricalPose(0.0, 250.0, 100.0))
        ticket = arm.move_axes({"base": 3200}, wait=False)

        arm.close()
        ticket.wait()

        self.assertFalse(arm.pose_state.is_known)

    def test_the_ticket_can_go_straight_into_a_motion_batch(self) -> None:
        from modules.motion import MotionBatch

        arm = self.make_arm()
        batch = MotionBatch(clock=self.clock, sleep=self.sleep)

        batch.add(arm.move_axes({"lift": 800}, wait=False))

        report = batch.wait()
        self.assertTrue(report.ok)
        self.assertEqual(report.items[0].subject, "arm:lift")


if __name__ == "__main__":
    unittest.main()
