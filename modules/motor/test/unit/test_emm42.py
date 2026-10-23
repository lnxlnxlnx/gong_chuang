"""Emm42 V5.0 command frames, verified without a physical serial device."""

from __future__ import annotations

import unittest

from modules.motor.drive.emm42 import (
    Direction,
    Emm42Driver,
    HomeMode,
    MotorStatus,
)


class FakeSerial:
    """Records writes and can acknowledge Emm42 commands like the driver does."""

    def __init__(
        self,
        *,
        acknowledge_emm42: bool = False,
        read_replies: dict[int, list[bytes]] | None = None,
        control_replies: dict[int, bytes | list[bytes]] | None = None,
        write_foreign: list[bytes] | None = None,
    ) -> None:
        self._receive_buffer = bytearray()
        self._acknowledge_emm42 = acknowledge_emm42
        #: 功能码 -> 依次返回的应答帧；最后一条会一直重复，用来表达"状态不再变化"。
        self._read_replies = {
            function: list(frames) for function, frames in (read_replies or {}).items()
        }
        #: 功能码 -> 该条控制命令的应答帧（给一列就按次取，最后一条重复），
        #: 用来表达"02 以外的状态"（如停止命令的 E2）与"第一次的回包被打坏了"。
        self._control_replies = {
            function: list(frames) if isinstance(frames, (list, tuple)) else [frames]
            for function, frames in (control_replies or {}).items()
        }
        #: 每次写入后、本命令应答之前先塞进缓冲的"别人的帧"（模拟另一台驱动器插话）。
        #: 它必须出现在**写入之后**才逼真：发命令前清缓冲清不掉"在路上"的帧。
        self._write_foreign = list(write_foreign or [])
        self.writes: list[bytes] = []
        #: 发命令前清输入缓冲的次数；驱动每次发包都该清一次（见 ``reset_input_buffer``）。
        self.drained = 0
        self.is_open = True

    @property
    def in_waiting(self) -> int:
        """真串口的"还有多少字节在输入缓冲里"；驱动用它判断总线静了没有。"""
        return len(self._receive_buffer)

    def write(self, data: bytes) -> int:
        frame = bytes(data)
        self.writes.append(frame)
        if self._write_foreign:
            self._receive_buffer.extend(self._write_foreign.pop(0))
        # 读取命令是 3 字节帧 [地址][功能码][校验]，驱动器按功能码回一条定长数据。
        # 读命令不参与控制命令的 ACK 机制，所以两条分支互斥（真机上也是这样：
        # Response=None 仍然会回答读命令，见说明书 4.2 / 8.1）。
        sequence = self._read_replies.get(frame[1]) if len(frame) == 3 else None
        control = self._control_replies.get(frame[1]) if len(frame) > 3 else None
        if sequence:
            self._receive_buffer.extend(
                sequence.pop(0) if len(sequence) > 1 else sequence[0]
            )
        elif control is not None:
            self._receive_buffer.extend(
                control.pop(0) if len(control) > 1 else control[0]
            )
        elif self._acknowledge_emm42 and len(frame) >= 3 and frame[-1] == 0x6B:
            self._receive_buffer.extend(bytes((frame[0], frame[1], 0x02, 0x6B)))
        return len(data)

    def flush(self) -> None:
        pass

    def read(self, size: int) -> bytes:
        data = bytes(self._receive_buffer[:size])
        del self._receive_buffer[:size]
        return data

    def reset_input_buffer(self) -> None:
        """丢掉还没被读走的字节：驱动每次发包前都会调一次。

        真驱动器的 ``Response=None`` 不回答合法控制命令，却会回答**被它判为非法**的帧；
        漏读一次就会毒到下一条命令。本假件的应答都在 :meth:`write` 里当场产生，
        所以这里清掉的只可能是测试自己塞进去的"残留"（见 :meth:`push_stale`）。
        """
        self.drained += 1
        self._receive_buffer.clear()

    def push_stale(self, data: bytes) -> None:
        """模拟"总线上别人留下的字节"（例如 ``expect_ack=False`` 那一侧漏读的错误帧）。"""
        self._receive_buffer.extend(data)

    def close(self) -> None:
        self.is_open = False


class FakeClock:
    """A monotonic clock that only moves when the injected sleep function is called."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def no_ack_driver(
    read_replies: dict[int, list[bytes]] | None = None,
) -> tuple[Emm42Driver, FakeSerial]:
    """A driver shaped like the arm's axes: ``Response=None``, so no control ACKs."""
    serial = FakeSerial(read_replies=read_replies)
    motor = Emm42Driver(
        "/dev/ttyUSB1", motor_id=1, serial_port=serial, expect_ack=False
    )
    return motor, serial


class Emm42DriverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.serial = FakeSerial(acknowledge_emm42=True)
        self.motor = Emm42Driver("/dev/ttyUSB1", motor_id=1, serial_port=self.serial)

    def test_initialize_keeps_injected_serial_port_open(self) -> None:
        self.assertIs(self.motor.initialize(), self.motor)
        self.assertTrue(self.serial.is_open)

    def test_enable_writes_enable_command(self) -> None:
        self.motor.enable(True)
        self.assertEqual(self.serial.writes, [bytes.fromhex("01 F3 AB 01 00 6B")])

    def test_enable_rejects_missing_driver_ack(self) -> None:
        motor = Emm42Driver("/dev/ttyUSB1", motor_id=1, serial_port=FakeSerial())

        with self.assertRaisesRegex(RuntimeError, "no ACK"):
            motor.enable(True)

    def test_enable_can_skip_ack_when_driver_response_is_none(self) -> None:
        serial = FakeSerial()
        motor = Emm42Driver(
            "/dev/ttyUSB1",
            motor_id=1,
            serial_port=serial,
            expect_ack=False,
        )

        motor.enable(True)

        self.assertEqual(serial.writes, [bytes.fromhex("01 F3 AB 01 00 6B")])

    def test_an_unacked_command_leaves_the_bus_idle_before_the_next_frame(self) -> None:
        """发完不管的命令后面必须留静默，否则下一帧会被这台 CH340 弄坏。

        arm 的 6/7 号就是这种驱动器：它们的使能帧发完 0 ms 就是底盘 1 号的使能帧，
        真机上坏掉的是后面那一帧——1 号收不到合法命令，缓冲里却多出一条 7 号的
        ``07 00 EE 6B``（错误命令）。
        """
        slept: list[float] = []
        serial = FakeSerial()
        motor = Emm42Driver(
            "/dev/ttyUSB1",
            motor_id=6,
            serial_port=serial,
            expect_ack=False,
            sleep=slept.append,
        )

        motor.enable(True)

        self.assertEqual(slept, [Emm42Driver.UNACKED_SETTLE_SECONDS])

    def test_an_acked_command_does_not_settle_because_the_ack_already_spaces_frames(self) -> None:
        """等回包的命令不用再加静默：读 ACK 的一个来回已经把两帧分开了。"""
        slept: list[float] = []
        serial = FakeSerial(acknowledge_emm42=True)
        motor = Emm42Driver(
            "/dev/ttyUSB1",
            motor_id=1,
            serial_port=serial,
            sleep=slept.append,
        )

        motor.enable(True)

        self.assertEqual(slept, [])

    def test_set_speed_encodes_direction_speed_and_acceleration(self) -> None:
        self.motor.set_speed(Direction.CCW, rpm=1500, acceleration=10)
        self.assertEqual(self.serial.writes, [bytes.fromhex("01 F6 01 05 DC 0A 00 6B")])

    def test_move_relative_converts_degrees_to_pulses(self) -> None:
        self.motor.move_relative(
            degrees=-360.0,
            rpm=1500,
            acceleration=0,
        )
        self.assertEqual(
            self.serial.writes,
            [bytes.fromhex("01 FD 01 05 DC 00 00 00 0C 80 00 00 6B")],
        )

    def test_move_relative_can_be_buffered_for_synchronized_start(self) -> None:
        self.motor.move_relative(degrees=90.0, rpm=120, acceleration=5, synchronized=True)
        self.assertEqual(
            self.serial.writes,
            [bytes.fromhex("01 FD 00 00 78 05 00 00 03 20 00 01 6B")],
        )

    def test_trigger_sync_motion_broadcasts_start_frame(self) -> None:
        self.motor.trigger_sync_motion()
        self.assertEqual(self.serial.writes, [bytes.fromhex("00 FF 66 6B")])

    def test_trigger_sync_motion_accepts_the_answering_drivers_address(self) -> None:
        """广播帧的回包地址是**回话那台驱动器自己的地址**，不是广播地址 00。

        真机上 ``00 FF 66 6B`` 收到的是 ``01 FF 02 6B``（1 号用自己的地址回话）。
        要求地址字节等于 0 会把这**条执行成功**的回包判成怪 ACK，四轮同步起步的脉冲动作
        当场抛异常（位置帧已经缓存、同步帧也已经跑掉了，报错纯属误报）。
        """
        serial = FakeSerial(control_replies={0xFF: bytes.fromhex("01 FF 02 6B")})
        motor = Emm42Driver("/dev/ttyUSB1", motor_id=1, serial_port=serial)

        motor.trigger_sync_motion()

        self.assertEqual(serial.writes, [bytes.fromhex("00 FF 66 6B")])

    def test_trigger_sync_motion_still_rejects_an_unaccepted_status(self) -> None:
        """放宽的只有地址字节：状态不是 ``02`` 的广播回包仍然报错，不能被吞掉。"""
        serial = FakeSerial(control_replies={0xFF: bytes.fromhex("01 FF E2 6B")})
        motor = Emm42Driver("/dev/ttyUSB1", motor_id=1, serial_port=serial)

        with self.assertRaisesRegex(RuntimeError, "unexpected ACK: 01 ff e2 6b"):
            motor.trigger_sync_motion()

    def test_a_broadcast_ack_is_matched_by_function_not_by_address(self) -> None:
        """广播没有地址可过滤，功能码对不上的整帧不能顶替同步 ACK。

        清输入缓冲挡不住"在路上"的迟到帧（另一个模块的回包在广播帧之后才落到缓冲里），
        所以广播这一侧必须靠功能码认自己的回包；认出来是别人的就记进丢弃清单继续等。
        """
        serial = FakeSerial(control_replies={0xFF: bytes.fromhex("01 F3 02 6B")})
        motor = Emm42Driver("/dev/ttyUSB1", motor_id=1, serial_port=serial)

        with self.assertRaisesRegex(RuntimeError, "received no ACK.*01 f3 02 6b"):
            motor.trigger_sync_motion()

    def test_move_absolute_encodes_absolute_mode(self) -> None:
        self.motor.move_absolute(
            degrees=360.0,
            rpm=1000,
            acceleration=8,
        )
        self.assertEqual(
            self.serial.writes,
            [bytes.fromhex("01 FD 00 03 E8 08 00 00 0C 80 01 00 6B")],
        )

    def test_stop_writes_immediate_stop_command(self) -> None:
        self.motor.stop()
        self.assertEqual(self.serial.writes, [bytes.fromhex("01 FE 98 00 6B")])

    def test_stop_accepts_already_stopped_status(self) -> None:
        serial = FakeSerial(control_replies={0xFE: bytes.fromhex("01 FE E2 6B")})
        motor = Emm42Driver("/dev/ttyUSB1", motor_id=1, serial_port=serial)

        motor.stop()

        self.assertEqual(serial.writes, [bytes.fromhex("01 FE 98 00 6B")])

    def test_stale_input_is_drained_before_every_command(self) -> None:
        """总线上别人的字节不能算作自己的 ACK。

        arm(5/6/7) 与 chassis(1~4) 共用一条总线，``expect_ack=False`` 的驱动器发完命令
        不读应答；缓冲里留着的那 4 字节会顶替下一条命令的 ACK，报出来的地址还是别人的
        （真机上见过用 1 号的名义报出 6 号的错误帧 ``06 00 EE 6B``）。
        """
        serial = FakeSerial(acknowledge_emm42=True)
        serial.push_stale(bytes.fromhex("06 00 EE 6B"))
        motor = Emm42Driver("/dev/ttyUSB1", motor_id=1, serial_port=serial)

        motor.enable(True)

        self.assertEqual(serial.drained, 1)
        self.assertEqual(serial.writes, [bytes.fromhex("01 F3 AB 01 00 6B")])

    def test_a_foreign_frame_is_skipped_instead_of_becoming_our_ack(self) -> None:
        """别人在**发包之后**才到线的帧也拦得住。

        清缓冲清不掉"在路上"的应答：arm 的 6/7 号是 ``expect_ack=False``，发完就走，
        它们的错误帧是在底盘开始等 ACK 之后几毫秒才到的（真机上是 ``07 00 EE 6B``）。
        这不是自己的帧，跳过继续等，最后读到的仍是自己的 ``01 F3 02 6B``。
        """
        serial = FakeSerial(
            acknowledge_emm42=True,
            write_foreign=[bytes.fromhex("07 00 EE 6B")],
        )
        motor = Emm42Driver("/dev/ttyUSB1", motor_id=1, serial_port=serial)

        motor.enable(True)

        self.assertEqual(serial.writes, [bytes.fromhex("01 F3 AB 01 00 6B")])

    def test_skipped_foreign_frames_are_named_in_the_error(self) -> None:
        """等了半天只有别人的帧：报错要点名，不能再写成"1 号轮 ACK 不对"。"""
        serial = FakeSerial(write_foreign=[bytes.fromhex("07 00 EE 6B")])
        motor = Emm42Driver("/dev/ttyUSB1", motor_id=1, serial_port=serial)

        with self.assertRaisesRegex(RuntimeError, r"no ACK.*07 00 ee 6b"):
            motor.enable(True)

    def test_home_writes_requested_home_mode(self) -> None:
        self.motor.home(HomeMode.MULTI_TURN_COLLISION)
        self.assertEqual(self.serial.writes, [bytes.fromhex("01 9A 02 00 6B")])

    def test_absolute_zero_is_allowed_because_it_means_go_to_the_origin(self) -> None:
        """绝对模式下 0 是"回原点"，是回零之后归位的唯一一条命令形式。"""
        self.motor.move_absolute_pulses(0, rpm=300, acceleration=10)

        self.assertEqual(
            self.serial.writes,
            [bytes.fromhex("01 FD 00 01 2C 0A 00 00 00 00 01 00 6B")],
        )

    def test_relative_zero_is_still_rejected(self) -> None:
        """相对模式下 0 位移没有意义，照旧拒绝。"""
        with self.assertRaisesRegex(ValueError, "relative"):
            self.motor.move_relative_pulses(0, rpm=300, acceleration=10)

        self.assertEqual(self.serial.writes, [])

    def test_absolute_negative_position_still_encodes_ccw(self) -> None:
        self.motor.move_absolute_pulses(-3200, rpm=300, acceleration=10)

        self.assertEqual(
            self.serial.writes,
            [bytes.fromhex("01 FD 01 01 2C 0A 00 00 0C 80 01 00 6B")],
        )

    def test_set_speed_rejects_invalid_rpm(self) -> None:
        with self.assertRaises(ValueError):
            self.motor.set_speed(Direction.CW, rpm=3001, acceleration=0)


class ChunkedFakeSerial(FakeSerial):
    """每次 ``read`` 最多回几字节：模拟一条回包跨两次读才凑齐（USB 分包）。"""

    def __init__(self, *, chunk: int = 2, **kwargs) -> None:
        super().__init__(**kwargs)
        self._chunk = chunk

    def read(self, size: int) -> bytes:
        return super().read(min(size, self._chunk))


class ArrivingFakeSerial(FakeSerial):
    """清缓冲时帧还在路上：``in_waiting`` 前几次报"还有字节在来"，之后才安静。"""

    def __init__(self, *, arriving: int = 2, **kwargs) -> None:
        super().__init__(**kwargs)
        self._arriving = arriving

    @property
    def in_waiting(self) -> int:
        return self._arriving if self._arriving > 0 else len(self._receive_buffer)

    def reset_input_buffer(self) -> None:
        super().reset_input_buffer()
        if self._arriving > 0:
            self._arriving -= 1


class Emm42GarbledReplyTests(unittest.TestCase):
    """回包被打坏时怎么办：坏帧不当成自己的 ACK、幂等命令重发、相对位置帧不重发。

    真机现场（``app/motion.py``）：5 号轴偶尔报
    ``command 0xFD returned unexpected ACK: 05 7f a0 ed``，紧接着清理时的失能报
    ``command 0xF3 received no ACK (discarded foreign frames: 45 f3 02 6b)``——
    后者是一条**结构完整**的 F3 应答，只有地址字节从 ``05`` 变成 ``45``（差一位）。
    也就是说命令确实到了驱动器、也执行了，坏的是读回来的字节。
    """

    #: 真机读到的坏帧：地址是自己的，功能码/状态/帧尾全是垃圾（合法帧必须以 6B 结尾）。
    GARBAGE = bytes.fromhex("01 7F A0 ED")

    def test_a_garbled_frame_is_named_instead_of_being_our_ack(self) -> None:
        """四个字节里只有地址对得上时，不能当成"驱动器回了怪东西"。

        报成 ``unexpected ACK: 01 7f a0 ed`` 会把人引向"驱动器坏了"；真相是这条回包
        在线上被打坏了，所以应该报"没读到合格回包"，并把线上那些字节原样列出来。
        """
        serial = FakeSerial(control_replies={0xF3: self.GARBAGE})
        motor = Emm42Driver("/dev/ttyUSB1", motor_id=1, serial_port=serial)

        with self.assertRaisesRegex(RuntimeError, r"no ACK.*01 7f a0 ed"):
            motor.enable(True)

    def test_an_idempotent_command_is_resent_when_its_reply_is_garbled(self) -> None:
        """失能/使能是幂等的：回包坏掉一次就再发一遍，别把整个清理流程拖崩。

        真机上第一段 traceback 就是这里：关闭时 ``disable`` 报 no ACK，把
        ``RobotArm`` 清理失败一路带成 ``Robot 清理失败``，业务异常反而被盖掉。
        """
        serial = FakeSerial(
            control_replies={
                0xF3: [self.GARBAGE, bytes.fromhex("01 F3 02 6B")],
            }
        )
        motor = Emm42Driver("/dev/ttyUSB1", motor_id=1, serial_port=serial)

        motor.enable(False)

        self.assertEqual(
            serial.writes,
            [bytes.fromhex("01 F3 AB 00 00 6B"), bytes.fromhex("01 F3 AB 00 00 6B")],
        )

    def test_a_relative_move_is_not_resent_because_that_would_move_twice(self) -> None:
        """相对位置帧表达的是"再走这么多"：重发一次就是让那根轴多走一段。"""
        serial = FakeSerial(
            control_replies={
                0xFD: [self.GARBAGE, bytes.fromhex("01 FD 02 6B")],
            }
        )
        motor = Emm42Driver("/dev/ttyUSB1", motor_id=1, serial_port=serial)

        with self.assertRaisesRegex(RuntimeError, r"no ACK.*01 7f a0 ed"):
            motor.move_relative_pulses(800, rpm=80, acceleration=200)

        self.assertEqual(len(serial.writes), 1)

    def test_an_absolute_move_is_resent_because_its_target_does_not_change(self) -> None:
        """绝对位置帧是幂等的：目标不变，重发一遍结果一样，所以允许重发。"""
        serial = FakeSerial(
            control_replies={
                0xFD: [self.GARBAGE, bytes.fromhex("01 FD 02 6B")],
            }
        )
        motor = Emm42Driver("/dev/ttyUSB1", motor_id=1, serial_port=serial)

        motor.move_absolute_pulses(3200, rpm=300, acceleration=10)

        self.assertEqual(len(serial.writes), 2)
        self.assertEqual(serial.writes[0], serial.writes[1])

    def test_a_real_rejection_is_reported_instead_of_being_resent(self) -> None:
        """驱动器回过话（哪怕是否定的）就是真结论：重发既没意义也会盖掉它。"""
        serial = FakeSerial(control_replies={0xF3: bytes.fromhex("01 F3 E2 6B")})
        motor = Emm42Driver("/dev/ttyUSB1", motor_id=1, serial_port=serial)

        with self.assertRaisesRegex(RuntimeError, r"unexpected ACK: 01 f3 e2 6b"):
            motor.enable(True)

        self.assertEqual(len(serial.writes), 1)

    def test_a_reply_split_across_two_reads_is_assembled(self) -> None:
        """回包跨两次读才到齐（USB 分包）时要拼起来，不能拿半帧去判对错。"""
        serial = ChunkedFakeSerial(chunk=2, acknowledge_emm42=True)
        motor = Emm42Driver("/dev/ttyUSB1", motor_id=1, serial_port=serial)

        motor.enable(True)

        self.assertEqual(serial.writes, [bytes.fromhex("01 F3 AB 01 00 6B")])

    def test_the_reached_packet_is_not_mistaken_for_a_control_ack(self) -> None:
        """``Response=Both`` 时驱动器运动结束会异步发 ``01 FD 9F 6B``。

        它不是任何一条命令的 ACK，撞进读窗口时丢掉继续等，而不是把下一条命令判成失败。
        """
        serial = FakeSerial(
            acknowledge_emm42=True,
            write_foreign=[bytes.fromhex("01 FD 9F 6B")],
        )
        motor = Emm42Driver("/dev/ttyUSB1", motor_id=1, serial_port=serial)

        motor.enable(True)

        self.assertEqual(serial.writes, [bytes.fromhex("01 F3 AB 01 00 6B")])

    def test_the_bus_is_waited_out_when_bytes_are_still_arriving(self) -> None:
        """清缓冲时帧还在路上：要等到线安静再发，否则剩下的半帧会被拼进自己的回包。"""
        slept: list[float] = []
        serial = ArrivingFakeSerial(arriving=2, acknowledge_emm42=True)
        motor = Emm42Driver(
            "/dev/ttyUSB1",
            motor_id=1,
            serial_port=serial,
            sleep=slept.append,
        )

        motor.enable(True)

        self.assertEqual(serial.drained, 2)
        self.assertEqual(slept, [Emm42Driver.QUIET_POLL_SECONDS])

    def test_a_quiet_bus_costs_nothing(self) -> None:
        """总线本来就静着时不清第二把、也不睡——与加这道闸之前完全一样。"""
        slept: list[float] = []
        serial = FakeSerial(acknowledge_emm42=True)
        motor = Emm42Driver(
            "/dev/ttyUSB1",
            motor_id=1,
            serial_port=serial,
            sleep=slept.append,
        )

        motor.enable(True)

        self.assertEqual(serial.drained, 1)
        self.assertEqual(slept, [])


class Emm42ReadCommandTests(unittest.TestCase):
    """读取类命令：帧格式、返回长度、符号解码与错误帧（说明书 6.3.2 / 6.3.4）。"""

    def test_position_reply_is_eight_bytes_because_of_the_sign_byte(self) -> None:
        """7 字节是错的：位置值前面还有一个符号字节，少算一个字节整帧就错位。"""
        self.assertEqual(Emm42Driver.REPLY_LENGTHS[0x36], 8)
        self.assertEqual(Emm42Driver.REPLY_LENGTHS[0x3A], 4)
        self.assertEqual(Emm42Driver.REPLY_LENGTHS[0x3B], 4)

    def test_read_position_matches_the_manual_example_and_decodes_negative(
        self,
    ) -> None:
        """说明书 6.3.4 的示例：01 36 01 00 01 00 00 6B 即 -0x00010000 = -360.0°。"""
        motor, serial = no_ack_driver(
            {0x36: [bytes.fromhex("01 36 01 00 01 00 00 6B")]}
        )

        self.assertEqual(motor.read_position(), -65536)
        self.assertEqual(serial.writes, [bytes.fromhex("01 36 6B")])

    def test_read_position_decodes_a_positive_payload(self) -> None:
        motor, _ = no_ack_driver({0x36: [bytes.fromhex("01 36 00 00 00 0C 80 6B")]})

        self.assertEqual(motor.read_position(), 3200)

    def test_read_position_rejects_an_unknown_sign_byte(self) -> None:
        """符号字节只有 00/01；别的值说明帧已经错位，不能猜出一个"看起来合理"的数。"""
        motor, _ = no_ack_driver({0x36: [bytes.fromhex("01 36 02 00 00 0C 80 6B")]})

        with self.assertRaisesRegex(RuntimeError, "sign byte"):
            motor.read_position()

    def test_read_encoder_returns_the_linearised_single_turn_value(self) -> None:
        """0x31 是单圈绝对值（说明书 6.3.4 示例 0x8D9E = 36254），区分不了圈数。"""
        motor, serial = no_ack_driver({0x31: [bytes.fromhex("01 31 8D 9E 6B")]})

        self.assertEqual(motor.read_encoder(), 36254)
        self.assertEqual(serial.writes, [bytes.fromhex("01 31 6B")])

    def test_read_status_decodes_every_documented_bit(self) -> None:
        motor, _ = no_ack_driver({0x3A: [bytes.fromhex("01 3A 0F 6B")]})

        status = motor.read_status()

        self.assertTrue(status.enabled)
        self.assertTrue(status.in_position)
        self.assertTrue(status.stalled)
        self.assertTrue(status.stall_protected)

    def test_read_status_clears_every_bit_on_zero(self) -> None:
        motor, _ = no_ack_driver({0x3A: [bytes.fromhex("01 3A 00 6B")]})

        self.assertEqual(
            motor.read_status(),
            MotorStatus(
                enabled=False, in_position=False, stalled=False, stall_protected=False
            ),
        )

    def test_read_homing_status_separates_failure_from_still_running(self) -> None:
        """0x0C = 正在回零(0x04) + 回零失败(0x08)：两个位是独立的，不能只看一个。"""
        motor, _ = no_ack_driver({0x3B: [bytes.fromhex("01 3B 0C 6B")]})

        status = motor.read_homing_status()

        self.assertTrue(status.homing)
        self.assertTrue(status.failed)
        self.assertFalse(status.encoder_ready)
        self.assertFalse(status.calibration_ready)

    def test_read_commands_ignore_the_control_command_ack_setting(self) -> None:
        """``Response`` 只管控制动作命令，读命令两种设置下都必须能读（4.2 / 8.1）。"""
        for expect_ack in (True, False):
            with self.subTest(expect_ack=expect_ack):
                serial = FakeSerial(read_replies={0x3A: [bytes.fromhex("01 3A 03 6B")]})
                motor = Emm42Driver(
                    "/dev/ttyUSB1",
                    motor_id=1,
                    serial_port=serial,
                    expect_ack=expect_ack,
                )

                self.assertTrue(motor.read_status().in_position)
                self.assertEqual(serial.writes, [bytes.fromhex("01 3A 6B")])

    def test_read_command_surfaces_the_driver_error_frame(self) -> None:
        motor, _ = no_ack_driver({0x36: [bytes.fromhex("01 00 EE 6B")]})

        with self.assertRaisesRegex(RuntimeError, "rejected read command"):
            motor.read_position()

    def test_read_command_rejects_a_short_reply(self) -> None:
        motor, _ = no_ack_driver({0x36: [bytes.fromhex("01 36 00 00 0C 80 6B")]})

        with self.assertRaisesRegex(RuntimeError, "unexpected reply"):
            motor.read_position()

    def test_read_command_reports_a_missing_reply(self) -> None:
        motor, _ = no_ack_driver()

        with self.assertRaisesRegex(RuntimeError, "no reply"):
            motor.read_position()

    def test_read_command_rejects_an_unknown_function_code_before_writing(self) -> None:
        motor, serial = no_ack_driver()

        with self.assertRaises(ValueError):
            motor._read(0x99)

        self.assertEqual(serial.writes, [])


class Emm42WaitForPositionTests(unittest.TestCase):
    """``0x3A`` 轮询：先等到位位离开，再等它回来；超时与堵转都要能被调用方看见。"""

    def poll(self, flags_sequence, **kwargs):
        serial = FakeSerial(
            read_replies={
                0x3A: [bytes((0x01, 0x3A, flags, 0x6B)) for flags in flags_sequence]
            }
        )
        motor = Emm42Driver(
            "/dev/ttyUSB1", motor_id=1, serial_port=serial, expect_ack=False
        )
        clock = FakeClock()
        status = motor.wait_until_in_position(
            kwargs.pop("timeout", 5.0),
            poll_interval=0.25,
            sleep=clock.sleep,
            clock=clock,
            **kwargs,
        )
        return status, clock

    def test_waits_for_the_stale_in_position_bit_to_clear_before_waiting_for_it_again(
        self,
    ) -> None:
        """刚发完命令时到位位还留着上一次动作的结果；直接信它会等于根本没等。"""
        status, clock = self.poll([0x03, 0x01, 0x01, 0x03])

        self.assertTrue(status.in_position)
        self.assertEqual(len(clock.sleeps), 3)

    def test_returns_the_last_status_when_the_move_never_settles(self) -> None:
        status, clock = self.poll([0x01], timeout=1.0)

        self.assertFalse(status.in_position)
        self.assertFalse(status.stalled)
        self.assertGreaterEqual(clock.now, 1.0)

    def test_gives_up_once_the_axis_reports_a_stall_instead_of_burning_the_timeout(
        self,
    ) -> None:
        """堵转不会再到位：继续等只是白等，而且会把"撞上了"拖成一次超时。"""
        status, clock = self.poll([0x01, 0x05], timeout=30.0)

        self.assertTrue(status.stalled)
        self.assertFalse(status.in_position)
        self.assertLess(clock.now, 30.0)

    def test_gives_up_once_stall_protection_has_tripped(self) -> None:
        status, clock = self.poll([0x09], timeout=30.0)

        self.assertTrue(status.stall_protected)
        self.assertEqual(clock.sleeps, [])

    def test_an_already_settled_move_returns_without_burning_the_timeout(self) -> None:
        status, clock = self.poll([0x03], timeout=30.0, departure_timeout=0.0)

        self.assertTrue(status.in_position)
        self.assertEqual(clock.sleeps, [])

    def test_rejects_invalid_timing_arguments(self) -> None:
        motor, _ = no_ack_driver({0x3A: [bytes.fromhex("01 3A 00 6B")]})

        with self.assertRaises(ValueError):
            motor.wait_until_in_position(-1.0)
        with self.assertRaises(ValueError):
            motor.wait_until_in_position(1.0, poll_interval=0.0)
        with self.assertRaises(ValueError):
            motor.wait_until_in_position(1.0, departure_timeout=-0.5)


if __name__ == "__main__":
    unittest.main()
