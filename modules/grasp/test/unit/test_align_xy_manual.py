"""菜单 21：z 段（``--align-z``）、普通动作会话装配，以及"失败就不发动作"。

这一段是**开环**的——升降下降、伸缩后缩，走完之前画面里就看不到物块了，
所以真机上出错没有第二次机会。能在这里钉死的就是三条：

1. 顺序是"先下降、再后缩"；
2. 两个脉冲数来自配置，命令行的覆盖只作用于这一次；
3. 平面没对准就**一个动作帧都不发**（脉冲数只保证往下走了那么多，不保证下面有物块）。

普通动作（默认对准 / ``--align-and-descend``）现在统一走
:meth:`modules.grasp.runtime.Grasp._open_settings` 这一条设备创建路径：菜单只负责
提示与打印，动作交给实例方法。这里的断言因此是"菜单有没有把动作和覆盖原样交出去"，
以及"会话装配只发生一次、而且校验在前"。
"""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import Mock, patch

from modules.grasp.drive.servo_model import (
    GraspReadyResult,
    LiftScale,
    ObjectSelector,
    ServoConfig,
    ServoError,
    ServoJacobian,
    ServoResult,
    ServoTarget,
    ZDescentResult,
)
from modules.grasp.test.manual import align_xy as menu
from modules.grasp.test.manual.align_xy import main, run_align, run_descend_z, run_pick_ready
from modules.vision.drive.sighting import parse_sighting


def sighting(x: float, y: float):
    return parse_sighting(f"物料 4: green物料（green），xyz=({x:.6f},{y:.6f},113.2) mm")


def healthy_jacobian() -> ServoJacobian:
    return ServoJacobian(base_dx=0.012, base_dy=0.0035, radial_dx=-0.0021, radial_dy=0.0091)


class FakeArm:
    """只记动作：升降与伸缩各记一条，够断言顺序和脉冲数。

    ``stated_seconds`` 对应真臂 ``move_*`` 返回的动作报告里的 ``estimated_seconds``：
    脉冲原语会拿它估算等待时间，所以假臂也得给出一个数（给 0 才不用真等）。
    """

    def __init__(self, *, stated_seconds: float = 0.0) -> None:
        self.calls: list = []
        self.stated_seconds = stated_seconds

    def _report(self):
        return SimpleNamespace(estimated_seconds=self.stated_seconds)

    def move_down(self, pulses, **_kwargs):
        self.calls.append(("down", pulses))
        return self._report()

    def move_up(self, pulses, **_kwargs):
        self.calls.append(("up", pulses))
        return self._report()

    def move_backward(self, pulses, **_kwargs):
        self.calls.append(("back", pulses))
        return self._report()

    def move_forward(self, pulses, **_kwargs):
        self.calls.append(("forward", pulses))
        return self._report()


def arm_double(calls: list, *, estimated_seconds: float = 0.0) -> Mock:
    """一个"既有生命周期、又会如实上报动作时长"的假臂。

    低层工具分支会走 ``initialize`` / ``enable_all`` / ``stop_all`` / ``disable_all`` /
    ``close``；而 ``descend_z`` / ``retract_extension`` 会真的发脉冲并读动作报告里的
    ``estimated_seconds``。裸 ``Mock`` 的那个属性是 Mock 本身，``float()`` 会当场炸，
    所以两个动作方法要显式给一个数（0 才不用真等）。
    """
    report = SimpleNamespace(estimated_seconds=estimated_seconds)
    arm = Mock()

    def record(name):
        def action(pulses, **_kwargs):
            calls.append((name, pulses))
            return report

        return action

    arm.move_down.side_effect = record("down")
    arm.move_backward.side_effect = record("back")
    arm.move_up.side_effect = record("up")
    arm.move_forward.side_effect = record("forward")
    return arm


class FakeSource:
    """画面静止的假镜头：连续两次读数一致，取帧立刻返回。"""

    def __init__(self, x: float, y: float) -> None:
        self.x, self.y = x, y
        self.error = None

    def frame(self, **_kwargs):
        return (sighting(self.x, self.y),)


class FakeGrasp:
    """假会话：只提供菜单要读的四个值对象与两个动作方法。

    ``_config`` / ``_selector`` / ``_target`` / ``_jacobian`` 是提示横幅要用的
    **本会话已有**的值（工具内部使用，不是上层接口），所以假件必须带着它们——
    少了就是"横幅那几行会崩"，而真机上是按菜单 21 的人第一眼看到的东西。
    """

    def __init__(self, *, config=None, target=None, jacobian=None, result=None, error=None):
        self._config = config if config is not None else ServoConfig()
        self._selector = ObjectSelector()
        self._target = target if target is not None else ServoTarget(x_mm=0.0, y_mm=0.0)
        self._jacobian = jacobian if jacobian is not None else healthy_jacobian()
        self._lift_scale = LiftScale(dz_per_pulse=0.0121)
        self.result = result
        self.error = error
        self.align_calls = 0
        self.descend_calls: list = []

    def align_xy(self):
        self.align_calls += 1
        if self.error is not None:
            raise self.error
        return self.result

    def align_and_descend(self, *, z_pulses=None, retract_pulses=None):
        self.descend_calls.append((z_pulses, retract_pulses))
        if self.error is not None:
            raise self.error
        return self.result


def canned_result() -> GraspReadyResult:
    return GraspReadyResult(
        raised_pulses=250,
        aligned=ServoResult(
            target=ServoTarget(x_mm=0.0, y_mm=0.0),
            readings=((0.3, -0.2),),
            error_x=0.3,
            error_y=-0.2,
            base_pulses=8,
            radial_pulses=-3,
            iterations=1,
            tolerance_mm=2.0,
        ),
        descent=ZDescentResult(
            pulses=2200, planar_mm=0.36, start_z_mm=100.0, seconds=1.2
        ),
        retracted_pulses=150,
        lift_seconds=1.4,
        align_seconds=1.1,
        descend_seconds=1.6,
        retract_seconds=0.2,
        hover_pulses=-120,
    )


def canned_align_result() -> ServoResult:
    return ServoResult(
        target=ServoTarget(x_mm=0.0, y_mm=0.0),
        readings=((0.3, -0.2),),
        error_x=0.3,
        error_y=-0.2,
        base_pulses=8,
        radial_pulses=-3,
        iterations=1,
        tolerance_mm=2.0,
    )


class DescendZTests(unittest.TestCase):
    def run_step(self, *, x: float = 1.0, y: float = -1.0, **overrides):
        """跑一次菜单 21 的 z 段，返回 (退出码, 假臂, 打出来的字)。"""
        arm = FakeArm()
        printed = io.StringIO()
        with redirect_stdout(printed):
            status = run_descend_z(
                arm=arm,
                source=FakeSource(x, y),
                selector=ObjectSelector(),
                config=ServoConfig(),
                **overrides,
            )
        return status, arm, printed.getvalue()

    def test_the_z_step_descends_first_and_then_retracts(self) -> None:
        config = ServoConfig()
        status, arm, printed = self.run_step()
        self.assertEqual(status, 0)
        # 脉冲数跟着配置走（现场会改这两个数），不写死具体值。
        self.assertEqual(
            arm.calls,
            [("down", config.z_descend_pulses), ("back", config.extension_retract_pulses)],
        )
        # 两个数都要打在屏幕上：现场就是照着这两行核对配置改对没有。
        self.assertIn(str(config.z_descend_pulses), printed)
        self.assertIn(str(config.extension_retract_pulses), printed)

    def test_the_overrides_replace_both_counts_for_this_run_only(self) -> None:
        before = ServoConfig()
        status, arm, _ = self.run_step(z_pulses=800, retract_pulses=20)
        self.assertEqual(status, 0)
        self.assertEqual(arm.calls, [("down", 800), ("back", 20)])
        # 覆盖不写回配置：下一次还是配置文件里的值。
        after = ServoConfig()
        self.assertEqual(after.z_descend_pulses, before.z_descend_pulses)
        self.assertEqual(after.extension_retract_pulses, before.extension_retract_pulses)

    def test_a_plane_that_is_not_aligned_refuses_without_moving(self) -> None:
        status, arm, printed = self.run_step(x=20.0, y=0.0)
        self.assertEqual(status, 1)
        self.assertEqual(arm.calls, [])
        self.assertIn("没有发出动作", printed)

    def test_a_nonsense_override_stops_before_moving(self) -> None:
        status, arm, _ = self.run_step(z_pulses=0)
        self.assertEqual(status, 1)
        self.assertEqual(arm.calls, [])


class PickReadyTests(unittest.TestCase):
    """一键就位（``--align-and-descend``）：四步都交给会话的 ``align_and_descend``。

    菜单**不再**自己拼装 ``arm`` / ``link`` / ``selector`` / ``target`` / ``jacobian``
    / ``config`` 六个参数去调自由函数——那正是这一批重构消掉的东西。
    """

    def run_it(self, grasp, **kwargs):
        printed = io.StringIO()
        with redirect_stdout(printed):
            status = run_pick_ready(grasp=grasp, **kwargs)
        return status, printed.getvalue()

    def test_a_run_forwards_the_overrides_to_the_session_method(self) -> None:
        grasp = FakeGrasp(result=canned_result())
        status, printed = self.run_it(grasp, z_pulses=800, retract_pulses=20)

        self.assertEqual(status, 0)
        self.assertEqual(grasp.descend_calls, [(800, 20)])
        self.assertIn("就位完成", printed)
        self.assertIn("共 4.3 s", printed)
        self.assertIn("下一步：合上夹爪", printed)

    def test_without_overrides_the_two_counts_stay_none(self) -> None:
        """``None`` 是"用配置里的值"；菜单不能替它填数——那会让配置失去意义。"""
        grasp = FakeGrasp(result=canned_result())
        status, _printed = self.run_it(grasp)

        self.assertEqual(status, 0)
        self.assertEqual(grasp.descend_calls, [(None, None)])

    def test_a_failed_run_returns_one_and_prints_the_reason(self) -> None:
        grasp = FakeGrasp(error=ServoError("平面还没对准就下降"))
        status, printed = self.run_it(grasp)

        self.assertEqual(status, 1)
        self.assertIn("就位失败", printed)
        self.assertIn("平面还没对准", printed)


class AlignTests(unittest.TestCase):
    """默认对准（真动）：动作走 ``grasp.align_xy()``，横幅读的是会话里的值。"""

    def run_it(self, grasp):
        printed = io.StringIO()
        with redirect_stdout(printed):
            status = run_align(grasp=grasp)
        return status, printed.getvalue()

    def test_the_action_goes_through_the_session_method(self) -> None:
        grasp = FakeGrasp(result=canned_align_result())
        status, printed = self.run_it(grasp)

        self.assertEqual(status, 0)
        self.assertEqual(grasp.align_calls, 1)
        self.assertIn("x/y 对准完成", printed)
        self.assertIn("下一步：--align-z", printed)

    def test_the_banner_reads_the_session_values(self) -> None:
        """横幅必须用到本会话的四个值对象；缺一个就会在真机上当场崩。"""
        grasp = FakeGrasp(result=canned_align_result())
        _status, printed = self.run_it(grasp)

        self.assertIn(grasp._selector.describe(), printed)
        self.assertIn(grasp._target.summary(), printed)
        self.assertIn(grasp._jacobian.summary(), printed)

    def test_the_hover_banner_only_appears_when_hover_is_configured(self) -> None:
        configured = FakeGrasp(
            config=ServoConfig(hover_z_mm=105.0), result=canned_align_result()
        )
        _status, printed = self.run_it(configured)
        self.assertIn("悬停高度", printed)

        plain = FakeGrasp(config=ServoConfig(hover_z_mm=None), result=canned_align_result())
        _status, plain_printed = self.run_it(plain)
        self.assertNotIn("悬停高度", plain_printed)

    def test_a_failed_align_returns_one_and_prints_the_reason(self) -> None:
        grasp = FakeGrasp(error=ServoError("稳定看到物块"))
        status, printed = self.run_it(grasp)

        self.assertEqual(status, 1)
        self.assertIn("x/y 对准失败", printed)
        self.assertIn("稳定看到物块", printed)


def fake_motor_bus(port: str = "COM9") -> Mock:
    return Mock(port=port)


class MainTests(unittest.TestCase):
    """``main`` 的装配顺序：校验 -> 观测源 -> 机械臂，且只装配一次。

    真机上"提前开设备"的代价不是慢，是**校验被挪到设备之后**：``RobotArm.__init__``
    已经打开串口，这时候才发现没标定，机械臂已经在线上了。
    """

    def patch_loaders(self, *, target=None, jacobian=None):
        """换掉五个配置读取器：真文件一个都不碰。

        返回 ``(上下文, 假件字典)``——**假件要在上下文里取出来**，``with`` 结束之后
        模块属性会被还原，那时候再 ``menu.load_arm_config.return_value`` 就是真函数了。
        """
        mocks = dict(
            load_servo_config=Mock(return_value=ServoConfig()),
            load_object_selector=Mock(return_value=ObjectSelector()),
            load_target=Mock(return_value=target),
            load_jacobian=Mock(return_value=jacobian),
            load_lift_scale=Mock(return_value=None),
            load_arm_config=Mock(return_value=object()),
            load_motor_bus_config=Mock(return_value=fake_motor_bus()),
        )
        return patch.multiple(menu, **mocks), mocks

    def patch_vision(self):
        """视觉快照读的是假配置：真文件、真相机都不该被碰到。"""
        settings = object()
        return patch.multiple(
            "modules.vision.runtime",
            _load_settings=Mock(return_value=settings),
            _override_settings=Mock(return_value=settings),
        )

    def build_session(self, *, target=None, jacobian=None, source=None, arm=None):
        """``main`` 走普通动作分支时的全套假件：返回假件字典。

        注意 ``patch.object`` 返回的是 ``_patch`` 对象，**没有** ``return_value``
        属性——所以观测源只能从构造参数 ``new=`` 给进去，不能在拿到 patch 之后再挂。
        """
        arm = Mock() if arm is None else arm
        arm.initialize.return_value = None
        arm.enable_all.return_value = None
        if source is None:
            source = Mock()
        # ``_read_object`` 会读 ``link.error`` 拼错误信息；Mock 自动生成的属性是
        # 真值，会让"没画面"那条分支以为链路报错了。
        source.error = None
        loaders = self.patch_loaders(target=target, jacobian=jacobian)
        open_source = Mock(return_value=source)
        return {
            "arm": arm,
            "source": source,
            "new_arm": patch("modules.grasp.runtime._new_arm", return_value=arm),
            "open_source": patch.object(menu, "_open_source", new=open_source),
            "open_source_mock": open_source,
            "vision": self.patch_vision(),
            "loaders": loaders[0],
            "load_arm_config": loaders[1]["load_arm_config"],
            "load_motor_bus_config": loaders[1]["load_motor_bus_config"],
        }

    def run_main(self, argv) -> "tuple":
        """跑一次 ``main``；返回 ``(退出码, 打出来的字)``。

        菜单是会打印的（现场就靠那些字核对），所以每次调用都要把 stdout 收走，
        否则整套测试的输出里会混进一堆"打开机械臂总线 COM9"。
        """
        printed = io.StringIO()
        with redirect_stdout(printed):
            status = main(argv)
        return status, printed.getvalue()

    def test_the_normal_branch_assembles_the_session_exactly_once(self) -> None:
        mocks = self.build_session(
            target=ServoTarget(x_mm=0.0, y_mm=0.0), jacobian=healthy_jacobian()
        )
        with mocks["loaders"], mocks["vision"], mocks[
            "open_source"
        ] as open_source, mocks["new_arm"] as new_arm, patch.object(
            menu.Grasp, "align_xy", return_value=canned_align_result()
        ) as align:
            status, _printed = self.run_main([])
            arm_config = mocks["load_arm_config"].return_value

        self.assertEqual(status, 0)
        new_arm.assert_called_once_with("COM9", arm_config)
        open_source.assert_called_once()
        align.assert_called_once_with()

    def test_the_source_is_opened_from_inside_the_factory_after_validation(self) -> None:
        """顺序硬要求：``source_factory`` 只能在 ``_new_arm`` 之前、且由它自己调用。

        用"构造机械臂那一刻观测源已经取过几次"来断言，而不是靠先后两次 mock 调用的
        全局顺序——后者在同步代码里看不出"谁先建的设备"。
        """
        mocks = self.build_session(
            target=ServoTarget(x_mm=0.0, y_mm=0.0), jacobian=healthy_jacobian()
        )
        observed: dict = {}

        def build_arm(port, config):
            # 机械臂开工的这一刻，观测源已经被取过几次？必须是 1（由 source_factory
            # 在 `_open_settings` 里调用），而不是 0 或 2。
            observed["source_calls_at_arm_time"] = mocks["open_source_mock"].call_count
            return mocks["arm"]

        with mocks["loaders"], mocks["vision"], mocks[
            "open_source"
        ] as open_source, patch(
            "modules.grasp.runtime._new_arm", side_effect=build_arm
        ) as new_arm, patch.object(
            menu.Grasp, "align_xy", return_value=canned_align_result()
        ):
            status, _printed = self.run_main([])
            new_arm.assert_called_once_with("COM9", mocks["load_arm_config"].return_value)
            self.assertEqual(open_source.call_count, 1)

        self.assertEqual(status, 0)
        self.assertEqual(observed["source_calls_at_arm_time"], 1)

    def test_the_align_and_descend_branch_still_assembles_once(self) -> None:
        mocks = self.build_session(
            target=ServoTarget(x_mm=0.0, y_mm=0.0), jacobian=healthy_jacobian()
        )
        with mocks["loaders"], mocks["vision"], mocks[
            "open_source"
        ] as open_source, mocks["new_arm"] as new_arm, patch.object(
            menu.Grasp, "align_and_descend", return_value=canned_result()
        ) as descend:
            status, _printed = self.run_main(
                ["--align-and-descend", "--z-pulses", "800", "--retract-pulses", "20"]
            )

        self.assertEqual(status, 0)
        new_arm.assert_called_once()
        open_source.assert_called_once()
        descend.assert_called_once_with(z_pulses=800, retract_pulses=20)

    def test_a_missing_calibration_stops_before_any_device(self) -> None:
        """菜单里直接按 21 的人多半还没标定；这时候开硬件毫无意义。"""
        mocks = self.build_session(target=None, jacobian=None)
        with mocks["loaders"], mocks["vision"], mocks[
            "open_source"
        ] as open_source, mocks["new_arm"] as new_arm:
            status, printed = self.run_main([])

        self.assertEqual(status, 1)
        open_source.assert_not_called()
        new_arm.assert_not_called()
        self.assertIn("先做两步标定", printed)

    def test_status_is_not_blocked_by_the_calibration_gate(self) -> None:
        """``--status`` 只读不动：缺 target / jacobian 也必须能跑（它正是用来核对的）。"""
        source = Mock()
        source.frame.side_effect = [
            (sighting(1.0, -1.0),),
            (sighting(1.0, -1.0),),
            KeyboardInterrupt,
        ]
        mocks = self.build_session(target=None, jacobian=None, source=source)
        with mocks["loaders"], mocks["vision"], mocks[
            "open_source"
        ], mocks["new_arm"] as new_arm:
            status, printed = self.run_main(["--status"])

        self.assertEqual(status, 0)
        new_arm.assert_not_called()
        self.assertNotIn("先做两步标定", printed)

    def test_calibrate_is_not_blocked_by_the_calibration_gate(self) -> None:
        """否则就是"缺标定所以不能跑标定"的死循环。"""
        arm = Mock()
        arm.initialize.return_value = None
        arm.enable_all.return_value = None
        mocks = self.build_session(target=None, jacobian=None)
        with mocks["loaders"], mocks["vision"], mocks["open_source"], patch.object(
            menu, "RobotArm", return_value=arm
        ) as robot_arm, patch.object(
            menu, "probe_jacobian", return_value=healthy_jacobian()
        ) as probe, patch.object(
            menu, "save_calibration", return_value="fake.json"
        ), mocks["new_arm"] as new_arm:
            status, printed = self.run_main(["--calibrate"])

        self.assertEqual(status, 0)
        robot_arm.assert_called_once()
        probe.assert_called_once()
        new_arm.assert_not_called()
        self.assertNotIn("先做两步标定", printed)

    def test_manual_modes_do_not_go_through_the_grasp_session(self) -> None:
        """``--align-z`` 有自己的前提（它只要求平面已对准），不走普通动作门禁。

        判据是**走的是哪条路**：低层工具分支自己建 ``RobotArm``、自己收尾；
        普通动作那条会话路径（``modules.grasp.runtime._new_arm``）一次都不该被碰到。
        """
        # 用会如实上报动作时长的假臂，而不是裸 Mock：``run_descend_z`` 真的会发脉冲，
        # 并读动作报告里的 ``estimated_seconds``。
        config = ServoConfig()
        pulses: list = []
        arm = arm_double(pulses)
        loaders = self.patch_loaders(target=None, jacobian=None)
        open_source = Mock(return_value=FakeSource(1.0, -1.0))
        with loaders[0], self.patch_vision(), patch.object(
            menu, "_open_source", new=open_source
        ), patch.object(
            menu, "RobotArm", return_value=arm
        ) as robot_arm, patch("modules.grasp.runtime._new_arm") as new_arm:
            status, printed = self.run_main(["--align-z"])
            source_calls = open_source.call_count
            robot_calls = robot_arm.call_count

        self.assertEqual(status, 0)
        self.assertEqual(source_calls, 1)
        self.assertEqual(robot_calls, 1, "低层工具分支自己建机械臂")
        self.assertEqual(
            pulses,
            [("down", config.z_descend_pulses), ("back", config.extension_retract_pulses)],
        )
        new_arm.assert_not_called()
        self.assertNotIn("先做两步标定", printed)


if __name__ == "__main__":
    unittest.main()
