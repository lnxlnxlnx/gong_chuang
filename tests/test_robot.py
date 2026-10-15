"""``Robot`` 总入口：装配、共享资源、借用抓取与统一收尾（全部用假设备）。

这里钉住的是方案 §6 里"管理类一键启闭 / 一键开启失败 / 共享资源 / 关闭不重复"
那几行验收，以及"未选模块不偷偷开设备"这条。

**不用真设备**：``Robot._open_shared_bus`` 与各模块的装配接缝都被替换成假件，
所以这些用例在没有任何硬件的机器上也能跑。
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from robot import Robot


class FakeModule:
    """一个假模块：只记 ``close`` 的次数，并记进用例的调用序列。

    刻意不用 ``Mock``：这里要断言的是"哪一个模块、在什么顺序上被收尾"，
    显式的小假件比一串 ``Mock`` 属性更好读，也不会被 ``reset_mock`` 的开关绊到。
    """

    def __init__(self, name: str, calls: list, *, fail_on_close=None) -> None:
        self.name = name
        self.closed = False
        self.close_calls = 0
        self._calls = calls
        self._fail = fail_on_close

    def close(self) -> None:
        self.close_calls += 1
        self._calls.append(self.name + ".close")
        if self._fail is not None:
            raise self._fail


def healthy_arm_config():
    """一份地址与底盘不冲突的假 arm 配置。"""
    return SimpleNamespace(
        axes={
            "base": SimpleNamespace(motor_id=5),
            "lift": SimpleNamespace(motor_id=6),
            "extension": SimpleNamespace(motor_id=7),
        }
    )


def healthy_chassis_profile():
    return SimpleNamespace(motor_ids={"FL": 1, "FR": 2, "RL": 3, "RR": 4})


class RobotTestCase(unittest.TestCase):
    """把"读配置 + 开设备"整条链换成假件，只留 Robot 自己的编排逻辑。"""

    def setUp(self) -> None:
        self.calls: list = []
        # 共享总线也用同一套假件：它的 close 要参与"最后关"的顺序断言。
        self.bus = FakeModule("bus", self.calls)
        self.arm = FakeModule("arm", self.calls)
        self.chassis = FakeModule("chassis", self.calls)
        self.gripper = FakeModule("gripper", self.calls)
        self.vision = FakeModule("vision", self.calls)
        self.grasp = FakeModule("grasp", self.calls)

        self.patchers = [
            patch("robot.Robot._open_shared_bus", side_effect=lambda port: self.bus),
            patch("modules.arm.config.load_arm_config", return_value=healthy_arm_config()),
            patch("modules.motor.config.load_bus_config",
                  return_value=SimpleNamespace(port="FAKE_BUS")),
            patch("modules.arm.runtime.RobotArm._from_settings", return_value=self.arm),
            patch("modules.chassis.runtime.MecanumChassis.load_hardware_profile",
                  return_value=healthy_chassis_profile()),
            patch("modules.chassis.runtime.MecanumChassis._from_settings",
                  return_value=self.chassis),
            patch("modules.servo.config.load_gripper_config",
                  return_value=SimpleNamespace(servo_id=1)),
            patch("modules.servo.drive.bus_servo.load_bus_config",
                  return_value=SimpleNamespace(port="SERVO", baudrate=115200)),
            patch("modules.servo.runtime.GripperServo.open", return_value=self.gripper),
            patch("modules.vision.runtime._load_settings", return_value=Mock()),
            patch("modules.vision.runtime.Vision._open_config", return_value=self.vision),
            patch("modules.grasp.runtime.Grasp._from_modules_settings",
                  return_value=self.grasp),
            patch("modules.grasp.drive.servo_model.load_servo_config", return_value=Mock()),
            patch("modules.grasp.drive.servo_model.load_object_selector", return_value=Mock()),
            patch("modules.grasp.drive.servo_model.load_target", return_value=Mock()),
            patch(
                "modules.grasp.drive.servo_model.load_jacobian",
                return_value=Mock(pulses_for=Mock(return_value=(0.0, 0.0))),
            ),
            patch("modules.grasp.drive.servo_model.load_lift_scale", return_value=None),
        ]
        for patcher in self.patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_open_requires_an_explicit_choice(self) -> None:
        """四个设备模块与 grasp 都默认 False：至少明确选一个。"""
        with self.assertRaises(ValueError) as caught:
            Robot.open()
        self.assertIn("至少", str(caught.exception))

    def test_open_all_uses_the_same_assembly_as_the_five_flags(self) -> None:
        """``open_all`` 与五项全选的 ``open`` 取得相同模块集合。"""
        with Robot.open_all() as everything:
            by_all = everything.modules
        with Robot.open(arm=True, gripper=True, vision=True, chassis=True, grasp=True) as five:
            by_flags = five.modules
        self.assertEqual(by_all, by_flags)
        self.assertEqual(by_all, ("arm", "gripper", "vision", "chassis", "grasp"))

    def test_grasp_implies_arm_and_vision_and_shares_them(self) -> None:
        """``grasp=True`` 包含 arm / vision 两项依赖，而且**共用**同一批对象。"""
        with Robot.open(grasp=True) as robot:
            self.assertEqual(robot.modules, ("arm", "vision", "grasp"))

        shared = patch("modules.grasp.runtime.Grasp._from_modules_settings")
        with shared as assemble:
            assemble.return_value = self.grasp
            with Robot.open(grasp=True) as robot:
                kwargs = assemble.call_args.kwargs
                self.assertIs(kwargs["arm"], robot.arm)
                self.assertIs(kwargs["vision"], robot.vision)

    def test_unselected_modules_refuse_access_instead_of_opening_late(self) -> None:
        """未启用的模块**立刻**报错，不在动作时偷偷打开设备。"""
        with Robot.open(arm=True) as robot:
            self.assertEqual(robot.modules, ("arm",))
            for name in ("gripper", "vision", "chassis", "grasp"):
                with self.subTest(module=name):
                    with self.assertRaises(RuntimeError):
                        getattr(robot, name)

    def test_arm_and_chassis_share_one_stepper_connection(self) -> None:
        """arm+chassis 只打开**一条**步进连接，两边都借用它。"""
        with patch("robot.Robot._open_shared_bus") as open_bus, patch(
            "modules.arm.runtime.RobotArm._from_settings"
        ) as make_arm, patch(
            "modules.chassis.runtime.MecanumChassis._from_settings"
        ) as make_chassis:
            open_bus.return_value = self.bus
            make_arm.return_value = self.arm
            make_chassis.return_value = self.chassis
            with Robot.open(arm=True, chassis=True):
                pass

        open_bus.assert_called_once_with("FAKE_BUS")
        self.assertIs(make_arm.call_args.kwargs["bus"], self.bus)
        self.assertIs(make_chassis.call_args.kwargs["bus"], self.bus)
        # 两个借用者都不拥有这条连接。
        self.assertFalse(make_arm.call_args.kwargs["owns_bus"])
        self.assertFalse(make_chassis.call_args.kwargs["owns_bus"])

    def test_close_releases_everything_once_in_the_documented_order(self) -> None:
        """一次 ``close`` 释放全部；借用的 grasp 最先失效，共享总线最后关。"""
        robot = Robot.open_all()
        robot.close()
        robot.close()  # 幂等：不再发命令

        self.assertEqual(
            self.calls,
            ["grasp.close", "vision.close", "gripper.close", "chassis.close",
             "arm.close", "bus.close"],
        )
        for module in (self.arm, self.chassis, self.gripper, self.vision, self.grasp):
            self.assertEqual(module.close_calls, 1, f"{module.name} 应只关一次")
        self.assertEqual(self.bus.close_calls, 1)

    def test_context_exit_closes_everything(self) -> None:
        with Robot.open_all() as robot:
            self.assertFalse(robot.closed)
        self.assertTrue(robot.closed)
        self.assertEqual(self.bus.close_calls, 1)

    def test_every_module_access_fails_after_close(self) -> None:
        robot = Robot.open_all()
        robot.close()
        for name in ("arm", "gripper", "vision", "chassis", "grasp"):
            with self.subTest(module=name):
                with self.assertRaises(RuntimeError):
                    getattr(robot, name)
        with self.assertRaises(RuntimeError):
            with robot:
                pass  # pragma: no cover - 进不来

    def test_a_partial_open_failure_releases_what_was_already_opened(self) -> None:
        """任一模块启动失败：已开启的全部清理，不返回半就绪 Robot。"""
        with patch(
            "modules.chassis.runtime.MecanumChassis._from_settings",
            side_effect=RuntimeError("chassis boom"),
        ):
            with self.assertRaises(RuntimeError) as caught:
                Robot.open(arm=True, chassis=True)

        self.assertIn("chassis boom", str(caught.exception))
        self.assertEqual(self.arm.close_calls, 1)
        self.assertEqual(self.bus.close_calls, 1)

    def test_a_failing_first_step_does_not_skip_the_rest(self) -> None:
        """清理第一项失败不阻断其他项，错误清单含每个失败阶段。"""
        self.arm = FakeModule("arm", self.calls, fail_on_close=RuntimeError("arm close broke"))
        with patch("modules.arm.runtime.RobotArm._from_settings", return_value=self.arm):
            robot = Robot.open_all()

            with self.assertRaises(RuntimeError) as caught:
                robot.close()

        message = str(caught.exception)
        self.assertIn("arm.close", message)
        self.assertIn("arm close broke", message)
        # 其余四项仍然尝试过。
        for module in (self.vision, self.gripper, self.chassis):
            self.assertEqual(module.close_calls, 1, f"{module.name} 仍要被收尾")
        self.assertEqual(self.bus.close_calls, 1)

    def test_business_error_wins_over_cleanup_error(self) -> None:
        """``with`` 里出业务错、收尾也失败：业务异常是主异常，清理错误是 cause。"""
        self.arm = FakeModule("arm", self.calls, fail_on_close=RuntimeError("cleanup broke"))
        with patch("modules.arm.runtime.RobotArm._from_settings", return_value=self.arm):
            with self.assertRaises(ValueError) as caught:
                with Robot.open_all():
                    raise ValueError("business broke")
        self.assertIn("business broke", str(caught.exception))
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)

    def test_conflicting_motor_addresses_are_rejected_before_any_connection(self) -> None:
        """arm 与 chassis 共用一条总线，地址冲突必须在**开连接之前**报出来。"""
        clashing = SimpleNamespace(
            axes={
                "base": SimpleNamespace(motor_id=1),  # 与底盘 FL 相同
                "lift": SimpleNamespace(motor_id=6),
                "extension": SimpleNamespace(motor_id=7),
            }
        )
        with patch("modules.arm.config.load_arm_config", return_value=clashing), patch(
            "robot.Robot._open_shared_bus"
        ) as open_bus:
            with self.assertRaises(ValueError) as caught:
                Robot.open(arm=True, chassis=True)

        self.assertIn("1", str(caught.exception))
        open_bus.assert_not_called()

    def test_config_is_validated_before_the_first_device(self) -> None:
        """所有被选模块的配置先读完再开第一个设备。"""
        with patch(
            "modules.vision.runtime._load_settings",
            side_effect=ValueError("bad vision config"),
        ), patch("robot.Robot._open_shared_bus") as open_bus:
            with self.assertRaises(ValueError):
                Robot.open(arm=True, vision=True)
        open_bus.assert_not_called()

    def test_a_failing_grasp_config_keeps_devices_closed(self) -> None:
        """``grasp=True`` 时缺 target 要在开设备之前失败（与 Grasp 同一套校验）。"""
        with patch(
            "modules.grasp.drive.servo_model.load_target", return_value=None
        ), patch("robot.Robot._open_shared_bus") as open_bus:
            with self.assertRaises(Exception) as caught:
                Robot.open(grasp=True)
        self.assertIn("target", str(caught.exception))
        open_bus.assert_not_called()


if __name__ == "__main__":
    unittest.main()
