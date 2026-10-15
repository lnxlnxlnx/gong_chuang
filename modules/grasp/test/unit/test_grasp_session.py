"""``Grasp`` 会话：资源所有权、校验先于设备、收尾与 ``with`` 的异常语义。

这个文件只测**会话生命周期**，不测伺服算法（那些在 ``test_eye_in_hand``）：

1. 普通动作需要 ``target`` 与 ``jacobian``，而 ``RobotArm.__init__`` 已经打开串口，
   所以"缺标定"必须在**构造设备之前**报出来——断了任何设备都不该被碰到；
2. 关闭时四项收尾（``stop_all`` / ``disable_all`` / ``arm.close`` / ``source.close``）
   **每项独立尝试**：``stop_all`` 失败不该让后面三项一个都不做；
3. 重复关闭不再发任何动作；关闭之后每个动作方法立即失败，不自动重连。
"""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from modules.grasp import Grasp
from modules.grasp.drive.servo_model import (
    ObjectSelector,
    ServoConfig,
    ServoError,
    ServoJacobian,
    ServoTarget,
)


def healthy_jacobian() -> ServoJacobian:
    """一个行列式足够大的雅可比：校验不该在这里拦住。"""
    return ServoJacobian(base_dx=0.012, base_dy=0.0035, radial_dx=-0.0021, radial_dy=0.0091)


def singular_jacobian() -> ServoJacobian:
    """两列成比例 -> ``is_singular`` 为真（和 ``test_eye_in_hand`` 用的是同一组数）。"""
    return ServoJacobian(base_dx=0.01, base_dy=0.02, radial_dx=0.02, radial_dy=0.04)


def fake_arm() -> Mock:
    """假臂：四个收尾动作都有，``initialize`` / ``enable_all`` 返回 None。"""
    arm = Mock()
    arm.initialize.return_value = None
    arm.enable_all.return_value = None
    return arm


class GraspSessionTests(unittest.TestCase):
    def settings(self, **overrides) -> dict:
        values = dict(
            config=ServoConfig(),
            selector=ObjectSelector(),
            target=ServoTarget(x_mm=0.0, y_mm=0.0),
            jacobian=healthy_jacobian(),
            lift_scale=None,
            arm_config=object(),
            port="FAKE",
        )
        values.update(overrides)
        return values

    def open_session(self, **overrides):
        """按 ``_open_settings`` 开一个会话，返回 ``(会话, 假臂, 假源, 假构造器)``。"""
        arm = fake_arm()
        source = Mock()
        with patch("modules.grasp.runtime._new_arm", return_value=arm) as new_arm:
            grasp = Grasp._open_settings(
                source_factory=lambda: source, **self.settings(**overrides)
            )
        return grasp, arm, source, new_arm

    # ------------------------------------------------------------ 校验早于设备
    def test_missing_target_opens_no_devices(self) -> None:
        source_factory = Mock()
        with patch("modules.grasp.runtime._new_arm") as new_arm:
            with self.assertRaises(ServoError):
                Grasp._open_settings(
                    source_factory=source_factory, **self.settings(target=None)
                )
        source_factory.assert_not_called()
        new_arm.assert_not_called()

    def test_missing_jacobian_opens_no_devices(self) -> None:
        source_factory = Mock()
        with patch("modules.grasp.runtime._new_arm") as new_arm:
            with self.assertRaises(ServoError):
                Grasp._open_settings(
                    source_factory=source_factory, **self.settings(jacobian=None)
                )
        source_factory.assert_not_called()
        new_arm.assert_not_called()

    def test_singular_jacobian_opens_no_devices(self) -> None:
        """奇异判据复用 ``pulses_for`` 的自查，所以这里也必须在设备之前拦住。"""
        jacobian = singular_jacobian()
        self.assertTrue(jacobian.is_singular, "这组数必须是奇异的，否则这条测试没测到东西")
        source_factory = Mock()
        with patch("modules.grasp.runtime._new_arm") as new_arm:
            with self.assertRaisesRegex(ServoError, "奇异"):
                Grasp._open_settings(
                    source_factory=source_factory, **self.settings(jacobian=jacobian)
                )
        source_factory.assert_not_called()
        new_arm.assert_not_called()

    def test_a_failing_source_factory_never_reaches_the_arm(self) -> None:
        """``source_factory`` 的失败由它自己收尾；所有权还没转移，就不该去开机械臂。"""
        def boom():
            raise ServoError("起不了观测源")

        with patch("modules.grasp.runtime._new_arm") as new_arm:
            with self.assertRaisesRegex(ServoError, "起不了观测源"):
                Grasp._open_settings(source_factory=boom, **self.settings())
        new_arm.assert_not_called()

    def test_a_session_uses_the_validated_order(self) -> None:
        """校验 -> 取观测源 -> 建机械臂 -> initialize -> enable_all，一步都不少。"""
        order: list = []
        arm_config = object()
        arm = Mock()
        arm.initialize.side_effect = lambda: order.append("initialize")
        arm.enable_all.side_effect = lambda: order.append("enable_all")

        def build_arm(port, config):
            order.append("new_arm")
            self.assertEqual((port, config), ("FAKE", arm_config))
            return arm

        with patch("modules.grasp.runtime._new_arm", side_effect=build_arm) as new_arm:
            grasp = Grasp._open_settings(
                source_factory=lambda: order.append("source") or Mock(),
                **self.settings(arm_config=arm_config),
            )

        self.assertEqual(order, ["source", "new_arm", "initialize", "enable_all"])
        self.assertIs(grasp._arm, arm)
        new_arm.assert_called_once()

    # ------------------------------------------------------------ 失败收尾
    def test_a_failing_initialize_still_closes_the_source_and_the_arm(self) -> None:
        arm = fake_arm()
        arm.initialize.side_effect = RuntimeError("串口没应答")
        source = Mock()
        with patch("modules.grasp.runtime._new_arm", return_value=arm):
            with self.assertRaisesRegex(RuntimeError, "串口没应答"):
                Grasp._open_settings(source_factory=lambda: source, **self.settings())

        source.close.assert_called_once_with()
        arm.close.assert_called_once_with()
        # 还没 enable 就失败了，但仍然逐项尝试：stop/disable 一次都不落。
        arm.stop_all.assert_called_once_with()
        arm.disable_all.assert_called_once_with()

    def test_cleanup_failure_becomes_the_cause_of_the_primary_error(self) -> None:
        """原业务错误必须是主异常（保留 traceback），收尾错误挂成 ``__cause__``。"""
        arm = fake_arm()
        arm.initialize.side_effect = ServoError("初始化失败")
        source = Mock()
        source.close.side_effect = RuntimeError("关不掉观测源")
        with patch("modules.grasp.runtime._new_arm", return_value=arm):
            with self.assertRaises(ServoError) as caught:
                Grasp._open_settings(source_factory=lambda: source, **self.settings())

        self.assertEqual(str(caught.exception), "初始化失败")
        cause = caught.exception.__cause__
        self.assertIsInstance(cause, RuntimeError)
        self.assertIn("Grasp 清理失败", str(cause))
        self.assertIn("关不掉观测源", str(cause))

    # ------------------------------------------------------------ close
    def test_failed_stop_still_disables_closes_arm_and_source(self) -> None:
        grasp, arm, source, _new_arm = self.open_session()
        arm.stop_all.side_effect = RuntimeError("stop failed")

        with self.assertRaises(RuntimeError) as caught:
            grasp.close()
        self.assertIn("arm.stop_all", str(caught.exception))

        # 第二次关：安静返回，不再发任何动作。
        grasp.close()

        arm.stop_all.assert_called_once_with()
        arm.disable_all.assert_called_once_with()
        arm.close.assert_called_once_with()
        source.close.assert_called_once_with()

    def test_repeated_close_makes_no_extra_resource_calls(self) -> None:
        grasp, arm, source, _new_arm = self.open_session()
        grasp.close()
        grasp.close()
        grasp.close()
        for name in ("stop_all", "disable_all", "close"):
            with self.subTest(action=name):
                self.assertEqual(getattr(arm, name).call_count, 1, name)
        self.assertEqual(source.close.call_count, 1)

    def test_close_reports_every_failing_stage(self) -> None:
        """清单要列全：只报第一项会让人修完一个再发现下一个。"""
        grasp, arm, source, _new_arm = self.open_session()
        arm.stop_all.side_effect = RuntimeError("stop 坏了")
        arm.disable_all.side_effect = RuntimeError("disable 坏了")
        source.close.side_effect = RuntimeError("source 坏了")

        with self.assertRaises(RuntimeError) as caught:
            grasp.close()

        message = str(caught.exception)
        for stage in ("arm.stop_all", "arm.disable_all", "source.close"):
            self.assertIn(stage, message)
        self.assertNotIn("arm.close", message, "这一项没失败，不该进清单")
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)
        self.assertEqual(str(caught.exception.__cause__), "stop 坏了")

    # ------------------------------------------------------------ 关闭之后
    def test_actions_after_close_raise_from_every_method(self) -> None:
        grasp, _arm, _source, _new_arm = self.open_session()
        grasp.close()

        for name in ("align_xy", "align_and_descend"):
            with self.subTest(method=name):
                with self.assertRaisesRegex(RuntimeError, "已关闭"):
                    getattr(grasp, name)()

    def test_entering_a_closed_session_raises(self) -> None:
        grasp, _arm, _source, _new_arm = self.open_session()
        grasp.close()
        with self.assertRaisesRegex(RuntimeError, "已关闭"):
            with grasp:
                pass

    def test_enter_returns_the_session_itself(self) -> None:
        grasp, _arm, _source, _new_arm = self.open_session()
        with grasp as entered:
            self.assertIs(entered, grasp)

    # ------------------------------------------------------------ __exit__
    def test_exit_without_a_business_error_propagates_the_cleanup_error(self) -> None:
        grasp, arm, _source, _new_arm = self.open_session()
        arm.disable_all.side_effect = RuntimeError("disable 坏了")

        with self.assertRaises(RuntimeError) as caught:
            with grasp:
                pass

        self.assertIn("Grasp 清理失败", str(caught.exception))
        self.assertIn("disable 坏了", str(caught.exception))

    def test_exit_with_a_business_error_keeps_it_primary_and_attaches_the_cleanup(self) -> None:
        grasp, arm, _source, _new_arm = self.open_session()
        arm.stop_all.side_effect = RuntimeError("stop 坏了")
        business = ServoError("平面还没对准就下降")

        caught = None
        try:
            with grasp:
                raise business
        except ServoError as error:
            caught = error

        self.assertIsNotNone(caught)
        assert caught is not None
        self.assertEqual(str(caught), "平面还没对准就下降")
        self.assertIsInstance(caught.__cause__, RuntimeError)
        self.assertIn("Grasp 清理失败", str(caught.__cause__))
        # 业务异常的 traceback 必须保留（现场要能看到是哪一行抛的）。
        self.assertTrue(caught.__traceback__ is not None)


if __name__ == "__main__":
    unittest.main()
