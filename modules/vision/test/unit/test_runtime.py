"""``Vision`` 会话：装配次数、取帧/定位语义、资源所有权与失败收尾。

这里钉住的是**接口承诺**，不是实现细节：

1. 一次会话只开一台相机、只加载一次模型；连续 ``locate()`` 不重复装配；
2. ``locate()`` 与 ``frame()`` 共用一次取帧/定位实现，各自只读一帧、只推理一次；
3. 空报告与"读不到画面"是**两种不同的结果**（前者正常，后者抛错/返回 ``None``）；
4. 所有取得的资源都会被释放；重复关闭不再执行资源动作；一项清理失败不跳过其余；
5. 构造接缝可以注入假设备，但业务 API 上**不**暴露假设备参数。

全部离线：假相机、假定位器、假配置，不碰真实摄像头与模型。
"""

from __future__ import annotations

import itertools
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from modules.vision import Vision
from modules.vision.config import ModelSettings, TargetSettings, VisionConfig
from modules.vision.reports import RegionReport
from modules.vision.runtime import _override_settings, _sightings_from_report


def _settings(**overrides) -> VisionConfig:
    """一份最小可用快照：camera_ids 里有当前平台的编号就够了。"""
    values = dict(camera_ids={"windows": 7, "jetson": 7}, targets=TargetSettings())
    values.update(overrides)
    return VisionConfig(**values)


def _fake_region(material_id: int, *, radius: float = 40.0):
    """造一个 ``RegionReport``：字段真实到足以验证每一项换算。"""
    region = SimpleNamespace(center_x=100.0 + material_id, center_y=200.0, radius=radius)
    classification = SimpleNamespace(
        label="红色物料", name="red", score=0.75, found=True
    )
    face = SimpleNamespace(
        interior_std=12.5,
        edge_support=0.83,
        sampled_rays=36,
        summary=lambda: "内部σ=12.5，边界支撑=83%（36 个方向）",
    )
    return RegionReport(
        region=region,
        classification=classification,
        face=face,
        accepted=True,
        x_mm=1.0 * material_id,
        y_mm=2.0 * material_id,
        distance_mm=110.0 + material_id,
    )


class VisionSessionTests(unittest.TestCase):
    def test_one_camera_one_locator_for_two_calls(self) -> None:
        """B7 的直接验收：两次 ``locate()`` 只装配一次设备与模型。"""
        settings = SimpleNamespace(camera_id=Mock(return_value=7))
        camera = Mock()
        camera.read.return_value = object()
        report = SimpleNamespace(best=None)
        locator = Mock()
        locator.locate.return_value = report
        with patch("modules.vision.runtime._load_settings", return_value=settings) as load:
            with patch("modules.vision.runtime._build_locator", return_value=locator) as build:
                with patch("modules.vision.runtime._new_camera", return_value=camera) as new:
                    with Vision.open(config_path="custom.json") as vision:
                        self.assertIs(vision.locate(), report)
                        self.assertIs(vision.locate(), report)
                    vision.close()
        load.assert_called_once_with("custom.json", None)
        build.assert_called_once_with(settings, focal_px=None)
        new.assert_called_once_with(7)
        camera.initialize.assert_called_once_with()
        camera.release.assert_called_once_with()
        locator.close.assert_called_once_with()
        self.assertEqual(locator.locate.call_count, 2)
        # 关闭之后任何操作都失败，而且**不自动重开**。
        with self.assertRaises(RuntimeError):
            vision.locate()

    def test_closed_session_cannot_reenter_context(self) -> None:
        vision = Vision(_settings())
        vision.close()
        with self.assertRaises(RuntimeError):
            with vision:
                pass  # pragma: no cover - 进不来

    def test_config_paths_are_handed_to_the_one_settings_loader(self) -> None:
        settings = _settings()
        with patch("modules.vision.runtime._load_settings", return_value=settings) as load:
            with patch("modules.vision.runtime._build_locator"):
                with patch("modules.vision.runtime._new_camera") as new:
                    new.return_value.read.return_value = None
                    Vision.open(config_path="a.json", calibration_path="b.json").close()
        load.assert_called_once_with("a.json", "b.json")

    def test_missing_camera_id_fails_before_allocating_anything(self) -> None:
        """编号缺失要在**分配模型之前**失败：不占模型、不开相机。"""
        settings = VisionConfig(camera_ids={}, targets=TargetSettings())
        with patch("modules.vision.runtime._build_locator") as build:
            with patch("modules.vision.runtime._new_camera") as new:
                with self.assertRaises(ValueError):
                    Vision._open_config(settings)
        build.assert_not_called()
        new.assert_not_called()

    def test_model_failure_leaves_zero_cameras(self) -> None:
        """模型加载失败 → 一台相机都不开；已构造的东西由构造器自己收尾。"""
        settings = _settings()
        with patch("modules.vision.runtime._build_locator", side_effect=FileNotFoundError("no model")):
            with patch("modules.vision.runtime._new_camera") as new:
                with self.assertRaises(FileNotFoundError):
                    Vision._open_config(settings)
        new.assert_not_called()

    def test_camera_failure_releases_the_locator(self) -> None:
        """相机开不起来 → 定位器（模型）必须被释放，原失败原因保留。"""
        settings = _settings()
        locator = Mock()
        camera = Mock()
        camera.initialize.side_effect = RuntimeError("camera busy")
        with patch("modules.vision.runtime._build_locator", return_value=locator):
            with patch("modules.vision.runtime._new_camera", return_value=camera):
                with self.assertRaisesRegex(RuntimeError, "camera busy"):
                    Vision._open_config(settings)
        locator.close.assert_called_once_with()

    def test_partial_initialisation_failure_keeps_the_primary_error(self) -> None:
        """清理也失败时，构造失败仍是主异常，清理错误挂在 ``__cause__`` 上。"""
        settings = _settings()
        locator = Mock()
        locator.close.side_effect = RuntimeError("cleanup broke")
        camera = Mock()
        camera.initialize.side_effect = RuntimeError("camera busy")
        with patch("modules.vision.runtime._build_locator", return_value=locator):
            with patch("modules.vision.runtime._new_camera", return_value=camera):
                with self.assertRaises(RuntimeError) as caught:
                    Vision._open_config(settings)
        self.assertIn("camera busy", str(caught.exception))
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)
        self.assertIn("cleanup broke", str(caught.exception.__cause__))

    def test_failed_cleanup_still_releases_the_other_resource(self) -> None:
        """相机释放失败不跳过定位器释放；错误文本要含每个失败阶段。"""
        vision = Vision(_settings())
        camera = Mock()
        camera.release.side_effect = RuntimeError("release broke")
        locator = Mock()
        locator.close.side_effect = RuntimeError("close broke")
        vision._camera, vision._locator = camera, locator
        with self.assertRaises(RuntimeError) as caught:
            vision.close()
        message = str(caught.exception)
        self.assertIn("camera.release", message)
        self.assertIn("locator.close", message)
        self.assertIn("release broke", message)
        self.assertIn("close broke", message)
        # 两个都尝试过了。
        camera.release.assert_called_once_with()
        locator.close.assert_called_once_with()

    def test_repeated_close_makes_no_extra_calls(self) -> None:
        vision = Vision(_settings())
        camera, locator = Mock(), Mock()
        vision._camera, vision._locator = camera, locator
        vision.close()
        vision.close()
        vision.close()
        camera.release.assert_called_once_with()
        locator.close.assert_called_once_with()

    def test_business_error_wins_over_cleanup_error_in_context_exit(self) -> None:
        """``with`` 里出业务错、收尾也失败时：业务异常是主异常，清理错误是 cause。"""
        vision = Vision(_settings())
        camera = Mock()
        camera.release.side_effect = RuntimeError("release broke")
        vision._camera = camera
        with self.assertRaises(ValueError) as caught:
            with vision:
                raise ValueError("business broke")
        self.assertIn("business broke", str(caught.exception))
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)


class VisionReadTests(unittest.TestCase):
    """取帧与定位：丢帧、重试、时间戳、两种失败语义。"""

    def _session(self, *, frames=None, clock=None, drain=2):
        """``frames`` 是 ``camera.read()`` 依次的返回值；给 ``None`` 表示**永远读到帧**。

        时钟默认**冻结**：真实 monotonic 会让"按 0.05 秒重试"这类断言随时序抖动；
        冻结之后 deadline 判据不会自己到期，超时行为只由测试给的帧序列决定。
        """
        vision = Vision(_settings())
        camera = Mock()
        if frames is None:
            camera.read.return_value = object()
        else:
            camera.read.side_effect = list(frames)
        locator = Mock()
        locator.locate.return_value = SimpleNamespace(best=None)
        vision._camera, vision._locator = camera, locator
        vision._drain_frames = drain
        vision._clock = clock if clock is not None else (lambda: 1000.0)
        return vision, camera, locator

    def test_read_failure_raises_for_locate_and_returns_none_for_frame(self) -> None:
        """读不到画面：``locate`` 抛错，``frame`` 返回 ``None``——两种语义并存。"""
        vision, camera, locator = self._session(frames=[None] * 8)
        with self.assertRaisesRegex(RuntimeError, "读不到摄像头画面"):
            vision.locate()
        self.assertGreaterEqual(camera.read.call_count, 2)
        locator.locate.assert_not_called()
        self.assertGreaterEqual(vision.misses, 1)

    def test_empty_report_is_not_a_failure(self) -> None:
        """画面里没物料是**正常空报告**，不是失败。"""
        vision, _camera, locator = self._session()
        report = vision.locate()
        self.assertIsNone(report.best)
        self.assertEqual(vision.frames, 1)
        locator.locate.assert_called_once()

    def test_each_call_reads_frames_and_infers_exactly_once(self) -> None:
        """一次调用读一帧、推理一次；不重用上次结果。"""
        vision, camera, locator = self._session()
        vision.locate()
        vision.locate()
        # 每次调用先丢 drain 帧（2 帧），再读 1 帧。
        self.assertEqual(camera.read.call_count, 6)
        self.assertEqual(locator.locate.call_count, 2)
        self.assertEqual(vision.frames, 2)

    def test_frame_returns_none_but_locate_raises_on_the_same_source(self) -> None:
        vision, _camera, _locator = self._session(frames=[None] * 8)
        self.assertIsNone(vision.frame(timeout=0.0))

    def test_timeout_retries_before_giving_up(self) -> None:
        """``timeout`` 给了就按 0.05 秒间隔重试，而不是一次读不到就放弃。

        时钟冻结，所以"还没到期"只取决于测试：drain 2 帧 → 第一次读失败（进入重试）
        → 第二次读成功。
        """
        good = object()
        vision, camera, locator = self._session(frames=[None, None, None, good])
        with patch("modules.vision.runtime.time.sleep") as sleep:
            report, _stamp = vision._read_report(timeout=1.0)
        self.assertIsNotNone(report)
        sleep.assert_called_once_with(0.05)
        locator.locate.assert_called_once_with(good)

    def test_timeout_gives_up_once_the_deadline_passes(self) -> None:
        """时钟走过 deadline 之后立刻放弃，不再空转重试。"""
        # 冻结时钟下 deadline 永不到期，所以用"每次调用返回不同值"的钟：
        # 算 deadline 时读到 100，判到期时读到 200。
        ticks = iter([100.0] + [200.0] * 8)
        vision, _camera, locator = self._session(
            frames=[None] * 8, clock=lambda: next(ticks)
        )
        with patch("modules.vision.runtime.time.sleep") as sleep:
            result = vision._read_report(timeout=1.0)
        self.assertIsNone(result)
        sleep.assert_not_called()  # 已经过期，不该再等
        locator.locate.assert_not_called()

    def test_timestamp_is_taken_after_the_read_and_before_the_inference(self) -> None:
        """时间戳必须是"读到帧之后、推理之前"：不能用推理结束时间冒充采集时间。"""
        order = []
        good = object()

        def clock():
            order.append("clock")
            return 123.0

        vision, _camera, locator = self._session(frames=[None, None, good], clock=clock)
        locator.locate.side_effect = (
            lambda _frame: order.append("locate") or SimpleNamespace(best=None)
        )
        _report, stamp = vision._read_report()
        self.assertEqual(stamp, 123.0)
        # 顺序：先取帧（丢帧阶段不读时钟）→ 读时钟 → 推理。
        self.assertEqual(order, ["clock", "locate"])

    def test_drain_reads_a_minimum_of_one_frame(self) -> None:
        """``max(1, drain_frames)``：drain 为 0 时也至少丢一帧。"""
        vision, camera, _locator = self._session(drain=0)
        vision.locate()
        self.assertEqual(camera.read.call_count, 2)  # 1 次丢弃 + 1 次真读
        self.assertEqual(vision.frames, 1)


class VisionConversionTests(unittest.TestCase):
    """``MaterialReport`` → 观测元组：编号、单位与时间戳。"""

    def test_numbering_follows_accepted_faces_and_starts_at_one(self) -> None:
        """编号只在**已接受**的顶面上连续编号，不能先枚举 regions 再过滤。

        ``faces`` 是报告自己过滤好的视图；这里断言编号就是 1..N，
        与 ``MaterialReport.summary()`` 打印的「物料 N:」对得上。
        """
        report = SimpleNamespace(faces=(_fake_region(1), _fake_region(2), _fake_region(3)))
        sightings = _sightings_from_report(report, received_at=456.0)
        self.assertEqual([item.material_id for item in sightings], [1, 2, 3])

    def test_units_are_converted_exactly_once(self) -> None:
        """半径 → 直径（×2）、边缘支撑比例 → 百分比（×100）、深度 → z。"""
        region = _fake_region(3, radius=41.5)
        report = SimpleNamespace(faces=(region,))
        (item,) = _sightings_from_report(report, received_at=99.0)
        self.assertEqual(item.diameter_px, 83.0)
        self.assertEqual(item.edge_support, 83.0)
        self.assertAlmostEqual(item.z_mm, region.distance_mm)
        self.assertEqual(item.inner_sigma, 12.5)
        self.assertEqual(item.edge_directions, 36)
        self.assertEqual(item.confidence, 0.75)
        self.assertEqual(item.color, "red")
        self.assertEqual(item.received_at, 99.0)
        self.assertEqual(item.raw, region.summary())

    def test_empty_report_becomes_an_empty_tuple(self) -> None:
        """空报告 → 空元组（"读到帧了，只是里面没有物料"）。"""
        self.assertEqual(_sightings_from_report(SimpleNamespace(faces=()), received_at=1.0), ())


class VisionOverrideTests(unittest.TestCase):
    """工具覆盖：只生成新快照，不写文件、不改传入的快照。"""

    def test_overrides_do_not_mutate_the_input_snapshot(self) -> None:
        original = _settings()
        changed = _override_settings(
            original,
            camera_id=3,
            conf_thresh=0.5,
            depth_range_mm=(60.0, 80.0),
            material_bgr=(1.0, 2.0, 3.0),
            colour_tolerance=42.0,
            top_diameter_mm=25.0,
        )
        self.assertIsNot(changed, original)
        self.assertEqual(original.camera_ids, {"windows": 7, "jetson": 7})
        self.assertEqual(original.targets.color_tolerance, 60.0)
        self.assertEqual(changed.camera_id("windows"), 3)
        self.assertEqual(changed.model.conf_thresh, 0.5)
        self.assertEqual(changed.targets.working_distance_mm, (60.0, 80.0))
        self.assertEqual(changed.targets.material_bgr, (1.0, 2.0, 3.0))
        self.assertEqual(changed.targets.color_tolerance, 42.0)
        self.assertEqual(changed.targets.material_diameter_mm, 25.0)

    def test_none_means_do_not_override(self) -> None:
        original = _settings()
        self.assertIs(_override_settings(original), original)

    def test_model_path_selects_the_backend_by_suffix(self) -> None:
        onnx = _override_settings(_settings(), model_path="x/best.onnx")
        self.assertEqual(onnx.model.backend, "onnx")
        self.assertEqual(str(onnx.model.onnx_path).replace("\\", "/"), "x/best.onnx")
        engine = _override_settings(_settings(), model_path="x/best.engine")
        self.assertEqual(engine.model.backend, "tensorrt")
        with self.assertRaises(ValueError):
            _override_settings(_settings(), model_path="x/best.pt")

    def test_invalid_values_are_rejected_by_the_existing_rules(self) -> None:
        """复用 config 里已有的校验：非法值当场拒绝，而不是安静地当成没覆盖。

        ``0`` 尤其要走到校验里：用 ``or`` 判"没给"会把 ``colour_tolerance=0`` 变成
        "没覆盖"，用户会以为覆盖生效了。
        """
        with self.assertRaises(ValueError):
            _override_settings(_settings(), camera_id=-1)
        with self.assertRaises(ValueError):
            _override_settings(_settings(), conf_thresh=1.5)
        with self.assertRaises(ValueError):
            _override_settings(_settings(), colour_tolerance=0.0)
        with self.assertRaises(ValueError):
            _override_settings(_settings(), top_diameter_mm=0.0)

    def test_model_snapshot_keeps_the_configured_materials(self) -> None:
        """覆盖阈值不该顺手换掉"哪些类别算物料"。"""
        original = _settings(
            model=ModelSettings(class_names=("red", "blue"), materials=("red",))
        )
        changed = _override_settings(original, conf_thresh=0.9)
        self.assertEqual(changed.model.materials, ("red",))
        self.assertEqual(changed.model.class_names, ("red", "blue"))


if __name__ == "__main__":
    unittest.main()
