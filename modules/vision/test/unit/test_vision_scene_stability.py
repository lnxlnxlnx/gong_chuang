"""``Vision`` 的画面级停稳接口：``scene_stability`` / ``scene_stable`` / ``wait_scene_stable``。

假相机直接给合成帧；这里守的是"整幅画面"这件事：

- 相同帧必须判静止，有东西在挪必须判还在动；
- 所有相邻帧对都要通过，而不是只看首尾；
- 等待接口只在"看满最小等待之后仍然静止"时放行，头几眼的静止、一直动都不作数；
- 连拍只在开头丢一次缓冲（几帧要在时间上相邻）。
"""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import numpy as np

from modules.vision import Vision
from modules.vision.config import TargetSettings, VisionConfig


def _settings() -> VisionConfig:
    return VisionConfig(camera_ids={"windows": 7, "jetson": 7}, targets=TargetSettings())


def blank(value: int = 0) -> np.ndarray:
    return np.full((48, 64, 3), value, dtype=np.uint8)


def with_square(x: int, *, value: int = 220) -> np.ndarray:
    frame = blank()
    frame[16:32, x:x + 16] = value
    return frame


def _vision(frames) -> "tuple[Vision, Mock]":
    """一个只差设备的会话：相机按 ``frames`` 依次返回合成帧。

    ``_read_burst`` 开头固定丢 ``max(1, drain_frames)`` 帧，``drain_frames=0`` 时也是 1 帧，
    所以每个判断周期消耗 ``1 + frames`` 次 ``read()``。
    """
    vision = Vision(_settings())
    camera = Mock()
    camera.read.side_effect = list(frames)
    vision._camera, vision._locator = camera, Mock()
    vision._drain_frames = 0
    return vision, camera


class SceneStabilityTests(unittest.TestCase):
    def test_a_still_scene_is_stable(self) -> None:
        frame = blank()
        vision, _camera = _vision([frame] * 5)
        result = vision.scene_stability()

        self.assertTrue(result.stable)
        self.assertEqual(result.frames, 4)
        self.assertEqual(result.max_mean_abs_diff, 0.0)
        self.assertIn("静止", result.summary())

    def test_a_moving_block_makes_the_scene_unstable(self) -> None:
        vision, _camera = _vision(
            [blank(), with_square(4), with_square(12), with_square(20), with_square(28)]
        )
        result = vision.scene_stability()

        self.assertFalse(result.stable)
        self.assertGreater(result.max_changed_ratio, 0.0)
        self.assertIn("还在动", result.summary())

    def test_scene_stable_returns_the_boolean_shorthand(self) -> None:
        still = blank()
        vision, _camera = _vision([still] * 5)
        self.assertIs(vision.scene_stable(), True)

        moving, _camera = _vision(
            [blank(), with_square(4), with_square(12), with_square(20), with_square(28)]
        )
        self.assertIs(moving.scene_stable(), False)

    def test_the_drain_happens_once_for_the_whole_burst(self) -> None:
        frame = blank()
        vision = Vision(_settings())
        camera = Mock()
        camera.read.side_effect = [frame] * 2 + [frame] * 4
        vision._camera, vision._locator = camera, Mock()
        vision._drain_frames = 2

        vision.scene_stability(frames=4)

        self.assertEqual(camera.read.call_count, 6, "2 帧丢弃 + 4 帧连拍，不逐帧丢缓冲")

    def test_frames_must_be_at_least_two(self) -> None:
        vision, camera = _vision([blank()] * 5)
        with self.assertRaisesRegex(ValueError, "至少需要两帧"):
            vision.scene_stability(frames=1)
        camera.read.assert_not_called()

    def test_a_closed_session_refuses(self) -> None:
        vision, _camera = _vision([blank()] * 5)
        vision.close()
        with self.assertRaises(RuntimeError):
            vision.scene_stable()

    def test_the_last_judgement_is_kept_for_diagnostics(self) -> None:
        """超时失败时要能报出"判据当时差了多少"，所以每次判断都留最后一份。"""
        vision, _camera = _vision(
            [blank(), blank(), with_square(4), with_square(12), with_square(20)]
        )
        self.assertIsNone(vision.last_scene_stability)

        result = vision.scene_stability()

        self.assertFalse(result.stable)
        self.assertIs(vision.last_scene_stability, result)


class VirtualClock:
    """假时钟：时间只由 ``sleep`` 推进，测试不必真等最小等待那几秒。

    装上它之后 ``time.sleep`` 也换成 :meth:`sleep`，整个等待循环按虚拟时间走——
    看第几眼、睡多久都可预测。
    """

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += max(0.0, seconds)


class WaitSceneStableTests(unittest.TestCase):
    """``wait_scene_stable``：**头一眼"静止"不算数**，看满最小等待才可能放行。

    一次判断只连拍 4 帧（约 130 ms），"这一眼没动"太便宜：物块正要被带走之前画面
    也是静的。所以判据是"看满 ``min_seconds`` 之后，这一眼仍然静止"；中间哪一眼还在动
    就不放行，一直动就等满 ``timeout`` 返回 ``False``。

    涉及最小等待的用例都用 :class:`VirtualClock`：虚拟时间按 ``interval`` 推进，
    不必真的 sleep 3.5 s。
    """

    def _virtual(self, frames):
        vision, camera = _vision(frames)
        clock = VirtualClock()
        vision._clock = clock
        return vision, camera, clock

    def test_it_returns_after_motion_settles(self) -> None:
        """动过又停下——看满最小等待的第二眼放行。"""
        moving = [blank(), with_square(4), with_square(12), with_square(20)]
        still = [blank()] * 4
        vision, camera = _vision([blank()] + moving + [blank()] + still)

        self.assertTrue(
            vision.wait_scene_stable(timeout=1.0, interval=0.0, min_seconds=0.0)
        )
        self.assertEqual(camera.read.call_count, 10, "判两次：第一次还在动，第二次稳定")

    def test_the_opening_still_looks_are_skipped_until_the_minimum_wait(self) -> None:
        """全程静止也要看满 ``min_seconds``：0.0 / 0.3 s 的静止都不算，0.6 s 才放行。"""
        frame = blank()
        vision, camera, clock = self._virtual([frame] * 20)

        with patch("time.sleep", clock.sleep):
            self.assertTrue(
                vision.wait_scene_stable(timeout=8.0, interval=0.3, min_seconds=0.6)
            )

        self.assertEqual(
            camera.read.call_count, 15, "三眼判断：前两眼在最小等待内，不作数"
        )

    def test_a_motion_inside_the_minimum_wait_delays_the_verdict(self) -> None:
        """最小等待之内还在动：那一轮不算，要等它之后的下一次静止。"""
        still = [blank()] * 5
        moving = [blank(), blank(), with_square(4), with_square(12), with_square(20)]
        vision, camera, clock = self._virtual(still + still + moving + still)

        with patch("time.sleep", clock.sleep):
            self.assertTrue(
                vision.wait_scene_stable(timeout=8.0, interval=0.3, min_seconds=0.6)
            )

        self.assertEqual(
            camera.read.call_count, 20, "四眼：静止、静止、还在动、动完停稳"
        )

    def test_it_gives_up_while_the_scene_keeps_moving(self) -> None:
        """一直在动：看满 ``timeout`` 也不放行。"""
        moving = [blank(), blank(), with_square(4), with_square(12), with_square(20)]
        vision, camera, clock = self._virtual([blank()] + moving * 3)

        with patch("time.sleep", clock.sleep):
            self.assertFalse(
                vision.wait_scene_stable(timeout=0.6, interval=0.3, min_seconds=0.6)
            )

        self.assertEqual(
            camera.read.call_count, 15, "0.0 / 0.3 / 0.6 s 各判一眼，都不放行"
        )

    def test_the_minimum_wait_never_exceeds_the_timeout(self) -> None:
        """最小等待比 ``timeout`` 大时按 ``timeout`` 算：等待上限由调用方定。"""
        frame = blank()
        vision, camera = _vision([frame] * 5)

        self.assertTrue(vision.wait_scene_stable(timeout=0.0, min_seconds=3.5))
        self.assertEqual(camera.read.call_count, 5, "预算为 0 时看一眼就下结论")

    def test_the_wait_leaves_the_judgement_it_timed_out_on(self) -> None:
        """超时返回 ``False`` 之后，诊断要能拿到等待里最后一次判断的实测值。"""
        vision, _camera = _vision(
            [blank()] + [blank(), with_square(4), with_square(12), with_square(20)]
        )

        self.assertFalse(vision.wait_scene_stable(timeout=0.0, interval=0.0))
        last = vision.last_scene_stability
        self.assertIsNotNone(last)
        self.assertFalse(last.stable)
        self.assertIn("灰阶", last.summary())

    def test_negative_timeouts_are_rejected(self) -> None:
        vision, _camera = _vision([blank()] * 5)
        with self.assertRaises(ValueError):
            vision.wait_scene_stable(timeout=-1.0)
        with self.assertRaises(ValueError):
            vision.wait_scene_stable(interval=-0.1)
        with self.assertRaises(ValueError):
            vision.wait_scene_stable(min_seconds=-0.1)

    def test_the_boolean_and_detail_interfaces_share_the_same_thresholds(self) -> None:
        """放宽阈值之后同一组运动帧也算静止：两个入口不能各有一套判据。"""
        moving = [blank(), with_square(4), with_square(12), with_square(20), with_square(28)]
        vision, _camera = _vision(moving)
        self.assertFalse(vision.scene_stable())

        relaxed, _camera = _vision(moving)
        self.assertTrue(
            relaxed.scene_stable(max_mean_diff=255.0, max_changed_ratio=1.0)
        )

    def test_a_mild_frame_wide_shake_is_not_reported_as_motion(self) -> None:
        """桌面 / 相机轻抖一两个像素不该再判"还在动"——转盘可能已经停了。

        旧的 0.2% 档对整幅轻微晃动太敏感：边缘像素一抖就超限。这里在 180×320 的画面里
        放一个 60×60 的方块、每帧挪 2 px，变化像素约 0.42%，正好落在旧上限与新上限之间。
        """

        def shaken(step: int) -> np.ndarray:
            frame = np.zeros((180, 320, 3), dtype=np.uint8)
            left = 100 + step
            frame[60:120, left:left + 60] = 220
            return frame

        vision, _camera = _vision(
            [shaken(0), shaken(0), shaken(2), shaken(4), shaken(6)]
        )
        result = vision.scene_stability()

        self.assertGreater(result.max_changed_ratio, 0.002, "这一档旧的 0.2% 上限会判它还在动")
        self.assertLess(result.max_changed_ratio, 0.005)
        self.assertTrue(result.stable, result.summary())


if __name__ == "__main__":
    unittest.main()
