"""``Vision`` 的画面级停稳接口：``scene_stability`` / ``scene_stable`` / ``wait_scene_stable``。

假相机直接给合成帧；这里守的是"整幅画面"这件事：

- 相同帧必须判静止，有东西在挪必须判还在动；
- 所有相邻帧对都要通过，而不是只看首尾；
- 等待接口会重试到静止或超时，不会无限转；
- 连拍只在开头丢一次缓冲（几帧要在时间上相邻）。
"""

from __future__ import annotations

import unittest
from unittest.mock import Mock

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


class WaitSceneStableTests(unittest.TestCase):
    """``wait_scene_stable`` 只接受**下一次**稳定，不接受进来时的那一眼。"""

    def test_it_retries_until_the_scene_stops(self) -> None:
        moving = [blank(), with_square(4), with_square(12), with_square(20)]
        still = [blank()] * 4
        vision, camera = _vision([blank()] + moving + [blank()] + still)

        self.assertTrue(vision.wait_scene_stable(timeout=1.0, interval=0.0))
        self.assertEqual(camera.read.call_count, 10, "两轮判断，每轮 1 帧丢弃 + 4 帧连拍")

    def test_a_currently_stable_scene_is_not_accepted_on_the_first_look(self) -> None:
        """进来时已经稳定也要再看一次：当前稳定可能只是"马上又要动"的一瞬间。"""
        frame = blank()
        vision, camera = _vision([frame] * 10)

        self.assertTrue(vision.wait_scene_stable(timeout=1.0, interval=0.0))
        self.assertEqual(camera.read.call_count, 10, "第一眼不算数，必须看到下一次稳定")

    def test_it_waits_for_a_new_stable_window_after_motion(self) -> None:
        """稳定 -> 又在动 -> 再稳定：返回的是**最后一次**稳定，不是进来时那一眼。"""
        still = [blank()] * 5
        moving = [blank(), blank(), with_square(4), with_square(12), with_square(20)]
        vision, camera = _vision(still + moving + still)

        self.assertTrue(vision.wait_scene_stable(timeout=1.0, interval=0.0))
        self.assertEqual(camera.read.call_count, 15, "三轮判断：当前稳定、还在动、下一次稳定")

    def test_it_gives_up_at_the_deadline(self) -> None:
        vision, camera = _vision(
            [blank()] + [blank(), with_square(4), with_square(12), with_square(20)]
        )
        self.assertFalse(vision.wait_scene_stable(timeout=0.0, interval=0.0))
        self.assertEqual(camera.read.call_count, 5, "超时只做第一眼判断，不做第二次")

    def test_a_stable_first_look_does_not_beat_the_deadline(self) -> None:
        """第一眼稳定也不能直接放行：没有时间做第二次确认时返回 ``False``。"""
        frame = blank()
        vision, camera = _vision([frame] * 5)
        self.assertFalse(vision.wait_scene_stable(timeout=0.0, interval=0.0))
        self.assertEqual(camera.read.call_count, 5)

    def test_negative_timeouts_are_rejected(self) -> None:
        vision, _camera = _vision([blank()] * 5)
        with self.assertRaises(ValueError):
            vision.wait_scene_stable(timeout=-1.0)
        with self.assertRaises(ValueError):
            vision.wait_scene_stable(interval=-0.1)

    def test_the_boolean_and_detail_interfaces_share_the_same_thresholds(self) -> None:
        """放宽阈值之后同一组运动帧也算静止：两个入口不能各有一套判据。"""
        moving = [blank(), with_square(4), with_square(12), with_square(20), with_square(28)]
        vision, _camera = _vision(moving)
        self.assertFalse(vision.scene_stable())

        relaxed, _camera = _vision(moving)
        self.assertTrue(
            relaxed.scene_stable(max_mean_diff=255.0, max_changed_ratio=1.0)
        )


if __name__ == "__main__":
    unittest.main()
