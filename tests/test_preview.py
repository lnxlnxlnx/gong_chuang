"""Verify the HighGUI preview helpers that keep the camera window usable."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from modules.vision.drive.camera import looks_blank
from modules.vision.drive.preview import PREVIEW_WINDOW_NAME, open_preview
from modules.vision.drive import preview


class FakeCv2:
    WINDOW_NORMAL = 0

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def namedWindow(self, name: str, flag: int) -> None:
        self.calls.append((name, flag))


class OpenPreviewTests(unittest.TestCase):
    def test_creates_the_window_explicitly_with_the_default_ascii_name(self) -> None:
        fake = FakeCv2()
        with patch.object(preview, "_cv2", return_value=fake):
            name = open_preview()

        self.assertEqual(name, PREVIEW_WINDOW_NAME)
        self.assertEqual(fake.calls, [(PREVIEW_WINDOW_NAME, FakeCv2.WINDOW_NORMAL)])

    def test_rejects_non_ascii_window_names(self) -> None:
        # 中文窗口名在 Windows 上会显示成 ????（见 vision/preview.py 说明）。
        with self.assertRaises(ValueError):
            open_preview("项目视觉演示")

    def test_rejects_an_empty_window_name(self) -> None:
        with self.assertRaises(ValueError):
            open_preview("")


class LooksBlankTests(unittest.TestCase):
    def test_black_frame_is_blank(self) -> None:
        self.assertTrue(looks_blank(np.zeros((48, 64, 3), dtype=np.uint8)))

    def test_white_frame_is_blank(self) -> None:
        self.assertTrue(looks_blank(np.full((48, 64, 3), 255, dtype=np.uint8)))

    def test_frame_with_content_is_not_blank(self) -> None:
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[10:20, 10:20] = 200
        self.assertFalse(looks_blank(frame))

    def test_missing_frame_counts_as_blank(self) -> None:
        self.assertTrue(looks_blank(None))


if __name__ == "__main__":
    unittest.main()
