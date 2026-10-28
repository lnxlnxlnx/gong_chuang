"""板载屏文字：字号适配、空输入、以及"显示完不阻塞"（不开真窗口）。"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from modules.vision.drive import text_screen
from modules.vision.drive.text_screen import TEXT_WINDOW_NAME, render_text, show_text


class FitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cv2 = text_screen._cv2()

    def test_font_shrinks_as_the_string_gets_longer(self) -> None:
        short, _ = text_screen._fit(self.cv2, ["134"], 800, 600)
        long, _ = text_screen._fit(self.cv2, ["134+123+314+231+411+234"], 800, 600)
        self.assertLess(long, short)
        self.assertLessEqual(short, text_screen._MAX_SCALE)

    def test_fitted_text_stays_inside_the_canvas(self) -> None:
        # 这次缩放要是算错，字就会被画布边缘切掉——那正是 test_gui2.py 换屏就溢出的原因。
        lines = ["314+321+134+132", "12345+67890"]
        width, height = 320, 240
        scale, thickness = text_screen._fit(self.cv2, lines, width, height)
        widest, tallest = text_screen._measure(self.cv2, lines, scale, thickness)
        margin = text_screen._MARGIN
        self.assertLessEqual(widest, width - 2 * margin)
        self.assertLessEqual(len(lines) * tallest * text_screen._LINE_GAP, height - 2 * margin)


class RenderTests(unittest.TestCase):
    def test_canvas_keeps_the_requested_size_and_the_ink_is_inside(self) -> None:
        canvas = render_text("134+123+314+231", size=(320, 240))
        self.assertEqual(canvas.shape, (240, 320, 3))

        ink = np.argwhere((canvas < 128).any(axis=2))
        self.assertGreater(len(ink), 0, "文字没画到画布上")
        margin = text_screen._MARGIN
        self.assertGreaterEqual(int(ink[:, 1].min()), margin - 1)
        self.assertLessEqual(int(ink[:, 1].max()), 320 - margin)
        self.assertGreaterEqual(int(ink[:, 0].min()), margin - 1)
        self.assertLessEqual(int(ink[:, 0].max()), 240 - margin)

    def test_newlines_make_more_lines(self) -> None:
        one = render_text("134+123", size=(320, 240))
        three = render_text("134\n+123\n+314", size=(320, 240))
        self.assertGreater(
            int((three < 128).any(axis=2).sum()),
            int((one < 128).any(axis=2).sum()),
        )

    def test_blank_text_is_rejected(self) -> None:
        for value in ("", "   ", None):
            with self.assertRaises(ValueError):
                render_text(value, size=(320, 240))

    def test_tiny_canvas_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            render_text("134", size=(10, 10))


class ShowTests(unittest.TestCase):
    def test_returns_without_waiting_for_a_key(self) -> None:
        with patch("cv2.imshow") as imshow, \
                patch("cv2.waitKey") as wait_key, \
                patch("cv2.namedWindow"):
            name = show_text("134+123+314+231", size=(320, 240))

        self.assertEqual(name, TEXT_WINDOW_NAME)
        self.assertEqual(imshow.call_count, 1)
        self.assertEqual(imshow.call_args[0][0], TEXT_WINDOW_NAME)
        # waitKey 只等 1 ms：显示完就返回，窗口留给调用方继续用。
        self.assertEqual(wait_key.call_args[0][0], 1)


if __name__ == "__main__":
    unittest.main()
