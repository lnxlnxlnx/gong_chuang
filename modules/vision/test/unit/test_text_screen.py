"""板载屏文字：字号适配、空输入、以及"显示完不阻塞"（不开真窗口）。"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest
from unittest.mock import Mock, patch

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
        self.assertLessEqual(
            len(lines) * tallest * text_screen._LINE_GAP, height - 2 * margin,
        )


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
        process = Mock()
        process.poll.return_value = None
        process.stdin = io.StringIO()
        process.stdout = io.StringIO('{"ok": true}\n{"ok": true}\n')
        self.addCleanup(text_screen._close_display)
        with patch("cv2.imshow") as imshow, \
                patch("cv2.waitKey") as wait_key, \
                patch("subprocess.Popen", return_value=process) as popen:
            name = show_text("134+123+314+231", size=(320, 240))
            show_text("123+321+213+132", size=(800, 600))

        self.assertEqual(name, TEXT_WINDOW_NAME)
        # 调用方不参与 GUI 事件处理；两次更新发送到同一个显示进程。
        imshow.assert_not_called()
        wait_key.assert_not_called()
        self.assertEqual(popen.call_count, 1)
        commands = [json.loads(line) for line in process.stdin.getvalue().splitlines()]
        self.assertEqual(commands, [
            {"text": "134+123+314+231", "size": [320, 240]},
            {"text": "123+321+213+132", "size": [800, 600]},
        ])

    def test_display_failure_is_reported_to_the_caller(self) -> None:
        process = Mock()
        process.stdin = io.StringIO()
        process.stdout = io.StringIO('{"error": "no display backend"}\n')
        self.addCleanup(text_screen._close_display)
        with patch("subprocess.Popen", return_value=process):
            with self.assertRaisesRegex(RuntimeError, "no display backend"):
                show_text("123")

    def test_invalid_input_does_not_start_a_display_process(self) -> None:
        with patch("subprocess.Popen") as popen:
            with self.assertRaises(ValueError):
                show_text(" ")
            with self.assertRaises(ValueError):
                show_text("123", size=(10, 10))
        popen.assert_not_called()


@unittest.skipUnless(
    sys.platform == "win32" and os.environ.get("ROBOT_TEXT_SCREEN_GUI_TEST") == "1",
    "需要 Windows 桌面；设置 ROBOT_TEXT_SCREEN_GUI_TEST=1 运行真实窗口验收",
)
class WindowsDisplayTests(unittest.TestCase):
    def test_window_responds_without_main_thread_gui_events(self) -> None:
        """主程序停止泵事件六秒后，文字窗仍须响应，且不受预览收尾影响。"""
        script = textwrap.dedent('''\
            import ctypes
            import time
            import cv2
            from robot import Robot
            from modules.vision import Vision
            from modules.vision.config import VisionConfig

            user32 = ctypes.windll.user32
            user32.FindWindowW.restype = ctypes.c_void_p
            user32.IsHungAppWindow.argtypes = [ctypes.c_void_p]
            robot = Robot(vision=Vision(VisionConfig()))
            assert robot.vision.show_text('123+321+213+132') == 'vision-text'
            window = user32.FindWindowW(None, 'vision-text')
            assert window, '文字窗口未创建'
            time.sleep(6)
            assert not user32.IsHungAppWindow(window), '调用返回后文字窗口失去响应'
            cv2.destroyAllWindows()
            assert user32.FindWindowW(None, 'vision-text'), '预览收尾关闭了文字窗'
            robot.vision.show_text('321+123', size=(320, 240))
            assert user32.FindWindowW(None, 'vision-text') == window, '更新新建了窗口'
            robot.close()
            print('responsive, isolated, reused', flush=True)
        ''')
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[4],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("responsive, isolated, reused", result.stdout)


if __name__ == "__main__":
    unittest.main()
