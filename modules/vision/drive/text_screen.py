"""在板载屏幕上显示一段文字：画出来就留着，不阻塞、不关闭。

## 为什么还是 HighGUI

板子上本来就跑着 OpenCV（相机预览窗口就是它，见 :mod:`modules.vision.drive.preview`），
所以复用同一套后端，不为"显示一串数字"再引 Tk / Qt / 直写帧缓冲。

## 字号是算出来的

``tests/test_gui2.py`` 里字号 4.2 / 笔画 16 是照着 800×600 和一串特定长度的数字
调出来的：换一台屏、或者数字再长两个，字就溢出画布。这里保留那组**比例**
（:data:`_MAX_SCALE` 与 :data:`_STROKE_RATIO` 就是那次实拍的 4.2 与 16），
但按画布把字号缩到刚好装得下——"能不能显示出来"不该依赖调用方猜对字号。

用法::

    from robot import Robot

    with Robot.open(vision=True) as robot:
        robot.vision.show_text("134+123+314+231")   # 返回后窗口仍然留着

不需要相机也能用（它只画一张画布）::

    from modules.vision.drive.text_screen import show_text

    show_text("314+321+134+132")

命令行自测（点击关闭按钮结束）::

    python -m modules.vision.drive.text_screen "134+123+314+231"
"""

from __future__ import annotations

import atexit
import json
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
from threading import Lock, Thread
from typing import List, Optional, Sequence, Tuple

import numpy as np

from modules.vision.drive.preview import open_preview

#: 文字窗口的固定名字。重复调用覆盖同一个窗口，所以循环里每帧调也不会堆出一排窗口。
TEXT_WINDOW_NAME = "vision-text"

#: 默认画布尺寸 ``(宽, 高)``；板载屏不是这个尺寸时传 ``size=``。
DEFAULT_SIZE: Tuple[int, int] = (800, 600)

#: 背景与文字颜色（BGR）：白底黑字，和 ``tests/test_gui2.py`` 一致，远看清。
BACKGROUND = (255, 255, 255)
COLOUR = (0, 0, 0)

#: 画布四边的留白（像素）。
_MARGIN = 10

#: 字号上限，对应实拍的 4.2；再大在 800×600 上就只剩两三个字了。
_MAX_SCALE = 4.2

#: 笔画粗细与字号的比例（实拍 16 / 4.2 ≈ 3.8），让"字变小"时笔画也跟着细。
_STROKE_RATIO = 3.8

#: 行距是字高的几倍（实拍 120 / 92 ≈ 1.3）。
_LINE_GAP = 1.3

# 窗口及事件循环由独立进程拥有，避免抢占控制线程或混用相机预览的 GUI 后端。
_display: Optional[_TextDisplay] = None
_display_lock = Lock()


def _cv2():
    """Import and return ``cv2`` lazily."""
    import cv2
    return cv2


def _lines(text: str) -> List[str]:
    """按换行拆行；空文字是**调用方的错**，当场报出来，而不是显示一块白屏。"""
    value = "" if text is None else str(text)
    lines = value.splitlines() or [value]
    if not any(line.strip() for line in lines):
        raise ValueError("要显示的文字不能为空")
    return lines


def _canvas_size(size: Sequence[int]) -> Tuple[int, int]:
    width, height = int(size[0]), int(size[1])
    if width <= 2 * _MARGIN or height <= 2 * _MARGIN:
        raise ValueError(f"画布太小，塞不下留白：收到 {size!r}")
    return width, height


def _measure(
    cv2, lines: Sequence[str], scale: float, thickness: int,
) -> Tuple[float, float]:
    """在给定字号下量出 ``(最宽行宽, 最高行高+基线下降)``；只用来排版。"""
    widest = 0.0
    tallest = 0.0
    for line in lines:
        (w, h), base = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        widest = max(widest, float(w))
        tallest = max(tallest, float(h + base))
    return widest, tallest


def _fit(cv2, lines: Sequence[str], width: int, height: int) -> Tuple[float, int]:
    """能把这些行**全部**塞进画布的统一字号，以及跟着变的笔画粗细。

    笔画是整数、且随字号变化，所以"按 1.0 量一次再线性缩放"会偏小或偏大一点点；
    这里量两遍：先估字号，再按真实的笔画重量复量一次，超了就按比例收一点。
    """
    usable_w = width - 2 * _MARGIN
    usable_h = height - 2 * _MARGIN
    widest, tallest = _measure(cv2, lines, 1.0, 1)
    scale = min(
        _MAX_SCALE,
        usable_w / widest if widest > 0 else _MAX_SCALE,
        usable_h / (len(lines) * tallest * _LINE_GAP) if tallest > 0 else _MAX_SCALE,
    )
    thickness = max(1, int(round(scale * _STROKE_RATIO)))
    # 第二遍：笔画变粗会让字变宽/变高，按复量结果收一次。
    widest, tallest = _measure(cv2, lines, scale, thickness)
    overflow = max(
        widest / usable_w if widest > 0 else 0.0,
        len(lines) * tallest * _LINE_GAP / usable_h if tallest > 0 else 0.0,
    )
    if overflow > 1.0:
        scale /= overflow
        thickness = max(1, int(round(scale * _STROKE_RATIO)))
    return scale, thickness


def render_text(text: str, *, size: Sequence[int] = DEFAULT_SIZE) -> np.ndarray:
    """把文字画成一张 BGR 画布并返回；纯函数，不开窗口、不读相机。

    字号按画布自动缩小（见 :func:`_fit`），所以字符串再长也不会画出边界；
    多行由文字里的换行符决定，整体水平、垂直居中。
    """
    cv2 = _cv2()
    lines = _lines(text)
    width, height = _canvas_size(size)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = BACKGROUND

    scale, thickness = _fit(cv2, lines, width, height)
    _widest, tallest = _measure(cv2, lines, scale, thickness)
    step = max(1, int(round(tallest * _LINE_GAP)))
    # putText 的 y 是**基线**，所以第一行的基线要再往下让出字高的一半左右。
    baseline = (height - step * len(lines)) // 2 + int(round(tallest))
    for line in lines:
        (w, _h), _base = cv2.getTextSize(
            line, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness,
        )
        cv2.putText(
            canvas,
            line,
            ((width - w) // 2, baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            COLOUR,
            thickness,
            cv2.LINE_AA,
        )
        baseline += step
    return canvas


class _TextDisplay:
    """只传文字和尺寸，显示进程确认绘制后返回，不传相机帧或模型。"""

    def __init__(self) -> None:
        self.process = subprocess.Popen(
            [sys.executable, "-u", "-m",
             "modules.vision.drive.text_screen", "--worker"],
            cwd=Path(__file__).resolve().parents[3],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            # 隐藏 Windows 子进程控制台，HighGUI 窗口仍正常显示。
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        self.responses: Queue = Queue()
        self.reader = Thread(target=self._read_responses, daemon=True)
        self.reader.start()

    def _read_responses(self) -> None:
        """等待管道时休眠；EOF 也必须唤醒调用方，避免显示失败却返回成功。"""
        try:
            for line in self.process.stdout:
                self.responses.put(json.loads(line))
        except (OSError, ValueError) as exc:
            self.responses.put({"error": str(exc)})
        finally:
            self.responses.put({
                "error": "文字显示进程已退出，请检查显示后端和 DISPLAY",
            })

    def show(self, text: str, size: Tuple[int, int]) -> None:
        """等待本次绘制确认，之后主程序无需再处理窗口事件。"""
        try:
            self.process.stdin.write(json.dumps({"text": text, "size": size}) + "\n")
            self.process.stdin.flush()
            response = self.responses.get(timeout=10)
        except Empty as exc:
            raise RuntimeError("文字显示进程在 10 秒内未完成绘制") from exc
        except OSError as exc:
            raise RuntimeError("无法向文字显示进程发送内容") from exc
        if "error" in response:
            raise RuntimeError(response["error"])

    def close(self) -> None:
        """关闭输入让窗口正常退出；显示后端卡住时也不能拖住机器人退出。"""
        try:
            self.process.stdin.close()
        except OSError:
            pass
        try:
            self.process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.reader.join(timeout=1)
        self.process.stdout.close()


def _close_display() -> None:
    """显示窗口随主程序退出，避免留下占屏的孤儿进程。"""
    global _display
    with _display_lock:
        if _display is not None:
            _display.close()
            _display = None


atexit.register(_close_display)


def show_text(text: str, *, size: Sequence[int] = DEFAULT_SIZE) -> str:
    """显示文字并返回窗口名；窗口事件在独立进程中持续处理。

    首次调用需要启动显示进程，每次调用在绘制确认后返回，不等待按键。
    后续程序无需调用 waitKey；静态文字不反复绘制，也不读相机或运行推理。
    重复调用更新同一窗口，主进程的 destroyAllWindows 不影响它。
    关闭文字窗可点击窗口关闭按钮，主程序退出时也会自动关闭。

    :raises ValueError: 文字为空，或画布尺寸装不下留白。
    :raises RuntimeError: 显示进程启动、绘制失败或超时。
    """
    global _display
    value = "\n".join(_lines(text))
    dimensions = _canvas_size(size)
    with _display_lock:
        if _display is not None and _display.process.poll() is not None:
            _display.close()
            _display = None
        if _display is None:
            _display = _TextDisplay()
        try:
            _display.show(value, dimensions)
        except Exception:
            _display.close()
            _display = None
            raise
    return TEXT_WINDOW_NAME


def _run_display() -> int:
    """GUI 只在显示进程的主线程运行，兼容 Win32 和 Qt 后端。"""
    commands: Queue = Queue()

    def read_commands() -> None:
        """输入线程只等管道，不调用 GUI；主进程退出后的 EOF 负责收尾。"""
        try:
            for line in sys.stdin:
                commands.put(json.loads(line))
        except (OSError, ValueError) as exc:
            commands.put(exc)
        finally:
            commands.put(None)

    Thread(target=read_commands, daemon=True).start()
    cv2 = _cv2()
    cv2.setNumThreads(1)
    opened = False
    idle = object()
    try:
        while True:
            try:
                command = commands.get(timeout=0.05) if opened else commands.get()
            except Empty:
                command = idle
            if command is None:
                return 0
            if isinstance(command, Exception):
                raise command
            if command is not idle:
                canvas = render_text(command["text"], size=command["size"])
                if not opened:
                    open_preview(TEXT_WINDOW_NAME)
                    opened = True
                cv2.imshow(TEXT_WINDOW_NAME, canvas)
                cv2.resizeWindow(TEXT_WINDOW_NAME, canvas.shape[1], canvas.shape[0])
            # 每约 50 ms 泵一次事件；静态图不重画，空闲时阻塞等待，避免忙轮询。
            cv2.waitKey(1)
            # GTK2 不提供 WND_PROP_VISIBLE；图像区域能直接判断窗口是否还存在。
            try:
                _x, _y, width, height = cv2.getWindowImageRect(TEXT_WINDOW_NAME)
            except cv2.error:
                return 0
            if width < 0 or height < 0:
                return 0
            if command is not idle:
                print(json.dumps({"ok": True}), flush=True)
    except Exception as exc:
        print(json.dumps({"error": str(exc)}), flush=True)
        return 1
    finally:
        if opened:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        raise SystemExit(_run_display())
    show_text(sys.argv[1] if len(sys.argv) > 1 else "134+123+314+231")
    print("关闭文字窗口以结束")
    _display.process.wait()
