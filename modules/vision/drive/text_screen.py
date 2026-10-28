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

命令行自测（按任意键之前窗口不关）::

    python -m modules.vision.drive.text_screen "134+123+314+231"
"""

from __future__ import annotations

import sys
from typing import List, Sequence, Tuple

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

#: 已经建过的窗口。OpenCV 的 ``imshow`` 会隐式建窗，但不同后端建出来的标题与
#: 初始尺寸不一致，所以第一条显示走 :func:`open_preview` 显式建窗（它也顺手拒掉
#: 非 ASCII 名字），之后只覆盖图像。
_created: set = set()


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


def _measure(cv2, lines: Sequence[str], scale: float, thickness: int) -> Tuple[float, float]:
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
        (w, _h), _base = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
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


def show_text(text: str, *, size: Sequence[int] = DEFAULT_SIZE) -> str:
    """把文字显示在板载屏幕上，**返回后窗口仍然留着**；返回窗口名。

    不阻塞、不关闭：只刷一帧就返回，调用方可以接着做别的事，也可以再调一次覆盖
    同一块屏。窗口随进程退出而消失；要提前关掉用
    ``cv2.destroyWindow(modules.vision.drive.text_screen.TEXT_WINDOW_NAME)``。

    **它不碰相机、模型和串口**：只画一张画布 + 一次 1 ms 的 ``waitKey``，所以和其余
    模块互不阻塞。窗口层面只有一条要记住：别的工具收尾时调用的
    ``preview.close_previews()`` 是 ``destroyAllWindows()``，会把它一起关掉；那时候
    再调一次本函数就回来了（``imshow`` 会自己重建窗口）。

    :raises ValueError: 文字为空，或画布尺寸装不下留白。
    :raises cv2.error: 板子上没有可用的显示后端（没接屏、没设 ``DISPLAY``）。
    """
    canvas = render_text(text, size=size)
    cv2 = _cv2()
    if TEXT_WINDOW_NAME not in _created:
        open_preview(TEXT_WINDOW_NAME)
        _created.add(TEXT_WINDOW_NAME)
    cv2.imshow(TEXT_WINDOW_NAME, canvas)
    # 只 imshow 不泵事件，某些后端不会真的把这一帧画出来。等一下就够了，不等按键。
    cv2.waitKey(1)
    return TEXT_WINDOW_NAME


if __name__ == "__main__":
    show_text(sys.argv[1] if len(sys.argv) > 1 else "134+123+314+231")
    print("按任意键关闭窗口")
    _cv2().waitKey(0)
    _cv2().destroyAllWindows()
