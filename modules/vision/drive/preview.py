"""OpenCV HighGUI 预览窗口的统一入口。

两个在 Windows 真机上实测过的坑，所有预览入口都从这里绕开：

1. **窗口名必须是 ASCII。** OpenCV 用 ANSI 代码页转换窗口名：本机实测把
   ``"项目视觉演示"`` 传进 ``cv2.imshow``，窗口标题栏真的变成了 ``"??????"``
   （Qt 后端里则是 ``"椤圭洰瑙嗚…"`` 这样的乱码）。窗口名只用来标识窗口，
   所以本项目统一用 :data:`PREVIEW_WINDOW_NAME`，非 ASCII 名字直接拒绝。
2. **先 ``namedWindow`` 再 ``imshow``。** 不同发行版的 OpenCV 用不同的窗口
   后端（pip 的 opencv-python 是 Win32UI，conda-forge 的是 Qt6），隐式建窗时
   标题转换和窗口初始尺寸都不一样；显式建窗后行为统一，标题也不走那条
   有损转换路径。

用法::

    from modules.vision import close_previews, open_preview

    window = open_preview()
    try:
        while True:
            cv2.imshow(window, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        close_previews()
"""

from __future__ import annotations

#: 所有预览窗口共用的 ASCII 窗口名。
PREVIEW_WINDOW_NAME = "vision-preview"

#: 连续这么多帧都是纯色（全黑/全白）时提示排查，而不是让用户对着黑窗口猜。
BLANK_FRAME_WARNING = 30


def _cv2():
    """Import and return ``cv2`` lazily."""
    import cv2
    return cv2


def open_preview(name: str = PREVIEW_WINDOW_NAME) -> str:
    """Create the HighGUI preview window explicitly and return its name.

    Raises:
        ValueError: ``name`` is empty or contains non-ASCII characters, which
            HighGUI renders as ``?`` in the title bar on Windows.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("窗口名不能为空")
    if not name.isascii():
        raise ValueError(
            f"HighGUI 窗口名必须只用 ASCII 字符，收到 {name!r}；"
            "中文名字在 Windows 上会显示成 ????（见 modules/vision/drive/preview.py 顶部说明）"
        )
    cv2 = _cv2()
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    return name


def close_previews() -> None:
    """Destroy every HighGUI window opened by the current process."""
    _cv2().destroyAllWindows()


def warn_blank_preview(camera_id: int, frames: int = BLANK_FRAME_WARNING) -> None:
    """Print the usual causes of a black or white camera preview window.

    ``read()`` keeps succeeding when another program owns the camera or the
    index points at a dead device, so the preview shows a flat colour instead
    of an error. The caller calls this once per run to turn that into advice.
    """
    print(f"[!] 摄像头 {camera_id} 连续 {frames} 帧输出纯色画面（全黑/全白）。逐项排查：")
    print("    · 是否被别的程序占用：系统“相机”App、钉钉/腾讯会议、浏览器里的摄像头页面，关掉再试")
    print("    · 序号是否选对：用系统“相机”App 确认能看到画面的序号，再改 modules/vision/config.json 的 windows.camera_id")
    print("    · 笔记本内置摄像头：Windows 设置 → 隐私和安全性 → 相机，允许桌面应用访问")
    print("    · 镜头盖/滑盖是否关着，或正对着完全黑暗的地方")
