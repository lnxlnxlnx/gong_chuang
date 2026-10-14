"""Menu-driven test for a trained obstacle model.

Reach it from the project test menu (no command line arguments needed)::

    python test.py      # 选择 15

It asks two questions --- where to run the test (USB camera, the validation
split, an image folder, one image / video file, or a camera self-check) and
which model and class table to use --- then prints one line per frame and
writes annotated JPEGs for the file-based modes, so a freshly exported model
can be checked without hardware.
"""

from __future__ import annotations

MENU_ENTRY = (15, "障碍物模型测试")

import os
from collections.abc import Callable
from pathlib import Path

from modules.vision import (
    DEFAULT_CLASS_NAMES,
    USBCamera,
    create_detector,
    default_model_path,
    load_camera_id,
    run_preview,
)

from project_paths import PROJECT_ROOT, models_path, output_path

#: Extension of the model this platform loads (matches ``modules.vision.default_model_path``).
MODEL_SUFFIX = ".engine" if os.name != "nt" else ".onnx"

#: Checkpoint exported by ``python modules/vision/training/train.py --export-onnx``.
NEW_MODEL = models_path(f"best_obstacle{MODEL_SUFFIX}")

#: Class table of the new single-class obstacle model, in training order.
NEW_CLASS_NAMES = ["obstacle"]

#: Validation split written by ``modules/vision/training/prepare_dataset.py``.
VAL_IMAGES = models_path("dataset", "images", "val")

#: Default output directory for annotated frames.
DEFAULT_SAVE = output_path("vision_predict")

#: Extensions accepted when the source is a folder or a single image.
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

#: Extensions treated as video files.
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}

#: Preview window title. Kept ASCII on purpose: OpenCV on Windows creates the
#: window through the ANSI API, so a Chinese title is displayed as "????" in
#: both the title bar and the taskbar.
WINDOW_TITLE = "obstacle model test"

#: How many consecutive failed reads are tolerated before giving up on a camera.
READ_ATTEMPTS = 30


def _relative(path: Path) -> str:
    """Show a path relative to the project root when possible."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _count_images(directory: Path) -> int:
    """Count image files directly inside ``directory`` (0 when unreadable)."""
    try:
        return sum(1 for path in directory.iterdir()
                   if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    except OSError:
        return 0


def _ask(prompt: Callable[[str], str], question: str, default: str) -> str:
    """Ask one question; an empty answer keeps ``default``."""
    return prompt(question).strip() or default


def _ask_float(prompt: Callable[[str], str], question: str, default: float) -> float:
    """Ask for a number, falling back to ``default`` on unparsable input."""
    text = _ask(prompt, question, str(default))
    try:
        return float(text)
    except ValueError:
        print(f"无法识别 {text!r}，改用 {default}")
        return default


def choose_plan(prompt: Callable[[str], str]):
    """Return ``(model_path, class_names)`` for the chosen plan, or ``None``."""
    competition = default_model_path()
    new_state = "" if NEW_MODEL.is_file() else "（未找到）"
    print("\n请选择要测试的模型：")
    print(f"  1. 新训练的障碍物模型  {_relative(NEW_MODEL)}  单类 obstacle {new_state}")
    print(f"  2. 比赛模型            {_relative(competition)}  "
          f"内置 {len(DEFAULT_CLASS_NAMES)} 类")
    print("  0. 返回菜单")
    choice = _ask(prompt, "请选择 [1]：", "1")

    if choice == "0":
        return None
    if choice == "2":
        if not competition.is_file():
            print(f"找不到模型文件：{competition}")
            return None
        return competition, None
    if choice != "1":
        print(f"无效选项：{choice}")
        return None
    if not NEW_MODEL.is_file():
        print(f"找不到模型文件：{NEW_MODEL}")
        print("先在终端里训练并导出：python training\\train.py --model yolo11n.pt --export-onnx")
        return None
    return NEW_MODEL, list(NEW_CLASS_NAMES)


def choose_mode(prompt: Callable[[str], str]):
    """Return the chosen test mode ('1'..'5'), '0' to go back, or ``None``."""
    print("\n请选择测试方式：")
    print("  1. 摄像头实时预览")
    print(f"  2. 验证集图片批量测试  {_relative(VAL_IMAGES)}  "
          f"（{_count_images(VAL_IMAGES)} 张）")
    print("  3. 指定图片文件夹")
    print("  4. 单个图片或视频文件")
    print("  5. 摄像头自检（0–3 号挨个试，看哪个能出画面；不需要模型）")
    print("  0. 返回菜单")
    choice = _ask(prompt, "请选择 [1]：", "1")
    if choice not in {"0", "1", "2", "3", "4", "5"}:
        print(f"无效选项：{choice}")
        return None
    return choice


def ask_source(prompt: Callable[[str], str], mode: str) -> Path | None:
    """Ask for the folder or file used by modes 3 and 4."""
    if mode == "3":
        text = _ask(prompt, f"图片文件夹 [默认 {_relative(VAL_IMAGES)}]：",
                    str(VAL_IMAGES))
    else:
        text = prompt("图片或视频文件路径：").strip()
        if not text:
            print("没有输入路径，已取消")
            return None
    source = Path(text).expanduser()
    if not source.exists():
        print(f"路径不存在：{source}")
        return None
    return source


def iter_frames(cv2, source: Path):
    """Yield ``(name, frame)`` from an image, an image folder or a video."""
    if source.is_dir():
        paths = sorted(path for path in source.iterdir()
                       if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        for path in paths:
            frame = cv2.imread(str(path))
            if frame is not None:
                yield path.stem, frame
        return

    if source.suffix.lower() in VIDEO_SUFFIXES:
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise SystemExit(f"无法打开视频：{source}")
        index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                yield f"{source.stem}_{index:05d}", frame
                index += 1
        finally:
            capture.release()
        return

    frame = cv2.imread(str(source))
    if frame is None:
        raise SystemExit(f"无法读取图片：{source}")
    yield source.stem, frame


def annotate_source(detector, cv2, source: Path, save_dir: Path) -> None:
    """Annotate an image, an image folder or a video without opening a window."""
    save_dir.mkdir(parents=True, exist_ok=True)
    frames = boxes = 0
    for name, frame in iter_frames(cv2, source):
        detections = detector.detect(frame)
        cv2.imwrite(str(save_dir / f"{name}.jpg"), detector.draw(frame, detections))
        frames += 1
        boxes += len(detections)
        summary = ", ".join(f"{det.name} {det.score:.2f}" for det in detections) or "未检出"
        print(f"  {name}: {summary}")
    print(f"\n共 {frames} 张图、{boxes} 个框；带框图片已写入 {save_dir}")


def ask_camera_id(prompt: Callable[[str], str]) -> int:
    """Ask for the camera index, defaulting to ``modules/vision/config.json``."""
    platform = "jetson" if os.name != "nt" else "windows"
    try:
        default_camera = load_camera_id(platform)
    except ValueError as error:
        print(f"读取 config/vision.json 失败：{error}")
        default_camera = 0
    text = _ask(prompt, f"摄像头序号 [{default_camera}]：", str(default_camera))
    try:
        return int(text)
    except ValueError:
        print(f"无法识别 {text!r}，改用 {default_camera}")
        return default_camera


def first_frame(camera, attempts: int = READ_ATTEMPTS):
    """Return the first readable frame; cameras routinely drop the first reads."""
    for _ in range(attempts):
        frame = camera.read()
        if frame is not None:
            return frame
    return None


def open_camera(cv2, camera_id: int):
    """Open one camera index, or return ``None`` and explain why it failed."""
    print(f"打开摄像头 {camera_id} ...")
    try:
        return USBCamera(camera_id, width=640, height=480, fps=30).initialize()
    except RuntimeError as error:
        print(f"打不开摄像头 {camera_id}：{error}")
        print("回菜单选 5 可以自检 0–3 号；也可以改 modules/vision/config.json 里的 camera_id")
        return None


def scan_cameras(cv2) -> None:
    """Try indices 0–3 once each and report which ones really deliver frames."""
    print("\n摄像头自检：依次试 0–3 号，每号最多等 10 帧 ...")
    working = []
    for index in range(4):
        camera = open_camera(cv2, index)
        if camera is None:
            continue
        with camera:
            frame = first_frame(camera, attempts=10)
            if frame is None:
                print(f"  序号 {index}：能打开，但取不到画面")
                continue
            brightness = float(frame.mean())
            print(f"  序号 {index}：{frame.shape[1]}x{frame.shape[0]}，平均亮度 "
                  f"{brightness:.0f}" + ("（全黑：可能被占用或未授权）" if brightness < 1 else ""))
            working.append(index)
    print(f"自检结束，能出画面的序号：{working or '无'}")
    print("把可用的序号填进 modules/vision/config.json 对应平台的 camera_id，然后回菜单选 1 预览")


def preview_camera(detector, cv2, prompt: Callable[[str], str]) -> None:
    """Show the live USB camera preview until the user presses q."""
    camera_id = ask_camera_id(prompt)
    camera = open_camera(cv2, camera_id)
    if camera is None:
        return

    with camera:
        frame = first_frame(camera)
        if frame is None:
            print(f"摄像头 {camera_id} 打开了，但连续 {READ_ATTEMPTS} 次都取不到画面。常见原因：")
            print("  · 序号不对（回菜单选 5 自检，或改 modules/vision/config.json 的 camera_id）")
            print("  · 被别的程序占用（相机、钉钉/腾讯会议、浏览器里的网页）")
            print("  · 笔记本内置摄像头被 Windows 隐私设置禁止桌面应用访问")
            return

        print(f"画面 {frame.shape[1]}x{frame.shape[0]}，开始预览；按 q 退出")
        print(f"（窗口标题是英文 \"{WINDOW_TITLE}\"：OpenCV 在 Windows 下画不出中文标题）")
        print("（窗口左下角 cap/det/age：取帧与推理各跑一条线程，"
              "cap 是摄像头实际帧率，det 是识别实际帧率）")
        # 第一帧已经用 READ_ATTEMPTS 次重试拿到手了（摄像头每次打开都先丢几帧），
        # 交给 run_preview 当起点，免得两处各写一套等待逻辑。
        run_preview(
            camera,
            detector.detect,
            detector.draw,
            window=WINDOW_TITLE,
            first=frame,
            print_every=30,
        )

    cv2.destroyAllWindows()
    print("预览已关闭")


def run_manual(prompt: Callable[[str], str]) -> None:
    """Ask what to test, then run it; nothing is opened before a choice is made."""
    mode = choose_mode(prompt)
    if mode in (None, "0"):
        return

    import cv2

    if mode == "5":  # the camera self-check needs no model at all
        scan_cameras(cv2)
        return

    plan = choose_plan(prompt)
    if plan is None:
        return
    model_path, class_names = plan

    source = ask_source(prompt, mode) if mode in {"3", "4"} else None
    if mode in {"3", "4"} and source is None:
        return
    conf = _ask_float(prompt, "置信度阈值 0..1 [0.25]：", 0.25)

    print(f"\n加载模型：{_relative(model_path)}（类别表："
          f"{'、'.join(class_names) if class_names else f'内置 {len(DEFAULT_CLASS_NAMES)} 类'}）")
    with create_detector(model_path, class_names=class_names,
                         conf_thresh=conf) as detector:
        if mode == "1":
            preview_camera(detector, cv2, prompt)
        else:
            annotate_source(detector, cv2, source or VAL_IMAGES, DEFAULT_SAVE)


def main() -> None:
    run_manual(prompt=input)


if __name__ == "__main__":
    main()
