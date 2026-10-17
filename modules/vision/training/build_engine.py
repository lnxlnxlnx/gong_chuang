#!/usr/bin/env python
"""在 Jetson 上把 ONNX 转成 TensorRT engine，并验证引擎真的能推理。

**必须在 Jetson 上跑。** TensorRT engine 是序列化后的、为"这块板子的 GPU 架构 +
这个 TensorRT 版本"编译好的推理计划：PC 上生成的 engine 拿到 Jetson 上反序列化会直接
失败（或更糟，报一个和模型无关的错）。所以这一步刻意留在板上做，仓库里也刻意不放任何
现成的 ``.engine`` 文件。

用法（在项目根目录，Jetson 的 tensorrt 环境里）::

    python -m modules.vision.training.build_engine              # 路径取自 modules/vision/config.json
    python -m modules.vision.training.build_engine --force      # 覆盖已存在的 engine
    python -m modules.vision.training.build_engine --no-fp16    # 关掉 FP16（默认开）
    python -m modules.vision.training.build_engine --check      # 不动模型，只报告已有 engine

构建完再拿一张真实照片验收整条链路（预处理 → 引擎 → 解码 → 类名）::

    python -m modules.vision.training.build_engine --check --image models/top_pic_2.jpg

路径从哪来
----------

ONNX 和 engine 的路径**优先读 ``modules/vision/config.json`` 的 ``model`` 块**
（``model.onnx`` / ``model.engine``），读不到就退回 ``models/best.onnx`` /
``models/best.engine``。这里的读取是刻意自包含的：本脚本要在还没有任何推理依赖的
板子上跑起来，不应该因为推理侧的配置代码搬家就跟着坏。两边共用同一组键名。
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

from project_paths import PROJECT_ROOT, models_path

#: 没配 ``model.onnx`` 时的默认 ONNX 路径。
DEFAULT_ONNX = models_path("best.onnx")

#: 没配 ``model.engine`` 时的默认 engine 路径。
DEFAULT_ENGINE = models_path("best.engine")

#: 承载模型路径的配置文件，与 ``modules.vision.config`` 读的是同一份。
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.json"

#: ``trtexec`` 的常见位置；PATH 里找不到时按这些路径再找一遍。
TRTEXEC_CANDIDATES = (
    "/usr/src/tensorrt/bin/trtexec",
    "/usr/local/tensorrt/bin/trtexec",
)

#: 构建工作区大小（MiB）。Orin Nano 上 320×320 的 YOLO 用不了这么多，给足即可。
DEFAULT_WORKSPACE_MB = 2048

#: ``--check`` 里计时用的迭代次数（前若干次只热身，不计入）。
CHECK_ITERATIONS = 50
CHECK_WARMUP = 10


def _config_model_block(config_path: Path = CONFIG_PATH) -> dict:
    """读 ``config.json`` 的 ``model`` 块；文件不在、块不在或读坏了都返回空 dict。

    这里不 import 推理侧的配置模块：本脚本的职责是"把一个文件转成另一个文件"，
    它只要能在板子上独立跑起来。键名与 ``modules.vision.config.ModelSettings`` 一致。
    """
    try:
        document = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    block = document.get("model") if isinstance(document, dict) else None
    return block if isinstance(block, dict) else {}


def _configured_path(key: str, fallback: Path, config_path: Path = CONFIG_PATH) -> Path:
    """配置里的模型路径；相对路径按项目根解析，不按当前工作目录。"""
    value = _config_model_block(config_path).get(key)
    if not isinstance(value, str) or not value.strip():
        return fallback
    path = Path(value.strip()).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def default_onnx_path(config_path: Path = CONFIG_PATH) -> Path:
    """默认要转换的 ONNX：``config.json`` 的 ``model.onnx``，否则 ``models/best.onnx``。"""
    return _configured_path("onnx", DEFAULT_ONNX, config_path)


def default_engine_path(config_path: Path = CONFIG_PATH) -> Path:
    """默认写出的 engine：``config.json`` 的 ``model.engine``，否则 ``models/best.engine``。"""
    return _configured_path("engine", DEFAULT_ENGINE, config_path)


def configured_class_names(config_path: Path = CONFIG_PATH) -> list[str] | None:
    """``config.json`` 的 ``model.class_names``；没配就返回 ``None``（用内置类别表）。"""
    value = _config_model_block(config_path).get("class_names")
    if not isinstance(value, list) or not value:
        return None
    names = [str(item).strip() for item in value if str(item).strip()]
    return names or None


def find_trtexec(explicit: Path | None = None) -> Path | None:
    """定位 ``trtexec``：先看 ``--trtexec``，再看 PATH，最后看 JetPack 的固定安装位置。"""
    if explicit is not None:
        return explicit if explicit.is_file() else None
    found = shutil.which("trtexec")
    if found:
        return Path(found)
    for candidate in TRTEXEC_CANDIDATES:
        if Path(candidate).is_file():
            return Path(candidate)
    return None


def is_jetson() -> bool:
    """是不是在一块 Tegra 板子上跑（JetPack 会在 ``/etc/nv_tegra_release`` 留一份说明）。"""
    return platform.machine() == "aarch64" and Path("/etc/nv_tegra_release").is_file()


def _workspace_flag(trtexec: Path) -> str:
    """TensorRT 8.6 起 ``--workspace`` 改名成 ``--memPoolSize``；按帮助文本挑一个能用的。"""
    try:
        help_text = subprocess.run(
            [str(trtexec), "--help"], capture_output=True, text=True, timeout=60
        ).stdout
    except (OSError, subprocess.SubprocessError):
        help_text = ""
    return "--memPoolSize=workspace:{}MiB" if "--memPoolSize" in help_text else "--workspace={}"


def trtexec_command(
    trtexec: Path,
    onnx_path: Path,
    engine_path: Path,
    *,
    fp16: bool,
    workspace_mb: int,
    verbose: bool,
) -> list[str]:
    """拼出构建命令；单独一个函数是为了能打印给人看、也能被测试钉住。"""
    command = [
        str(trtexec),
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        _workspace_flag(trtexec).format(workspace_mb),
    ]
    if fp16:
        command.append("--fp16")
    if verbose:
        command.append("--verbose")
    return command


def build_with_trtexec(command: list[str]) -> int:
    """流式跑 ``trtexec``：构建要几分钟，把它的输出实时透出来，别让人以为卡死了。"""
    print("$ " + " ".join(command), flush=True)
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except OSError as error:
        print(f"跑不起 trtexec：{error}", file=sys.stderr)
        return 2
    assert process.stdout is not None
    try:
        for line in process.stdout:
            print("  " + line.rstrip())
    except KeyboardInterrupt:
        process.kill()
        print("\n已中断；engine 没有生成完整文件", file=sys.stderr)
        return 130
    return process.wait()


def build_with_python_trt(
    onnx_path: Path, engine_path: Path, *, fp16: bool, workspace_mb: int
) -> int:
    """``trtexec`` 不在时的退路：直接用 TensorRT 的 Python API 构建。"""
    try:
        import tensorrt as trt
    except ImportError as error:
        print(f"既没有 trtexec，也 import 不到 tensorrt：{error}", file=sys.stderr)
        return 2

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    print(f"解析 ONNX：{onnx_path}")
    if not parser.parse(onnx_path.read_bytes()):
        for index in range(parser.num_errors):
            print(f"  ONNX 解析错误：{parser.get_error(index)}", file=sys.stderr)
        return 2

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb << 20)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    print(f"构建 engine（TensorRT {trt.__version__}，fp16={fp16}）...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("构建失败：build_serialized_network 返回了 None", file=sys.stderr)
        return 3
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))
    return 0


def _human_size(path: Path) -> str:
    return f"{path.stat().st_size / (1024 * 1024):.1f} MB"


def _describe_engine(engine_path: Path):
    """加载引擎并报出输入/输出形状与单帧耗时；返回 (detector, 每帧毫秒)。"""
    import numpy as np

    from modules.vision.drive.yolov8 import YOLOv8Detector

    detector = YOLOv8Detector(str(engine_path))
    blob = np.zeros(detector.input_shape, dtype=np.float32)
    for _ in range(CHECK_WARMUP):
        detector.infer(blob)
    started = time.perf_counter()
    for _ in range(CHECK_ITERATIONS):
        detector.infer(blob)
    elapsed_ms = (time.perf_counter() - started) * 1000.0 / CHECK_ITERATIONS
    return detector, elapsed_ms


def check_engine(
    engine_path: Path,
    *,
    image: Path | None = None,
    class_names: list[str] | None = None,
) -> int:
    """报告引擎的输入输出、推理耗时；给了图片就顺带跑一遍真实检测。"""
    if not engine_path.is_file():
        print(f"找不到 engine：{engine_path}", file=sys.stderr)
        print("先构建：python -m modules.vision.training.build_engine", file=sys.stderr)
        return 2
    try:
        detector, elapsed_ms = _describe_engine(engine_path)
    except Exception as error:  # TensorRT 的失败类型很杂，统一翻译成"重建一次"
        print(f"加载 engine 失败：{type(error).__name__}: {error}", file=sys.stderr)
        print(
            "engine 和当前这块板子/TensorRT 版本对不上时必须重建"
            "（engine 不能从别的机器拷过来）：",
            file=sys.stderr,
        )
        print("  python -m modules.vision.training.build_engine --force", file=sys.stderr)
        return 3

    try:
        print(f"engine       : {engine_path}  ({_human_size(engine_path)})")
        print(f"输入张量     : {detector.input_shape}")
        print(f"输出张量     : {detector.output_shape}")
        print(f"输入边长     : {detector.input_size}")
        classes = class_names if class_names else list(getattr(detector, "class_names", ()) or ())
        print(f"类别表       : {', '.join(classes) if classes else '(未知)'}")
        print(f"单帧推理     : {elapsed_ms:.2f} ms（{CHECK_ITERATIONS} 次平均，含显存拷贝）")

        # 类别表错了框还是准的、只是名字整体错位；输出通道数是最便宜的交叉验算：
        # YOLOv8/v11 的输出是 (1, 4 + 类别数, 锚框数)。
        shape = tuple(detector.output_shape)
        if len(shape) == 3 and classes and shape[1] != 4 + len(classes):
            print(
                f"\n⚠ 输出通道数是 {shape[1]}，按 {len(classes)} 类应该是 {4 + len(classes)}："
                f"类别表和模型对不上，框会是对的、类名会整体错位。"
                f"改 config.json 的 model.class_names（顺序=训练时的类别顺序）。"
            )

        if image is not None:
            if not image.is_file():
                print(f"找不到图片：{image}", file=sys.stderr)
                return 2
            import cv2

            frame = cv2.imread(str(image))
            if frame is None:
                print(f"读不出图片：{image}", file=sys.stderr)
                return 2
            if class_names:
                detector.class_names = list(class_names)
            started = time.perf_counter()
            hits = detector.detect(frame)
            total_ms = (time.perf_counter() - started) * 1000.0
            print(f"\n{image}  {frame.shape[1]}x{frame.shape[0]}  "
                  f"检出 {len(hits)} 个框（端到端 {total_ms:.1f} ms）")
            for hit in hits:
                print(
                    f"  {hit.name:<10} score={hit.score:.3f} "
                    f"box=({hit.x1:.0f},{hit.y1:.0f})-({hit.x2:.0f},{hit.y2:.0f})"
                )
            if not hits:
                print("  一个框都没有：确认图片里有物料，或调低 conf_thresh")
    finally:
        detector.close()
    return 0


def _print_environment() -> None:
    print(f"平台         : {platform.machine()} ({platform.system()})"
          f"{'' if is_jetson() else '  ← 不在 Jetson 上；engine 必须在板上生成'}")
    try:
        import tensorrt

        print(f"TensorRT     : {tensorrt.__version__}")
    except ImportError:
        print("TensorRT     : 没装（当前 Python 里 import 不到 tensorrt）")
    try:
        import pycuda

        version = getattr(pycuda, "VERSION", None)
        print(f"pycuda       : {'.'.join(map(str, version)) if version else '已安装'}")
    except ImportError:
        print("pycuda       : 没装（TensorRT 通道需要它，见 README 的环境说明）")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="在 Jetson 上把 ONNX 转成 TensorRT engine（engine 不能跨机器生成）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--onnx", type=Path, default=None,
                        help="输入 ONNX；默认取 config.json 的 model.onnx")
    parser.add_argument("--engine", type=Path, default=None,
                        help="输出 engine；默认取 config.json 的 model.engine")
    parser.add_argument("--trtexec", type=Path, default=None,
                        help="trtexec 路径；默认在 PATH 和 JetPack 固定位置里找")
    parser.add_argument("--workspace-mb", type=int, default=DEFAULT_WORKSPACE_MB,
                        help="构建工作区大小（MiB）")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false",
                        help="不启用 FP16（默认启用，Orin 上快很多）")
    parser.add_argument("--verbose", action="store_true", help="把 trtexec 的详细日志也打出来")
    parser.add_argument("--force", action="store_true",
                        help="engine 已存在时也重新构建（默认跳过没变过的）")
    parser.add_argument("--python-builder", action="store_true",
                        help="不用 trtexec，直接用 TensorRT 的 Python API 构建")
    parser.add_argument("--check", action="store_true",
                        help="只检查已有 engine，不构建")
    parser.add_argument("--image", type=Path, default=None,
                        help="配合 --check：在一张真实图片上跑一遍端到端检测")
    parser.add_argument("--class-names", default=None,
                        help='配合 --check：逗号分隔的类别表，例如 "red,yellow,blue,green"；'
                             "默认取 config.json 的 model.class_names")
    return parser


def _is_stale(engine_path: Path, onnx_path: Path) -> bool:
    """engine 比 ONNX 旧就说明模型换过了，需要重建。"""
    return engine_path.stat().st_mtime < onnx_path.stat().st_mtime


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    onnx_path = (args.onnx or default_onnx_path()).resolve()
    engine_path = (args.engine or default_engine_path()).resolve()
    class_names = (
        [name.strip() for name in args.class_names.split(",") if name.strip()]
        if args.class_names
        else configured_class_names()
    )

    if args.check:
        _print_environment()
        print(f"ONNX 路径    : {onnx_path}")
        return check_engine(engine_path, image=args.image, class_names=class_names)

    print("=" * 72)
    _print_environment()
    print(f"ONNX         : {onnx_path}")
    print(f"engine 输出  : {engine_path}")
    print(f"FP16         : {'开' if args.fp16 else '关'}    工作区：{args.workspace_mb} MiB")
    print("=" * 72)

    if not onnx_path.is_file():
        print(f"找不到 ONNX：{onnx_path}", file=sys.stderr)
        print("先把导出的模型放到这个路径（或改 config.json 的 model.onnx）；"
              "Windows 上导出的 best.onnx 直接拷过来即可，转换在板上做。", file=sys.stderr)
        return 2

    if engine_path.exists() and not args.force:
        note = "ONNX 更新过，建议重建" if _is_stale(engine_path, onnx_path) else "看起来还是新的"
        print(f"engine 已存在（{_human_size(engine_path)}，{note}）：{engine_path}")
        if not _is_stale(engine_path, onnx_path):
            print("没有要做的；要强制重建加 --force，只想看看它就加 --check")
            return 0
        print("要重建就加 --force：python -m modules.vision.training.build_engine --force")
        return 0

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    if args.python_builder:
        status = build_with_python_trt(
            onnx_path, engine_path, fp16=args.fp16, workspace_mb=args.workspace_mb
        )
    else:
        trtexec = find_trtexec(args.trtexec)
        if trtexec is None:
            print("没找到 trtexec，改用 TensorRT Python API 构建"
                  "（想用 trtexec 就加 --trtexec /path/to/trtexec）")
            status = build_with_python_trt(
                onnx_path, engine_path, fp16=args.fp16, workspace_mb=args.workspace_mb
            )
        else:
            print(f"trtexec      : {trtexec}")
            status = build_with_trtexec(
                trtexec_command(
                    trtexec,
                    onnx_path,
                    engine_path,
                    fp16=args.fp16,
                    workspace_mb=args.workspace_mb,
                    verbose=args.verbose,
                )
            )

    if status != 0 or not engine_path.is_file():
        print(f"\n构建失败（退出码 {status}）", file=sys.stderr)
        return status or 3

    print(f"\n构建完成：{engine_path}  ({_human_size(engine_path)}，"
          f"用时 {time.perf_counter() - started:.0f} s)")
    print("接着验收一遍（加载引擎 + 真实图片端到端）：")
    print(f"  python -m modules.vision.training.build_engine --check"
          f"{' --image models/top_pic_2.jpg' if args.image is None else f' --image {args.image}'}")
    return check_engine(engine_path, image=args.image, class_names=class_names)


if __name__ == "__main__":
    raise SystemExit(main())
