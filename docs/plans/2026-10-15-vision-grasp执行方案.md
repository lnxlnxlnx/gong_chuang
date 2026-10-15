# 视觉与抓取接口重构执行说明

> **给执行 agent：**按本文逐项完成并验证。可使用 `superpowers:executing-plans`；不得把文档中的未来命令当成本轮已执行的结果。

**Goal：**让上层只传现有配置，通过 `Vision` / `Grasp` 对象使用视觉定位与抓取前就位，设备由对象管理。

**Architecture：**Vision 接收配置快照，实际拥有相机、定位器与模型；Grasp 实际拥有观测源、机械臂及主流程。现有数值、观测、脉冲原语按唯一职责迁移，控制算法保持原逻辑。

**Tech Stack：**Python 3.8+、unittest、OpenCV、现有 ONNX/TensorRT 后端、现有机械臂驱动；PowerShell。

**设计依据：**[总方案](../可评审性优化方案.md)、[接口专项](../vision-grasp复杂度专项.md)。三份文档一起交给执行 agent；本文件把设计选择落实到文件、函数和验收。

**状态：2026-10-15 编写并**已于同日执行完成**（任务 A→B→(C+D)→E）。静态依据为 `dev` / `3f1ceb0` 加当前工作区未提交内容。函数名是定位依据，行号可能变化。
**未执行的部分：**批次 5A/5B（退出菜单 19 与 laser_scan）、所有真机验收。二者按本文 §0 属独立批次 / 用户发起范围。

## 0. 执行边界与停止条件

- 本轮实现只覆盖总方案 P0/P1 中的视觉、抓取接口及必要消费者；不执行菜单 10/11/19 或 laser_scan 退出，不重写底盘、训练与其他业务模块。
- 不安装、恢复或更新依赖，不修改依赖文件、现场 JSON、标定值、端口、地址和脉冲默认值。
- 保留当前工作区已有修改，不从 HEAD 覆盖它们。实施前记录差异；出现与本任务无关的新增差异时保留并报告，不顺手整理。
- 不留旧入口兼容壳、不新造结果 DTO、不新增全局会话/设备缓存、模式框架或自动重连/重标/来源切换。
- 现有定位策略及其已存在的判据、粗焦距策略、动作预算与停稳逻辑原样保留；本次不能新增算法降级。配置无效直接失败，迁移后的正常会话不使用“坏配置后按默认值继续”的工具逻辑。
- 缺工具/依赖、基线存在无法归因的失败、源码与本文的业务契约冲突时，记录准确原因并停止依赖该条件的任务；不删测试、不改业务规则把结果做绿。
- 不主动运行真机或交互菜单。离线测试必须使用假设备；真机验收由用户在现有操作规范下发起。
- 新代码以 UTF-8 保存；给类、方法、单位、资源归属和显式延迟导入写简短注释。不得使用 Python 3.11 的 `ExceptionGroup` / `add_note` 等 API。

以下所有仓内路径以项目根为基准：`E:\SC_project\note\obsidian_note\fifth_semester\python_study\gong_chuang`。

## 1. 最终公开接口：不得自行加减

### 1.1 视觉

```python
from modules.vision import Vision

with Vision.open(config_path="modules/vision/config.json") as vision:
    report = vision.locate()
    if report.best is not None:
        print(report.best.x_mm, report.best.y_mm, report.best.z_mm)
```

```python
@classmethod
def open(cls, *, config_path=None, calibration_path=None) -> "Vision":
    """读取指定配置，加载模型并打开相机；返回拥有资源的会话。"""

def locate(self) -> "MaterialReport":
    """取帧并定位一次；读帧失败抛 RuntimeError，无物料返回正常空报告。"""

def close(self) -> None:
    """释放所有已取得资源；重复关闭不再执行资源动作。"""
```

另保留结构接口 `frame(self, *, timeout=None, max_age_seconds=None, min_received_at=None)`，返回 `Optional[Tuple[MaterialSighting, ...]]`，供抓取使用。未读到帧返回 `None`，空报告转换为空元组。`locate()` 和 `frame()` 不互相调用，统一调用 §4 的 `_read_report()`。

### 1.2 抓取

```python
from modules.grasp import Grasp

with Grasp.open(
    config_path="modules/grasp/eye_in_hand.json",
    vision_config_path="modules/vision/config.json",
) as grasp:
    result = grasp.align_and_descend()
```

```python
@classmethod
def open(cls, *, config_path=None, vision_config_path=None,
         vision_calibration_path=None, arm_config_path=None,
         motor_config_path=None) -> "Grasp":
    """从现有配置创建本机视觉、机械臂和普通动作会话。"""

def align_xy(self) -> "ServoResult":
    """包含原 run_align 的可选悬停，再按原规则检查安全高度并对准。"""

def align_and_descend(self, *, z_pulses=None,
                      retract_pulses=None) -> "GraspReadyResult":
    """执行抓取前就位；不闭合夹爪，不承诺完成搬运。"""

def close(self) -> None:
    """依次尝试停止、失能、关串口、关观测源；全部尝试后报告清理错误。"""
```

两个类都提供 `__enter__` / `__exit__`，关闭后操作抛 `RuntimeError`，不能自动重开。`open` 用 `@classmethod`；有设备状态的动作是实例方法，不能用静态方法保存隐式全局设备。

本节带 docstring 的代码只列接口签名；不能把只有 docstring 的方法当作已完成实现。真正方法体按后续任务实现，原有算法按符号迁移表保留。

## 2. 文件和符号归属表

### 2.1 视觉

| 文件 | 精确修改 |
|---|---|
| 新建 `modules/vision/runtime.py` | 定义真实 `Vision`；包含取帧、观测转换、快照装配与清理，不从 grasp 导入任何符号 |
| 新建 `modules/vision/reports.py` | 从 `drive/material_locate.py` 原样移动 `RegionReport`、`MaterialReport`；保留字段、属性、summary、排序与类身份的唯一来源 |
| 新建 `modules/vision/detector_factory.py` | 从包出口移动 `create_configured_detector`、`create_detector` 的真实函数体 |
| 修改 `modules/vision/config.py` | VisionConfig 增加 model；同一 root 解析 model；从包出口移动 `default_model_path`、`require_model_path` |
| 修改 `modules/vision/drive/material_yolo.py` | 固定构造时模型快照和物料类别；不逐帧重读默认配置；工厂从实际定义导入 |
| 修改 `modules/vision/drive/material_locate.py` | 使用 reports 的唯一定义；图片处理仍不打开相机；相机分支进入真实会话 |
| 修改 `modules/vision/drive/camera.py` | 修复 initialize 局部 capture 尚未移交给实例时的失败释放 |
| 修改 `modules/vision/drive/yolov8.py` | TensorRT 部分初始化失败自行释放；关闭失败不能阻断其他已分配对象的释放；不改推理数学 |
| 修改 `modules/vision/__init__.py` | 最终仅显式导出 `Vision`、`MaterialReport`、`RegionReport`；删大惰性表与旧工具出口，消费者同批迁移 |
| 删除 `modules/grasp/drive/camera_source.py` | 真实职责与消费者迁完后删除，不保留转发文件 |

`runtime.py` 的固定私有接缝如下，后续任务与测试使用这些名称：

```python
def _load_settings(config_path=None, calibration_path=None) -> "VisionConfig":
    """方法内导入配置模块；None 解析为该模块现有 DEFAULT_* 常量。"""

def _build_locator(settings, *, focal_px=None) -> "MaterialLocator":
    """实际加载 detector 并构造定位器；失败时自行关闭已构造 detector。"""

def _new_camera(camera_id):
    """方法内导入 USBCamera；只构造 1280×720 相机，不调用 initialize。"""

def _override_settings(settings, *, model_path=None, camera_id=None,
                       conf_thresh=None, depth_range_mm=None,
                       material_bgr=None, material_palette=None,
                       colour_tolerance=None, top_diameter_mm=None):
    """为已有工具生成本次快照；None 表示不覆盖；不读写文件。"""
```

保留 `MaterialSighting` 在 `vision/drive/sighting.py` 的唯一定义；转换时在方法内导入。reports 顶层仅导入 stdlib，`LockedRegion / ClassResult / FaceQuality / Detection / RingLock` 只用于注解时放入 `TYPE_CHECKING`，不可为导出报告而预加载 CLI 驱动。

### 2.2 抓取：eye_in_hand 的完整迁移清单

下表中的旧符号均位于 `modules/grasp/drive/eye_in_hand.py`。**移动原函数体，禁止重写算法。** 原调用者改从真实定义导入；控制器为自身使用而导入类型不算兼容层，禁止为了保旧 import 专门加别名/转发。

| 新归属 | 移入符号 |
|---|---|
| 新建 `drive/servo_model.py` | `DEFAULT_CONFIG_PATH`、`MIN_JACOBIAN_DETERMINANT`、`BASE_AXIS`、`RADIAL_AXIS`、`LIFT_AXIS`、`ServoError`、`ObjectSelector`、`ServoTarget`、`ServoJacobian`、`ServoConfig`、`LiftScale`、`ServoResult`、`ZResult`、`HoverResult`、`ZDescentResult`、`GraspReadyResult` |
| 同上 | `load_servo_config`、`load_object_selector`、`load_target`、`load_jacobian`、`load_z_target`、`load_lift_scale`、`save_calibration`、`_load_document`、`_number`、`_integer`、`_boolean`、`_positive`、`_non_negative`、`_non_negative_integer`、`_optional_positive` |
| 新建 `drive/servo_observation.py` | `FrameSource`、`select_object`、`_read_object`、`measure_offset`、`_settled_sighting`、`measure_settled`、`VISIBILITY_PROBE_FRAMES`、`_probe_visible` |
| 新建 `drive/servo_motion.py` | `_move_axis`、`_send_axis`、`_move_axes`、`_send_lift`、`_move_lift` |
| 保留 `drive/eye_in_hand.py` | `_apply_minimum`、`_clamp_pulses`、`probe_jacobian`、`align_x_y`、`_require_consistent_legs`、`hover_above_object`、`raise_before_rotating`、`find_object`、`_try_lift_direction`、`probe_lift_scale`、`_lift_step_within_guard`、`align_z`、`descend_z`、`retract_extension`、现有 `_self_test` 与 CLI 入口 |
| 新建 `modules/grasp/runtime.py` | 定义真实 `Grasp` 类，包含会话创建/收尾、`align_xy`；原自由函数 `align_and_descend` 的**编排体移动到实例方法**，原自由函数删除 |
| 修改 `modules/grasp/__init__.py` | 显式导出 `Grasp`、`ServoError`、`ServoResult`、`GraspReadyResult`；不导出工具装配和全部内部助手 |

`servo_model.DEFAULT_CONFIG_PATH` 继续为 `Path(__file__).resolve().parent.parent / "eye_in_hand.json"`，不能因移动多/少跳一层。观测与模型文件对 `MaterialSighting` 仅有注解需求时使用 `TYPE_CHECKING`。model 不依赖 observation / motion / controller；observation 和 motion 互不导入。探针、安全抬升、悬停留在控制器，避免把相互调用拆成循环。

已有底层 `align_x_y`、`align_z` 等保留为**实际算法和现场工具内部能力**，不会再从顶层包导出；Grasp 包含真实资源所有权与主编排，不能只写 `return old_align_and_descend(...)`。

`grasp/__init__.py` 显式从 `runtime` 导出 Grasp，从 `drive.servo_model` 导出结果/异常。runtime 顶层只依赖 stdlib 与纯 model；动作方法内才从 eye_in_hand 导入所需算法。eye_in_hand 不反向依赖 runtime，原自测 CLI 保留；这样导入父包不会提前载入 `python -m modules.grasp.drive.eye_in_hand` 的目标模块。此处延迟的是明确的动作运行边界，不是用局部 import 绕过循环。

Grasp runtime 的最小运行时导入固定如下；其余只供类型注解使用的符号放 TYPE_CHECKING，设备与算法在明确的 open/动作边界加载：

```python
from __future__ import annotations

import time
from modules.grasp.drive.servo_model import GraspReadyResult, ServoError
```

## 3. 任务 A：记录基线，锁定行为

**涉及：**只读当前差异；必要日志放 `docs/research/`。以下命令均为未来执行阶段使用。

- [ ] 记录已有修改，确认解释器；预期看见实际工具路径、版本与用户修改，不能清空工作区。

```powershell
Get-Command git, python
git status --short
git diff --stat
python --version
```

- [ ] 运行当前完整离线套件；预期退出码 0，零失败/错误；若失败，记录具体测试名与原始堆栈。历史的 962/1/165/1 不是本次基线，不据此批量“修测试”。

```powershell
python -c "import test; test.run_offline_tests()"
```

- [ ] 从当前未提交源码保留并标记下列现有测试预期：观测换算、动作顺序、悬停、安全高度、新帧/停稳、平面未准零下降、预算超限。后续迁移修改 import/patch，不能重算预期去迎合实现。

## 4. 任务 B：实现 Vision 并同时迁移视觉消费者

**文件：**§2.1 中全部文件、§7.1 的消费者与相关测试。子项有先后依赖，正式出口与旧路径退出在本任务闭合时一起完成。

### B1. 报告、配置与工厂

- [ ] 移动报告定义并更新全部 `MaterialReport / RegionReport` 消费者。报告逻辑中实际需要的 `math` 等 stdlib 也移动；不要复制类。
- [ ] `VisionConfig` 的字段尾部增加 `model: ModelSettings = field(default_factory=ModelSettings)`，避免改变已有位置参数顺序；`load_vision_config` 从已经读到的 root 构造 model：

```python
value = root.get("model")
model = (ModelSettings(source=Path(config_path)) if value is None else
         _model_settings(_mapping(value, "model"), Path(config_path)))
# 在原 VisionConfig(...) 的其余参数后增加 model=model。
```

- [ ] `_load_settings` 将非 None 路径转为 Path，将 None 换为配置模块的 `DEFAULT_CONFIG_PATH / DEFAULT_CALIBRATION_PATH`，只调用一次 `load_vision_config`。不要在这条链调用 `load_model_settings()` 重读另一个默认文件。
- [ ] 移动模型工厂原实现；`detector_factory` 依赖 config 与后端，不能从包出口拿 `require_model_path`。`.onnx/.engine` 分支、模型元数据处理和 PROJECT_ROOT 路径规则保持原样。
- [ ] `MaterialYOLO.__init__` 增加 `settings=None, input_size=None`。未给快照时在构造时读取一次默认 ModelSettings；Vision 路线始终传快照。已有显式 `model_path/class_names/conf_thresh/iou_thresh/materials` 优先级保持；`detector` 显式注入时不再构造第二个 detector。独立测试使用自定义类别的假 detector 时，也显式传 `materials` 或 settings，不靠猜类别兜底。
- [ ] 删除 `_configured_materials`；构造时确定 `_materials = tuple(materials) if materials is not None else tuple(settings.materials)`；`effective_materials` 只返回该元组。`detector` 属性从真实工厂导入，并传入 `settings`、`input_size` 与已有显式覆盖。无逐帧读文件、异常切换默认类别表。

验收：自定义配置的 model、类别、物料子集、阈值和尺寸抵达实际 detector；后续改文件不影响该实例。测试 `test_model_config.py / test_material_vision.py / test_vision_calibration_config.py` 随定义更新，不保留对已删除 `_configured_materials` 的 patch。

### B2. 实际定位器构造

`runtime._build_locator` 采用以下装配，不通过 `MaterialLocator.__enter__` 假装模型已加载。局部 import 是明确的显式后端加载边界，不能用它掩盖 vision→grasp 依赖。

```python
def _build_locator(settings, *, focal_px=None):
    from modules.vision.detector_factory import create_configured_detector
    from modules.vision.drive.material_locate import MaterialLocator
    from modules.vision.drive.material_yolo import MaterialYOLO

    # 工厂在这里真实打开模型；内部接收完整 settings.model。
    detector = create_configured_detector(settings=settings.model)
    try:
        yolo = MaterialYOLO(detector=detector, settings=settings.model,
                            materials=settings.model.materials)
        targets = settings.targets
        return MaterialLocator(
            yolo=yolo, focal_px=focal_px,
            depth_range_mm=targets.working_distance_mm,
            material_bgr=targets.material_bgr,
            material_palette=targets.material_palette or None,
            colour_tolerance=targets.color_tolerance,
            top_diameter_mm=targets.material_diameter_mm,
        )
    except BaseException as primary:
        try:
            detector.close()
        except Exception as cleanup:
            raise primary.with_traceback(primary.__traceback__) from cleanup
        raise
```

本批**不传 `settings.focal_px`**；只有原工具显式 `--focal` 进入参数 `focal_px`。默认毫米坐标继续沿用原抓取的粗焦距策略。targets 中 palette 与单一 BGR 的既有优先级由 MaterialLocator 处理。

### B3. 会话初始化、取帧和转换

`Vision` 内部状态固定为 `_settings, _camera, _locator, _closed, _clock, _drain_frames, frames, misses`。构造不打开设备；`_clock = time.monotonic`，`_drain_frames = 2`（原值）。不用全局实例。测试可直接 patch 私有构造接缝，不给业务 API 增加假设备参数。

构造函数和关闭检查可以直接采用：

```python
def __init__(self, settings):
    self._settings = settings
    self._camera = None
    self._locator = None
    self._closed = False
    self._clock = time.monotonic
    self._drain_frames = 2
    self.frames = 0
    self.misses = 0

def _require_open(self):
    if self._closed:
        raise RuntimeError("Vision 已关闭")
```

```python
@classmethod
def open(cls, *, config_path=None, calibration_path=None):
    settings = _load_settings(config_path, calibration_path)
    return cls._open_config(settings)

@classmethod
def _open_config(cls, settings, *, focal_px=None):
    session = cls(settings)
    # camera_id 检查在分配模型之前；工厂调用在 initialize 相机之前。
    from modules.vision.config import current_platform
    camera_id = settings.camera_id(current_platform())
    try:
        session._locator = _build_locator(settings, focal_px=focal_px)
        session._camera = _new_camera(camera_id)
        session._camera.initialize()
        return session
    except BaseException as primary:
        try:
            session.close()
        except Exception as cleanup:
            raise primary.with_traceback(primary.__traceback__) from cleanup
        raise
```

- [ ] `_read_report(self, *, timeout=None)` 的第一行调用 `_require_open()`，后者只检查 closed 并抛 RuntimeError。随后**逐语句迁入原 CameraFrameSource.frame 的丢帧和读取循环**：`max(1, drain_frames)` 次丢帧；按原 0.05 秒间隔重试；超时返回 None 并更新 misses；读到帧后、推理前取 `_clock()`；一次 `locator.locate(frame)`；frames 加一；返回 `(report, stamp)`。不重用上次结果。
- [ ] `locate` 调一次 `_read_report()`；None 抛 `RuntimeError("读不到摄像头画面")`，否则返回 report。`frame` 调一次 `_read_report(timeout=timeout)`，None 原样返回，成功调 `_sightings_from_report(report, received_at=stamp)`。
- [ ] 将 `sighting_from_region / sightings_from_report` 原体迁成 runtime 中的私有同名加下划线函数，参数保持。**原样 `enumerate(report.faces, start=1)`**，仅对已接受物料连续编号；不能枚举 regions 再过滤产生编号空洞。radius×2、edge_support×100、z=distance_mm、raw=summary、时间戳均不变。
- [ ] `max_age_seconds/min_received_at` 保留现有窄接口形状；本地源仍是“读帧后、推理前”的时间戳，具体陈旧/停稳筛选仍在 servo_observation。不能声称获得了相机曝光时刻或用推理结束时间冒充采集时间。
- [ ] 默认 locate 不打印“读不到摄像头”或菜单；失败通过异常/返回值传达。旧回调提示迁到已有工具层，不向普通业务调用者增加交互。

迁入读取循环时把 `self.camera/locator/drain_frames` 分别改为 `self._camera/_locator/_drain_frames`；移除只负责打印的 `_say` 调用，计数/时序/返回值保持。不能机械粘贴后遗漏属性名导致运行时错误。

### B4. 关闭和部分初始化失败

- [ ] `Vision.close` 采用以下方法体；后续所有资源生命周期测试均以此所有权契约验收：

```python
def close(self):
    if self._closed:
        return
    self._closed = True
    camera, locator = self._camera, self._locator
    self._camera = self._locator = None
    errors = []
    if camera is not None:
        try:
            camera.release()
        except Exception as error:
            errors.append(("camera.release", error))
    if locator is not None:
        try:
            locator.close()
        except Exception as error:
            errors.append(("locator.close", error))
    if errors:
        details = "; ".join("{}: {}".format(stage, error)
                            for stage, error in errors)
        raise RuntimeError("Vision 清理失败：" + details) from errors[0][1]
```
- [ ] `USBCamera.initialize` 在局部 capture 创建后，用 try 包住 isOpened 和所有 set；成功才移交给 self.capture；失败 release 局部 capture 后原异常继续抛。release 也失败时遵循 §6 的主异常链规则。
- [ ] `YOLOv8Detector.__init__` 在原字段初始化之后，用 try 包住 `_load_engine`、尺寸检查、`_alloc_buffers`、`_warmup`；失败调用自身 close 后继续抛原异常。构造未返回的对象只能由构造器自己收尾。
- [ ] TensorRT close 先保存并清空各已取得引用；stream 同步失败后仍尝试其余资源。对 PyCUDA DeviceAllocation 调用 `free()`，对有 base.free 的 pagelocked host 数组释放其 base，未取得对象跳过，TRT/context/stream 去除最后持有引用；不双重释放。每项错误收集并在末尾报告，不能用原 synchronize 异常提前退出。用假 stream/假 allocation 验证中途失败和一次释放，不需要真实 CUDA。

本任务的资源验收：模型失败零相机、相机失败释放定位器、构造中途失败无泄漏；重复关闭零额外动作；清理失败后其余资源仍释放。测试新增到 `modules/vision/test/unit/test_runtime.py`，后端资源细节测试归 `test_vision.py`。

### B5. 工具覆盖与预览：冻结内部调用方式

`_override_settings` 使用 `dataclasses.replace` 生成新快照，不改变传入快照。按下面规则逐个写分支，不构造通用配置合并框架：

| 非 None 覆盖 | 写入位置与校验 |
|---|---|
| camera_id | copy camera_ids 后更新 current_platform 键；复用 config._camera_ids 的整数/非负检查规则 |
| model_path | Path 后按 .onnx/.engine 设置对应路径与 backend；其他后缀抛 ValueError；显式 CLI 相对路径保留原 cwd 解释，配置内相对路径仍按 PROJECT_ROOT |
| conf_thresh | model.conf_thresh；用 config._probability 保留 [0,1] 校验 |
| depth_range_mm | targets.working_distance_mm；用 config._optional_depth_range |
| material_bgr / material_palette | 对应 targets 字段；用 config._optional_bgr / _material_palette |
| colour_tolerance / top_diameter_mm | targets.color_tolerance / material_diameter_mm；用 config._positive_number |

focal_px 独立传给 `_open_config`，不写 calibration，也不填 `settings.focal_px`。数值 0 不能用 `or` 判断成未覆盖。这里只在已有 CLI 表达“未给覆盖”的地方用 None，不添加清空配置字段的新语义。

- [ ] 菜单18 `parse_args` 不读配置、不打开设备；全部覆盖默认值设 None，保留现有参数名称和 parser 类型。删 `configured_targets` 的坏配置默认继续分支。`run` 改为 `run(settings, *, focal_px=None) -> int`，由 main 加载快照、覆盖后传入；所有旧调用测试同批更新，不留旧签名兼容。
- [ ] `demo_material(model_path=None, camera_id=None, *, focal_px=None)` 的签名保留，内部加载快照并应用显式覆盖，使用同一 `_open_config`。不再调用 `resolve_target_settings` 的错误降级分支。
- [ ] 保留窗口与所有 watch / overlay / on_key 内容，只替换资源装配块：

```python
with Vision._open_config(settings, focal_px=focal_px) as vision:
    stats = run_preview(
        vision._camera, vision._locator.locate, vision._locator.draw,
        window=window, print_every=PRINT_EVERY, max_misses=MAX_MISSES,
        on_frame=watch, overlay=overlay, on_key=on_key,
    )
```

上例用于菜单18；demo 保留自己的现有参数和返回规则，不凭示例添加 demo 没有的 MAX_MISSES 常量。`run_preview` 仍执行原两线程方案；不能把回调换成 `vision.locate`，不能额外打开相机。工具仍在 finally 执行 `close_previews`。掉线返回码、空画面提醒、按 p 打印和 q 退出保持。

- [ ] `drive/material_locate.py` 的 --camera 分支复用相同快照/会话，--image 分支只使用 `_build_locator` 及图片，关闭定位器但不创建相机。`material_yolo / yolov8 / hough_lock` 的专用 CLI 仍用各自 detect/圆检测回调，不错误套入物料报告会话；只迁移其内部工具 import。普通 YOLO 相机 640×480 保持原值。

### B6. 同批退出 camera_source 的最小消费者迁移

本任务删除生产 camera_source 时，立即完成以下修改，不能拖到 D 后让 B 的测试因失效 import 无法运行：

- [ ] 菜单21删除 CameraFrameSource import；先仅迁移本地 `_open_source` 的创建，采用 §7.2 第3项签名及快照参数，main 在原创建 source 的位置之前读取本地快照，TCP 不读。普通动作移入 Grasp 的分流在 D 做；此处不提前引入 Grasp。
- [ ] `test_camera_source.py` 按 §7.3 改测 Vision 和 runtime 私有转换，保留该测试文件位置；菜单21现有测试同步调整 source 的新参数。
- [ ] 全仓检索 CameraFrameSource/camera_source 的 Python 消费者，全部改到实际定义后删除生产文件。此时原抓取自由函数仍可使用 Vision 的结构接口，不需要兼容源。

### B7. 最小可直接采用的会话测试

以下测试放进新 `test_runtime.py`，以实际参数为断言；另补 B1/B3/B4 的失败测试，不能只保留 happy path。

```python
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from modules.vision import Vision


class VisionSessionTests(unittest.TestCase):
    def test_one_camera_one_locator_for_two_calls(self):
        settings = SimpleNamespace(camera_id=Mock(return_value=7))
        camera = Mock()
        camera.read.return_value = object()
        report = SimpleNamespace(best=None)
        locator = Mock()
        locator.locate.return_value = report
        with patch("modules.vision.runtime._load_settings", return_value=settings) as load:
            with patch("modules.vision.runtime._build_locator", return_value=locator) as build:
                with patch("modules.vision.runtime._new_camera", return_value=camera) as new:
                    with Vision.open(config_path="custom.json") as vision:
                        self.assertIs(vision.locate(), report)
                        self.assertIs(vision.locate(), report)
                    vision.close()
        load.assert_called_once_with("custom.json", None)
        build.assert_called_once_with(settings, focal_px=None)
        new.assert_called_once_with(7)
        camera.initialize.assert_called_once_with()
        camera.release.assert_called_once_with()
        locator.close.assert_called_once_with()
        self.assertEqual(locator.locate.call_count, 2)
        with self.assertRaises(RuntimeError):
            vision.locate()
```

本任务验证命令，预期全部通过、无旧出口 import、无 runpy 预加载警告：

```powershell
python -m unittest modules.vision.test.unit.test_runtime modules.vision.test.unit.test_model_config modules.vision.test.unit.test_material_vision modules.vision.test.unit.test_material_camera_paths modules.vision.test.unit.test_material_locate modules.vision.test.unit.test_vision
python -m unittest modules.grasp.test.unit.test_camera_source modules.grasp.test.unit.test_align_xy_manual
python -m unittest tests.test_vision_type_exports tests.test_entrypoints tests.test_discovery_coverage tests.test_project_paths
```

## 5. 任务 C：拆出抓取基础职责并实现 Grasp（与 D 同批闭合）

**文件：**§2.2；`modules/grasp/test/unit/test_eye_in_hand.py`、新增 `test_grasp_session.py`，以及 §7.2 调用者。**C+D 是一个不可拆分的交付批次**：先移动唯一类型和原语、写会话，紧接着迁移 D 的菜单和测试，再统一验证。删除旧自由函数时同步改菜单 import/main/run_pick_ready，不能提交一个菜单无法导入的中间状态；不为中间状态保留兼容函数。

### C1. 原样迁移与配置读取

- [ ] 按 §2.2 迁移定义和内部依赖。旧 `_self_test` 留在 eye_in_hand 并改用真实定义，不删除测试场景或真实 CLI。
- [ ] Grasp 构造函数固定为 `__init__(self, *, config, selector, target, jacobian, lift_scale, on_message=None)`。只保存 `_config, _selector, _target, _jacobian, _lift_scale, _say`，初始化 `_arm=None, _source=None, _closed=False`；不读文件、不碰设备。None 回调取空函数，正常公开动作默认安静。
- [ ] `Grasp.open` 的五种路径各自转 Path，None 才取对应模块的现有 DEFAULT_CONFIG_PATH：servo_model、vision.config、arm.config、motor.config。不能把五个路径混用。
- [ ] 沿用五个现有读取器，从**同一指定的抓取文件**加载 config、selector、target、jacobian、lift_scale，构造后只使用这些值；不要在动作方法再次读文件。首批不强制合并现有五次解析为新配置框架。arm 使用 `load_arm_config(config_path=...)`；motor 使用 `load_bus_config(config_path=...)`，不能误把路径作为它的第一个 platform 位置参数。
- [ ] 视觉快照也在打开任何设备前用 `runtime._load_settings(vision_config_path, vision_calibration_path)` 读好。普通 open 传 `source_factory=lambda: Vision._open_config(vision_settings)` 给下述唯一内部创建路径。

公开 open 的完整装配体如下；配置加载器保留 §2.2 指定的真实定义，不重新实现解析：

```python
@classmethod
def open(cls, *, config_path=None, vision_config_path=None,
         vision_calibration_path=None, arm_config_path=None,
         motor_config_path=None):
    from pathlib import Path
    from modules.arm.config import DEFAULT_CONFIG_PATH as arm_default
    from modules.arm.config import load_arm_config
    from modules.motor.config import DEFAULT_CONFIG_PATH as motor_default
    from modules.motor.config import load_bus_config
    from modules.vision.runtime import Vision, _load_settings
    from .drive.servo_model import DEFAULT_CONFIG_PATH
    from .drive.servo_model import load_servo_config, load_object_selector
    from .drive.servo_model import load_target, load_jacobian, load_lift_scale

    grasp_path = DEFAULT_CONFIG_PATH if config_path is None else Path(config_path)
    arm_path = arm_default if arm_config_path is None else Path(arm_config_path)
    motor_path = motor_default if motor_config_path is None else Path(motor_config_path)
    config = load_servo_config(grasp_path)
    selector = load_object_selector(grasp_path)
    target = load_target(grasp_path)
    jacobian = load_jacobian(grasp_path)
    lift_scale = load_lift_scale(grasp_path)
    arm_config = load_arm_config(config_path=arm_path)
    port = load_bus_config(config_path=motor_path).port
    vision_settings = _load_settings(vision_config_path, vision_calibration_path)
    return cls._open_settings(
        config=config, selector=selector, target=target, jacobian=jacobian,
        lift_scale=lift_scale, arm_config=arm_config, port=port,
        source_factory=lambda: Vision._open_config(vision_settings),
    )
```

Grasp 构造函数也固定为以下实际代码；不要从构造函数提前打开设备：

```python
def __init__(self, *, config, selector, target, jacobian,
             lift_scale, on_message=None):
    self._config = config
    self._selector = selector
    self._target = target
    self._jacobian = jacobian
    self._lift_scale = lift_scale
    self._say = on_message if on_message is not None else (lambda message: None)
    self._arm = None
    self._source = None
    self._closed = False

def _require_open(self):
    if self._closed:
        raise RuntimeError("Grasp 已关闭")
```

### C2. 校验先于设备、创建与资源移交

在 `grasp/runtime.py` 定义 `_new_arm(port, arm_config)`，函数体内从 `modules.arm.drive.robot_arm` 导入 RobotArm 并返回 `RobotArm(port, config=arm_config)`。这是实际构造点；测试 patch 此处，不能 patch 一个执行时不查找的包别名。

```python
@classmethod
def _open_settings(cls, *, config, selector, target, jacobian,
                   lift_scale, arm_config, port, source_factory,
                   on_message=None):
    # 普通动作校验不能移动到创建 RobotArm 之后，其构造已打开串口。
    if target is None:
        raise ServoError("缺少 target；先运行现有设目标或示教工具")
    if jacobian is None:
        raise ServoError("缺少 jacobian；先运行现有标定工具")
    jacobian.pulses_for(0.0, 0.0)  # 复用原奇异判据，不新增阈值。
    session = cls(config=config, selector=selector, target=target,
                  jacobian=jacobian, lift_scale=lift_scale,
                  on_message=on_message)
    try:
        # factory 返回即转移资源所有权；返回前失败由 factory 自行收尾。
        session._source = source_factory()
        session._arm = _new_arm(port, arm_config)
        session._arm.initialize()
        session._arm.enable_all()
        return session
    except BaseException as primary:
        try:
            session.close()
        except Exception as cleanup:
            raise primary.with_traceback(primary.__traceback__) from cleanup
        raise
```

`_open_settings` 是 public open 与菜单普通动作的**唯一设备创建路径**。不新增 `allow_uncalibrated`、mode、源自动切换或“已有对象也许不归我管”参数。它取得的 source 都归会话关闭；TCP 与本地路线一致。lift_scale 可为 None，不能要求 z_target/平面标定/mm_per_motor_rev。

`RobotArm.__init__` 当前取得串口后才创建三个 Emm42Driver。为满足构造中途失败释放，允许在 `modules/arm/drive/robot_arm.py` **只改构造生命周期**：围住 `_motors` 创建，若失败且串口由该构造器创建则关闭后再抛；外部注入 serial_port 不由失败构造器擅自关闭。对应用例加在现有 `modules/arm/test/unit/test_robot_arm.py`。不改轴映射、指令或正常动作行为。

### C3. 动作方法的精确改法

```python
def align_xy(self):
    self._require_open()
    from modules.grasp.drive.eye_in_hand import hover_above_object, align_x_y

    if self._config.hover_z_mm is not None:
        hover_above_object(
            self._arm, self._source, self._selector, self._config,
            scale=self._lift_scale, on_message=self._say,
        )
    return align_x_y(
        self._arm, self._source, self._selector, self._target,
        self._jacobian, self._config,
        lift_scale=self._lift_scale, on_message=self._say,
    )
```

`align_and_descend` 不调用另一个旧同名自由函数。把当前自由函数体整体移入方法：

| 原局部输入 | 方法内对应状态 |
|---|---|
| arm / link | self._arm / self._source |
| selector / target / jacobian / config | self._selector / self._target / self._jacobian / self._config |
| lift_scale / say | self._lift_scale / self._say |
| z_pulses / retract_pulses | 保留为本次方法关键字参数 |

第一行 `_require_open()`，随后在方法体内从 `modules.grasp.drive.eye_in_hand` 导入 `hover_above_object / raise_before_rotating / align_x_y / descend_z / retract_extension`；其余 `time.monotonic`、四段耗时、日志与 GraspReadyResult 构造保留。具体顺序固定为：

```text
hover_above_object
→ raise_before_rotating
→ align_x_y(ensure_safe_height=False)
→ descend_z(pulses=z_pulses, scale=self._lift_scale)
→ retract_extension(pulses=retract_pulses)
```

这里不调用 `self.align_xy()`：它会重复悬停/安全检查，改变原主链。hover_pulses、raised_pulses 等字段不丢失。每阶段失败立即向外抛，不继续后续动作；不新增夹爪闭合、回零和自动重试。

### C4. 清理的准确边界

`Grasp.close` 先标记 closed、取出并清空资源引用，然后分四个独立 try：`arm.stop_all()`、`arm.disable_all()`、`arm.close()`、`source.close()`；资源不存在跳过。任何一步错误都进入清单，后续仍执行。末尾报错方式与 Vision 一致。构造 arm 失败时只关已取得 source；initialize 或部分 enable 失败时四项均尝试。重复 close 不再发 stop/disable。

可以直接采用下列方法体；这个局部循环只整理同一对象的收尾，不抽成全仓通用框架：

```python
def close(self):
    if self._closed:
        return
    self._closed = True
    arm, source = self._arm, self._source
    self._arm = self._source = None
    actions = []
    if arm is not None:
        actions.extend((("arm.stop_all", arm.stop_all),
                        ("arm.disable_all", arm.disable_all),
                        ("arm.close", arm.close)))
    if source is not None:
        actions.append(("source.close", source.close))
    errors = []
    for stage, action in actions:
        try:
            action()
        except Exception as error:
            errors.append((stage, error))
    if errors:
        details = "; ".join("{}: {}".format(stage, error)
                            for stage, error in errors)
        raise RuntimeError("Grasp 清理失败：" + details) from errors[0][1]
```

**限制需如实说明：**现有 `RobotArm.stop_all/disable_all` 内部按轴顺序执行，本计划保证这两项及后续串口/视觉收尾分别尝试，不借接口重构改成新的逐轴容错策略。

### C5. 可以直接采用的关键测试

新增 `modules/grasp/test/unit/test_grasp_session.py`：

```python
import unittest
from unittest.mock import Mock, patch
from modules.grasp import Grasp, ServoError
from modules.grasp.drive.servo_model import ServoConfig, ServoTarget, ServoJacobian
from modules.grasp.drive.servo_model import ObjectSelector


class GraspSessionTests(unittest.TestCase):
    def settings(self):
        return dict(config=ServoConfig(), selector=ObjectSelector(),
                    target=ServoTarget(0.0, 0.0),
                    jacobian=ServoJacobian(1.0, 0.0, 0.0, 1.0),
                    lift_scale=None, arm_config=object(), port="FAKE")

    def test_missing_target_opens_no_devices(self):
        values = self.settings()
        values["target"] = None
        source_factory = Mock()
        with patch("modules.grasp.runtime._new_arm") as new_arm:
            with self.assertRaises(ServoError):
                Grasp._open_settings(source_factory=source_factory, **values)
        source_factory.assert_not_called()
        new_arm.assert_not_called()

    def test_failed_stop_still_disables_closes_arm_and_source(self):
        arm, source = Mock(), Mock()
        with patch("modules.grasp.runtime._new_arm", return_value=arm):
            grasp = Grasp._open_settings(source_factory=lambda: source,
                                          **self.settings())
        arm.stop_all.side_effect = RuntimeError("stop failed")
        with self.assertRaises(RuntimeError):
            grasp.close()
        grasp.close()
        arm.stop_all.assert_called_once_with()
        arm.disable_all.assert_called_once_with()
        arm.close.assert_called_once_with()
        source.close.assert_called_once_with()
```

另以原 `test_eye_in_hand` 中的主链测试迁移为实例测试：用 list 记录 hover/raise/xy/descend/retract 的调用顺序；逐阶段 side_effect 报错，断言之后的函数未调用；断言 xy 的 ensure_safe_height=False；断言返回的全部耗时/hover 字段。普通 `align_xy` 分 hover 已配/未配两组，后者仍保留底层安全抬升检查。

以下命令在完成 D 的消费者迁移后与 D 一起运行，预期原算法预期与新增会话测试全部通过；不得在 C 后留下导入损坏的菜单再宣称本批完成：

```powershell
python -m unittest modules.grasp.test.unit.test_eye_in_hand modules.grasp.test.unit.test_grasp_session modules.arm.test.unit.test_robot_arm
```

## 6. 两类会话共用的异常语义

每个类自己实现 `_require_open / close / __enter__ / __exit__`，不新建通用生命周期框架。`close` 中错误清单的 RuntimeError 文案必须包含所有失败阶段和原错误文字，以第一项为 cause。

`__enter__` 检查未关闭再返回 self；两类的 `__exit__` 都使用以下规则。无业务错误则清理错误直接抛出；有业务错误且清理也失败，则原业务异常仍为主异常并保留 traceback，清理错误作为 cause 显示：

```python
def __exit__(self, exc_type, exc, traceback):
    try:
        self.close()
    except Exception as cleanup:
        if exc is not None:
            raise exc.with_traceback(traceback) from cleanup
        raise
    return False
```

不得沿用 `_exit_quietly` 的吞错。收尾清单只收普通 Exception，不把 KeyboardInterrupt/SystemExit 当作普通可忽略错误。创建阶段捕获 BaseException 只是为了中断时也收尾，然后原样再抛。

## 7. 任务 D：完成 C 的消费者和测试迁移（与 C 同批闭合）

### 7.1 视觉消费者，不漏包出口和文档示例

| 消费者 | 改法 |
|---|---|
| `vision/app/demo.py` | 物料预览按 B5；其余 demo 的相机/窗口/模型工具直接导入定义模块，不删除其他 demo |
| `vision/test/manual/material_locate.py` | B5 的新 run 签名；菜单18保留，所有覆盖保留 |
| `vision/test/manual/vision.py`、`obstacle_model.py` | 保留专用能力，改工具定义来源 |
| `vision/drive/hough_lock.py`、`material_yolo.py`、`material_locate.py`、`yolov8.py` | 相机分支和工厂 import 改真实归属；保留真实 CLI 与原回调 |
| `grasp/test/manual/grasp.py` | 仅更改使用中的视觉工具 import，不执行菜单19清理 |
| `tests/test_entrypoints.py`、`vision/test/unit/test_model_config.py` | default_model_path/require_model_path/load_model_settings 从 config 导入 |
| `tests/test_vision_type_exports.py` | 原测试依赖惰性表；改验证三个明确公开出口的对象身份与 __all__，并验证单独导入包不预加载 CLI/不开设备；不能为测试保留旧表 |
| `vision/test/unit/test_material_camera_paths.py` | 包出口断言改为新接口契约；patch 实际查找处（见下表），保留 camera/图片/退出码/空画面路径覆盖 |
| `vision/test/unit/test_material_vision.py` | 删除对 _configured_materials 的 patch，改成注入明确快照并检查类别/显式覆盖优先级 |
| `vision/drive/preview.py`、`stream.py`、包文档示例 | 同批更新已撤销的包导入示例 |

工具定义来源固定：相机/编号/空画面检查→`drive.camera`；窗口函数→`drive.preview`；run_preview→`drive.stream`；模型路径/设置→`config`；detector工厂→`detector_factory`；DistanceEstimator→`drive.distance`；默认类别→`drive.yolov8`。

| 被测行为 | patch 查找位置 |
|---|---|
| Vision 配置/装配次数 | `modules.vision.runtime._load_settings / _build_locator / _new_camera` |
| _build_locator 内部后端装配 | `modules.vision.detector_factory.create_configured_detector`、`modules.vision.drive.material_yolo.MaterialYOLO`、`modules.vision.drive.material_locate.MaterialLocator`，因函数内从定义处导入 |
| 工具窗口与预览 | 若在工具顶层导入，patch 对应工具模块的 open_preview/run_preview；若方法内导入，patch 真实定义模块；不得继续 patch modules.vision.USBCamera 等已删除出口 |
| Grasp 构造与主链 | 构造 patch `modules.grasp.runtime._new_arm`；动作方法内导入的算法 patch `modules.grasp.drive.eye_in_hand` 的实际定义 |
| 观测、脉冲助手 | 对应 `servo_observation / servo_motion` 中被执行函数实际读取的名字，不 patch 旧 eye_in_hand 路径 |

### 7.2 菜单21：先区分前置条件，再进入会话

**修改：**`modules/grasp/test/manual/align_xy.py`、`modules/grasp/test/unit/test_align_xy_manual.py`。

1. 顶层配置/值类型从 servo_model 导入；FrameSource/measure_offset/select_object 从 servo_observation；_move_lift 从 servo_motion；实际控制函数从 eye_in_hand。不要从包顶层导入内部工具。
2. 保留现有参数和模式。main 先加载抓取配置，应用 `--hover-z` 的 dataclasses.replace；仅非 TCP 路线读取视觉快照后应用 `--camera`。临时覆盖不能写文件，也不能先 public open 再关掉重开带覆盖的设备。
3. `_open_source(arguments)` 的 TCP 分支保留显式选择、端口、start 和失败信息；本地分支替换为接收已加载快照的工厂：定义改成 `_open_source(arguments, *, vision_settings)`，返回 `Vision._open_config(vision_settings)`。TCP 分支不得加载或验证本地模型、摄像头与视觉配置；只在非 TCP 路线读取 vision_settings，TCP 传 None。
4. 正常 **默认 align、--align-and-descend** 分支：不提前调用 _open_source 或 RobotArm；先读好 arm_config/port，再统一调用 `Grasp._open_settings(..., source_factory=lambda: _open_source(arguments, vision_settings=vision_settings), on_message=_say)`。with 拥有全部资源，工具分支不能再重复关一次。
5. `run_align` 签名改为 `run_align(*, grasp: Grasp) -> int`，保留原提示/结果打印，以 `grasp.align_xy()` 执行动作。提示需取本会话已有 `_selector/_target/_jacobian/_config`，只允许本仓工具内使用，不增加面向上层的十几个属性。`run_pick_ready` 改为 `run_pick_ready(*, grasp: Grasp, z_pulses=None, retract_pulses=None) -> int`，调用实例方法并保留“下一步合上夹爪”提示。二者只捕获并显示原 ServoError，main 的 with 负责收尾。
6. 其余工具保留原低层流程及前置条件，按下表，不强塞进普通 Grasp.open；source 由该工具分支创建/关闭，机械臂仍按原工具初始化/使能/收尾。可以复用 C4 的“每项独立尝试”规则，但不得添加允许缺标定的公开会话模式。

| 模式 | 本批调用方式与验收 |
|---|---|
| watch / status / set-target / target-centred | 原工具函数 + 所选 source；不创建 RobotArm；缺 target/J 时仍可运行；保留原交互 |
| calibrate / teach | 原 probe/示教流程；不以已有 target/J 作为启动前提；保留写标定用途 |
| follow | 保留原连续跟随算法与循环，不循环调用含悬停的 align_xy；保留它原有 target/J 闸门与动作行为 |
| align-z / align-z-visual / raise | 原专用函数与参数；不新增 target/J 条件，不改深度/升降保护；显式重新标定仍只在原对应模式发生 |
| TCP 的普通 align / align-and-descend | 与本地进入同一个 _open_settings，但 source_factory 返回 CameraLink；不创建 Vision，不做本地模型检查，不自动切换 |

若保留工具分支的 finally，stop/disable/arm.close/source.close 各自尝试，不让第一项异常跳过其他项；保留原业务异常并显示收尾失败。不要把全部分支共同提前开 source 后又在正常 Grasp 分支重开。

**TCP 部分启动失败也必须闭合所有权。**为满足 C2 的 source_factory 契约，额外仅修改 `modules/camera_link/drive/link.py` 的 start/stop 生命周期和现有 `modules/camera_link/test/unit/test_camera_link.py`；不改协议、队列、取帧或接收算法。

- `CameraLink.start` 的 setsockopt、bind、listen、getsockname、构造/启动线程全部进入同一个 try。保留原 bind 失败转换成 CameraLinkError 的消息，但删该分支内单独 close，统一由失败收尾负责；不可只包 bind。完整方法体如下，依赖仍为 link.py 已有的 socket/threading/CameraLinkError：

```python
def start(self):
    if self._thread is not None:
        return self
    try:
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self._server.bind((self.host, self._requested_port))
        except OSError as error:
            raise CameraLinkError(
                f"无法监听 {self.host}:{self._requested_port}：{error}；"
                "端口被占用或没有权限"
            ) from error
        self._server.listen(1)
        self.port = self._server.getsockname()[1]
        self._stop.clear()
        self._thread = threading.Thread(target=self._serve,
                                        name="camera-link", daemon=True)
        self._thread.start()
        return self
    except BaseException as primary:
        try:
            self.stop()
        except Exception as cleanup:
            raise primary.with_traceback(primary.__traceback__) from cleanup
        raise
```

- `stop` 先置 stop 事件，取出 client/server/thread 并把实例三项置 None，再分别尝试关两个 socket；shutdown 对未连接 socket 的既有 OSError 可忽略，**close 的错误必须收集**，不能静默吞掉。join 也独立尝试，随后统一抛包含各错误阶段的 CameraLinkError（cause 为第一项）。重复 stop 不重复 close。
- 启动线程失败时，线程对象可能已赋给 self._thread 但尚未启动。join 前必须使用下面条件，避免 `cannot join thread before it is started` 覆盖原启动错误；已正常启动/退出线程保留原 5 秒 join 超时。

```python
if (thread is not None and thread.ident is not None
        and thread is not threading.current_thread()):
    thread.join(timeout=5.0)
```

- 在现有 `CameraLinkLifecycleTests` 添加故障注入测试：假 socket 的 setsockopt/listen/getsockname 分别失败时，server.close 恰好一次，_server/_client/_thread 均为 None；假线程 start 失败、ident=None 时不调用 join，保留原 start 异常且 socket 释放；socket.close 失败时仍尝试另一个 socket 与已启动线程 join，并显示清理失败。使用 unittest.mock，不绑定真实端口验证这些失败场景。

### 7.3 现有测试文件的具体处置

- `modules/grasp/test/unit/test_camera_source.py` **保留文件位置和命令名**，改测 runtime 的私有转换/取帧与 Vision 所有权；不创建 test 转发文件。原转换与计数、时间戳用例保留。旧“注入的资源不拥有”用例改成“会话对所取得资源始终拥有”；测试可在构造接缝注入假设备，不能将借用模式暴露成新业务接口。
- `test_eye_in_hand.py` 的纯值/加载器/助手 import 改为 §2.2 归属，原主链测试改为 Grasp 方法，函数内 patch 改为实际查找处。保留所有安全/预算预期。
- `test_align_xy_manual.py` 不再 patch 菜单模块的旧 `align_and_descend`；向新 run_pick_ready 传入假 grasp，断言其 `.align_and_descend(z_pulses=..., retract_pulses=...)` 参数及原返回码/打印。另验证 main 的默认分支只走一次会话装配，status/标定不被正常会话拦截。
- `tests/test_main.py` 以及其他导入消费者按实际签名修改，保留现有 main 场景；不用改 main 的菜单/业务路由来适配测试。

本任务验证命令，预期全部通过，菜单号/入口发现与原行为一致：

```powershell
python -m unittest modules.grasp.test.unit.test_camera_source modules.grasp.test.unit.test_align_xy_manual modules.grasp.test.unit.test_eye_in_hand modules.grasp.test.unit.test_grasp_session
python -m unittest modules.camera_link.test.unit.test_camera_link
python -m unittest tests.test_main tests.test_entrypoints tests.test_discovery_coverage tests.test_project_paths tests.test_vision_type_exports
```

## 8. 任务 E：闭环检查与交付证据

### 8.1 全链路验收矩阵

| 输入/故障 | 可验证结果 |
|---|---|
| 单独 import 两个包 | 不读取配置、不构造相机、串口或模型，不预加载作为 python -m 的 CLI 模块 |
| 自定义视觉/抓取/arm/motor/标定路径 | 对应读取器都收到指定路径，无默认文件回读；正确模型、camera、类别、阈值与 targets 抵达实际实现 |
| 打开后更改配置文件 | 本会话结果与动作继续使用原快照，只有新会话读取更新 |
| Vision 连续 locate 两次 | detector/camera 只创建一次；每次单独读帧、定位一次；空报告和读失败语义不同 |
| 缺 target/J、奇异 J、无效配置 | 在 source_factory 和 RobotArm 构造前失败；零运动帧 |
| 模型构造失败 / camera 初始化失败 | 零机械臂；所有已取得视觉资源释放；原失败原因保留 |
| arm 构造 / initialize / enable 中途失败 | 视觉源关闭；构造器自身负责未移交资源；已有 arm 走完整收尾尝试 |
| align_xy | 保留原可选悬停、底层安全抬升、新帧/停稳和预算，不下降/后缩 |
| align_and_descend | hover→raise→xy→descend→retract，xy 跳过的仅是已完成的重复安全检查，结果字段与计时完整 |
| 任一动作失败 / 平面不准 | 后续动作不执行；特别是平面不准零下降，不自动重试 |
| 清理第一项失败 | 后续项仍执行，错误包含所有失败阶段；业务异常保留为主异常；重复 close 零额外资源动作 |
| --tcp / 缺标定的 status/calibrate/teach | 原场景仍可用，无隐式本地模型要求，无“缺标定不能运行标定”的循环 |
| TCP listen/线程启动失败 | source 尚未交给 Grasp 时由 CameraLink 自行清理；未启动线程不 join，不掩盖原失败 |
| 预览、图片、普通 YOLO CLI | 物料预览保留两线程；图片零相机；专用算法与原画幅不被物料会话替代 |

### 8.2 验证命令与交付

- [ ] 检索旧入口剩余消费者：CameraFrameSource/camera_source 查询预期**没有活动代码命中**；历史文档可以注明旧路径。align_and_descend 查询预期只保留一个实例方法。若 rg 不可用，用 Select-String，不安装工具。

```powershell
Get-Command rg
rg -n 'grasp\.drive\.camera_source|CameraFrameSource' modules tests main.py -g '*.py'
rg -n 'def align_and_descend' modules/grasp/runtime.py modules/grasp/drive/eye_in_hand.py
```

`def align_and_descend` 预期仅命中 Grasp 的实例方法，不能同时存在旧自由函数。另用 AST 核对每个 ImportFrom，不能只靠单行搜索证明跨行包出口 import 全部迁移。检查 vision 没有 grasp 依赖，model 没有 controller 依赖，所有移动类型只有一份定义。

- [ ] 运行最终完整离线验证与 Python 3.8 守卫，预期零失败/错误；记录实际测试数及 skip 原因。缺真实 3.8 解释器则写明“Python 3.8 运行兼容未验证”，不能把静态检查说成板端实测。

```powershell
python -c "import test; test.run_offline_tests()"
python -m unittest tests.test_python38_compatibility
git diff --check
git diff --stat
git status --short
```

- [ ] 更新 README 中受影响的调用示例、`modules/grasp/docs/抓取坐标链路.md` 的入口说明，以及本次两份设计稿的实施状态；只改实际受影响段落，保留用户已有内容。
- [ ] 交付：最终调用示例、文件/符号迁移表、测试命令与原始结果、未完成的真机验收边界。不能只说“已拆文件、测试通过”。不自动提交其他用户修改、不 push。

每个逻辑任务独立检查差异；如需回滚，仅逆转该任务新增修改，不能对用户原有修改执行 reset/checkout/clean。暂态集成失败必须在该任务闭合前修正，不能用兼容入口维持双实现。

## 9. 给执行 agent 的交接提示词

下面内容可连同三份文档直接交给执行 agent；**本轮没有发送或执行这个指令**：

> 请执行 `docs/plans/2026-10-15-vision-grasp执行方案.md`，先读它链接的总方案和接口专项，再核对当前源码与工作区差异。按 A→B→(C+D)→E 顺序推进；C+D 是一个不可拆分批次，删除旧自由函数与菜单/测试迁移同时闭合。所有消费者和测试以实际定义为准同步迁移，完成一个逻辑任务再报告证据。保留已有用户修改；不改现场配置、不安装依赖、不执行真机、不做历史菜单清理。真实实现写进 Vision/Grasp，不留旧入口兼容壳，不重写伺服算法。接口签名、配置传递、资源所有权、标定工具前置条件、两线程预览按本文执行。遇到与本文业务契约冲突或外部条件阻塞时，报告准确文件/符号/错误，不自行猜替代业务规则。每次报告列出完成项、实际验证及下一项；最终以验收矩阵和测试输出证明完成。
