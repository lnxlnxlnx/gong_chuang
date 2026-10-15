# vision + grasp 调用接口与内部重构专项

> 配套：[视觉与抓取优先的重构方案](可评审性优化方案.md)。本文定义两个模块如何被调用，以及内部实现必须满足的条件。
> 状态：**方案稿，示例接口尚未实现**。2026-10-15 静态核对当前工作区，未执行重构、套件或硬件。
> 用户已明确：优先改善视觉、抓取调用方式；采用清晰的类接口；**对象负责设备生命周期，上层只传配置并调用方法**。
> [逐文件执行说明](plans/2026-10-15-vision-grasp执行方案.md) 进一步冻结文件/符号归属、内部创建接缝、CLI 覆盖、迁移顺序和具体测试；交给执行 agent 时一起阅读，当前仍仅修改文档。

---

## 1. 从调用者的任务确定接口

上层主要需要两种能力：获取一次结构化物料定位结果；执行一次已经标定好的对准或对准下降流程。

目前上层要知道相机与定位器如何装配、多个配置怎么组合、哪些标定必需、机械臂何时使能，以及异常时如何停止和关闭。把自由函数改成静态方法但仍要求传入所有这些对象，并不能解决调用困难。

本方案选择：**类方法创建会话，实例方法执行操作，纯计算保留为值对象方法或静态方法**。相机、模型、串口和标定快照不存入全局变量或单例。

### 1.1 当前主链

```text
main.py → vision.app.demo → 物料定位与预览
菜单 18 → vision.test.manual.material_locate → 相机 + MaterialLocator
菜单 21 → grasp.test.manual.align_xy
           ├─ 读取 selector / target / jacobian / config / lift_scale
           ├─ CameraFrameSource → 相机 + MaterialLocator → MaterialSighting
           └─ RobotArm → eye_in_hand 自由函数 → 动作与测量
显式 --tcp → CameraLink → 同一窄观测接口
```

当前 `CameraFrameSource` 实际做了视觉会话的工作，却放在 grasp；正常抓取的装配和收尾又藏在 manual 脚本里。需要迁移这些真实职责，而不是再加一个只转发调用的门面。

## 2. 视觉公开接口

### 2.1 首批公开能力

| 接口 | 调用者需要知道的内容 |
|---|---|
| `Vision.open(*, config_path=..., calibration_path=...) -> Vision` | 读取指定配置并打开相机/模型，返回拥有资源的就绪实例；路径默认取现有模块常量 |
| `vision.locate() -> MaterialReport` | 采集并定位一次；返回现有报告类型，`best`、`faces`、坐标和分类可直接读取 |
| `vision.close()`、上下文管理 | 关闭本实例拥有的设备；可重复关闭；关闭后操作明确失败，不自动重开 |

不把相机、定位器、阈值、颜色表和模型参数重复铺到每次 `locate()` 的参数中。使用配置文件表达这些既有参数。

```python
from modules.vision import Vision

with Vision.open(config_path="modules/vision/config.json") as vision:
    report = vision.locate()
    for material in report.faces:
        print(material.name, material.x_mm, material.y_mm, material.z_mm)
```

没有通过校验的物料时返回正常的空结果，`report.best is None`；无法读取画面、配置无效、模型或设备不可用属于失败，不能伪装成正常“无物料”。公开 `locate()` 的失败异常须明确，复用适用的现有 `ValueError`、`FileNotFoundError`、设备 `RuntimeError`，不统一吞掉后返回空结果。

### 2.2 对象实际承担的职责

- 读取指定视觉配置，装配相机和 `MaterialLocator`，拥有并复用模型；一帧操作不能重新加载模型或打开相机。
- 迁入当前 `CameraFrameSource` 的本地取帧、丢弃缓冲帧、时间戳与观测转换职责；Vision 不导入 grasp。
- 内部保留 `frame(timeout, max_age_seconds, min_received_at)` 形式的窄观测能力，供抓取和测试使用；它沿用现有“未读到帧为 None”的观测契约，与公开 `locate()` 的失败语义分别写清。
- `locate()` 与内部 `frame()` 共用一次取帧/定位实现；一次调用只处理该次画面，不重复推理、不拿上次缓存结果冒充新观测。
- 不自动打开预览窗口，不打印控制台菜单，不调用 `input()`；`draw()`、预览与图片 CLI 保留为独立工具能力。

已有 `MaterialReport`、`RegionReport` 和 `MaterialSighting` 继续表达结果，不因新增入口再造一套 `VisionResult` / `GraspVisionResult`。

菜单 18 与 demo 已有 `--model`、`--camera`、`--conf`、`--focal`、`--depth`、颜色及直径等本次覆盖。工具层先读取配置快照，再应用这些明确覆盖，随后进入 Vision 同一真实创建路径；不把参数铺到每次 `locate()`，不写回文件、不再开一套相机/模型。原有端口、悬停高度和动作脉冲的 CLI 覆盖遵循相同规则。

实时预览仍由 vision 内部工具复用本会话的相机和定位器，沿用 `run_preview(camera, locate, draw)` 的两线程与原图检测、绘制、统计和按键回调。需要显示的图像与结果保持原有对应关系，不用同步 `vision.locate()` 阻塞显示循环，不为预览扩大普通定位接口；离线图片工具仍只处理给定图片。

### 2.3 配置必须传到真实实现

只在入口增加 `config_path` 参数不够。当前存在这些默认配置回读：

| 位置 | 事实 | 迁移要求 |
|---|---|---|
| `camera_source.py:145-159` | 配置、模型路径、相机编号分别通过默认入口读取 | 从同一份指定配置快照解析并传递 |
| `material_yolo.py:110-119` | 首次取 detector 时经包工厂读取默认模型设置 | 真实 detector 接收本会话模型路径、类别、输入尺寸和阈值 |
| `material_yolo.py:154` | 物料类别判断还会读取默认配置 | 使用会话装配时确定的类别，不逐帧读默认文件 |
| `vision/config.py:69-72` | 模型相对路径按 `PROJECT_ROOT` 解析 | 保持该规则，不静默改成配置文件目录或当前目录 |
| `load_vision_config()` | 视觉标定有独立 `calibration_path` | 公开创建参数明确标定路径；全链不得换回默认标定文件 |

验收：用与默认文件明显不同的自定义模型、相机编号、类别、阈值及 targets 配置，检查真实构造参数与运行结果；实例创建后改动配置文件，不影响本次快照，关闭后重新创建才加载新值。保留配置内现有 ONNX/engine 平台选择，不新增自动替换模型的退路。

按用户要求的 failfast，迁移后的 demo/菜单18不再捕获坏配置后按默认 targets 继续，MaterialYOLO 不再捕获坏类别配置后猜物料表；配置缺块时仍保留加载器本身明确规定的合法默认值。现有定位算法的策略顺序另按原逻辑保留，不在本次改写。

**焦距特别约束**：当前 `CameraFrameSource` 创建定位器时没有传 `settings.focal_px`，走既有粗焦距策略。将其改成使用标定焦距会改变毫米坐标和控制结果，不能冒充接口搬移。首批维持当前主线实际生效的焦距策略并明确说明；是否统一视觉工具与抓取的标定焦距，作为独立行为决策和坐标回归验收。

### 2.4 打开与关闭的契约

`MaterialLocator.__enter__()` 当前只返回自身，不能据此证明模型已就绪。`Vision.open()` 返回前必须真正加载并验证该会话的模型，失败时释放已经获取的资源，不等机械臂使能后才发现模型不可用。

相机和模型只由 Vision 关闭；初始化过程中途失败也要释放已成功创建的部分。关闭过程必须尝试释放全部资源，原始业务异常可追溯，清理失败可见，不沿用静默吞掉全部清理异常的做法。

## 3. 抓取公开接口

### 3.1 首批公开能力

| 接口 | 契约 |
|---|---|
| `Grasp.open(*, config_path=..., vision_config_path=..., vision_calibration_path=..., arm_config_path=..., motor_config_path=...) -> Grasp` | 从现有配置装配并拥有机械臂与一个 Vision；所有路径可明确指定 |
| `grasp.align_xy() -> ServoResult` | 对应现有普通 `run_align`：按配置先悬停，再检查安全高度并完成平面对准 |
| `grasp.align_and_descend(*, z_pulses=None, retract_pulses=None) -> GraspReadyResult` | 悬停、安全抬升、平面对准、固定脉冲下降和后缩；覆盖值仅作用于本次，不写回配置 |
| `grasp.close()`、上下文管理 | 按机械臂和视觉资源的所有权完整收尾；关闭后不再允许动作 |

`config_path` 指向现有 `eye_in_hand.json`；机械臂参数与总线端口仍分别来自 arm 和 motor 现有配置，不把它们复制成另一份配置。

```python
from modules.grasp import Grasp

with Grasp.open(
    config_path="modules/grasp/eye_in_hand.json",
    vision_config_path="modules/vision/config.json",
) as grasp:
    result = grasp.align_xy()
```

另一次调用可选择 `align_and_descend()` 完成已有主线。不需要上层传入 `arm/source/selector/target/jacobian/lift_scale`，也不需要分别调用 initialize、enable、stop、disable、close。

### 3.2 必须在开机械臂前完成的校验

- 读取并校验所有相关配置、物料选择条件、目标与雅可比；目标/雅可比缺失或雅可比奇异时明确拒绝正常动作会话。
- `lift_scale` 保持可选；正常脉冲主线不新增 `z_target`、`mm_per_motor_rev`、行程或平面标定前置条件。
- 视觉模型和相机就绪，才创建并使能机械臂。注意 `RobotArm.__init__()` 已经打开串口，不能把检查放在 `initialize()` 之前就宣称“开硬件之前”。
- 配置路径、标定和模型错误应保留具体原因，不从默认文件补读缺项，不自动重标，不尝试另一条硬件路线。

读取配置和打开设备本身不执行对准、探针、回零、升降或下降；实际动作只在对应动作方法被调用时发生。使能与停止/失能属于明确说明的会话生命周期。

### 3.3 不把全部现场工具强塞进普通动作会话

当前菜单 21 的 `--status`、观察、设目标、示教、标定、视觉 z 等模式有不同前置条件。需要目标和雅可比的普通 `Grasp.open()` 不能成为这些模式的统一入口，否则会出现“缺标定所以不能运行标定”的循环。

首批只公开正常的 `align_xy()`、`align_and_descend()`。已有状态/标定/示教工具保留其用途，在内部复用配置、观测和动作能力；只读模式不得因此打开或使能机械臂。不新建通用模式枚举或一串 `allow_uncalibrated` 参数绕过校验。

TCP 路线保留显式选择和现有消息契约，内部通过窄观测接口接入；本机 USB 是首批示例的默认路线，不新增“USB失败自动TCP”等策略。是否把 TCP 装配也作为公开工厂参数，应按现有使用场景另行确定，不能删除现有 CLI 路线。

内部 source 工厂返回前的失败由工厂自身释放：本机依赖 Vision 的部分初始化收尾，TCP 则补齐 CameraLink.start 在 listen/线程启动失败时的资源释放，且不 join 尚未启动的线程；不改变消息协议或正常接收行为。

### 3.4 资源所有权与失败处理

```text
Grasp 实例
  ├─ 拥有一个机械臂连接
  ├─ 拥有一个 Vision 会话（本地路线）
  └─ 持有本次配置、目标、雅可比、选择条件和可选升降尺度
Vision 实例
  ├─ 拥有相机
  └─ 拥有定位器与模型
```

上层单独创建的 Vision 与 Grasp 默认内部创建的 Vision，是两种独立使用方式；示例不同时打开同一相机。首批不新增“共享全局相机”或自动借用最近视觉会话。

正常退出、打开一半失败和动作异常都要清理。stop 或 disable 报错不能阻断串口与视觉关闭；重复关闭不重复发送动作。关闭后的实例再次操作立即失败，不自动重连。

## 4. 内部结构：先收敛接口，再按真实职责整理

### 4.1 目标归属

以下是最小职责划分，不按行数或文件数量继续细拆：

| 位置 | 实际职责 |
|---|---|
| `vision/runtime.py` | `Vision` 的真实实现，吸收现有本地相机会话与配置装配 |
| `vision/drive/material_locate.py` 等已有算法文件 | 继续承担帧级定位、颜色、圆检测、质量、分类和针孔计算 |
| `grasp/runtime.py` | `Grasp` 的真实会话生命周期和主编排；显式动作方法内才加载控制算法，避免包导入预加载自测 CLI |
| `grasp/drive/eye_in_hand.py` | 保留实际闭环/探针/升降算法及原自测 CLI；原主编排体移入 Grasp，不留同名自由函数转发 |
| `grasp/drive/servo_model.py` | 配置、数值模型、现有结果对象、`ServoError` 及配置读写；共用类型单一定义 |
| `grasp/drive/servo_observation.py` | `FrameSource`、物料选择、停稳与观测判据；不发电机动作 |
| `grasp/drive/servo_motion.py` | 实际脉冲发送与等待；不读视觉、不做策略选择 |

原 `camera_source.py` 的生产职责迁入视觉真实实现后，旧装配和转换入口同步退出，不留兼容文件。其现有转换、时序和资源测试随真实归属迁移；测试数量不作为保留旧路径的理由。

测试文件 `grasp/test/unit/test_camera_source.py` 保留原位置，改测 vision 实现，避免验收命令漂移。控制器中的底层闭环/探针继续是现场工具使用的真实内部算法；主编排 `align_and_descend` 移为 Grasp 实例方法，旧同名自由函数删除。`align_xy` 含悬停和原安全检查，不做旧编排函数的转发壳。

探针、安全抬升、悬停和闭环之间有真实依赖，不立即拆成相互调用的“平面/z/标定”文件，也不抽通用控制循环。确实与实例无关的数值操作可保留静态方法；已有 `ServoJacobian.pulses_for()` 和 `LiftScale.pulses_for()` 依赖各自数据，保留实例方法。

### 4.2 依赖方向

```text
应用 / 菜单 → modules.vision.Vision / modules.grasp.Grasp
Grasp → Vision + servo_model + eye_in_hand（显式动作时加载）
eye_in_hand → servo_model + servo_observation + servo_motion
servo_observation → servo_model + vision 的观测类型
servo_motion → servo_model + 现有机械臂/电机能力
Vision → vision 配置、观测类型与定位算法
vision 不依赖 grasp；model 不依赖控制器；observation 与 motion 互不依赖
```

`Vision` 不导入 grasp 的 `FrameSource` 或 `ServoError`；它通过已有形状的 `frame()` 提供观测，结构接口在抓取侧定义。

每个顶层符号、结果对象、常量、默认路径和私有助手必须有唯一归属。模块搬移后配置仍锚定原路径；不能用循环导入、动态符号表或局部 import 来掩盖错误的依赖方向。

### 4.3 包入口与导入副作用

保留 `from modules.vision import Vision`、`from modules.grasp import Grasp` 这种稳定入口。包内只显式导出少量本轮定义的公开类及必要结果/异常，不维护旧符号的大映射表。

- `Vision` 的实现文件不能在包导入阶段预加载仍作为 `python -m` CLI 的驱动模块。
- `Grasp` 定义在真实 runtime 模块，包导入不预加载 eye_in_hand 自测 CLI；运行时单向加载控制算法，不从算法反向导入会话。
- 纯结果类型若需要独立导入，可迁到不依赖 CLI 的结果模块，仍然只定义一份，不复制 DTO。
- 配置读取、模型导入和设备创建发生在显式 `open()`；类型注解不能迫使包初始化打开或预加载运行后端。
- 运行时内部直接依赖实际定义处，不反向从包出口拿自己的内部符号。
- 不把普通显式导出误认为旧兼容壳：公开类必须拥有真实实现；旧公开函数及其消费者同批迁移。

## 5. 结果与动作语义

### 5.1 观测数据保留可验证含义

| 含义 | 当前定义与要求 |
|---|---|
| 深度 | `RegionReport.z_mm` / `distance_mm` 为光轴深度，转换后对应 `MaterialSighting.z_mm` |
| 斜距 | `MaterialSighting.distance_mm` 为 xyz 模长，与深度不同；不能为统一字段名而交换含义 |
| 圆尺寸 | 定位区域是半径，观测 `diameter_px` 是直径，转换为 ×2 |
| 边缘支撑 | `FaceQuality.edge_support` 为 0–1，观测字段为百分比，转换为 ×100 |
| 时间 | 保留当前读取成功后、推理前的时间戳与丢帧策略；它是启发式新鲜度处理，不宣称具有硬件拍摄时间保证 |
| 文本 | `summary()` / `raw` 供诊断；本地控制从结构化结果取值，不走文本往返 |

`test_camera_source.py:58-74` 已有多项换算断言，应迁移并保留。不能只因 `eye_in_hand` 没直接读取某字段，就删掉网络解析、`summary()` 和诊断仍使用的字段。

不为了把 dataclass 减到某个数而合并类型；先减少调用者必须理解的类型。保持 `MaterialReport` 和控制器观测之间的明确转换，比把深度与斜距改成同名更重要。

### 5.2 抓取行为不因会话封装而缩减

- `align_xy()` 包含原 `run_align` 的悬停，不只是底层 `align_x_y()` 改名。
- `align_and_descend()` 保持悬停、安全抬升、平面对准、下降、后缩顺序；内部跳过重复高度检查时仍满足原前提。
- 下降保留平面对准检查和固定脉冲行为；可选升降尺度只用于现有估算，不把它变成新的下降裁剪规则。
- 保留粗细档、背隙补偿、停稳、过期帧处理、预算和失败即停止后续步骤的规则。
- 所有覆盖值仅影响本次操作；无隐式标定写回、无自动重试动作、无“尽量完成”模式。

## 6. 对原专项建议的修正

| 原建议或结论 | 修订 |
|---|---|
| 自测迁走后将所有子模块急切导入，删除 runpy 检查 | 真实图片/相机 CLI 仍在；包不能预加载它们，保留针对保留入口的检查 |
| 移动 training 因为没有生产调用 | training 是明确命令行工具，保留现有目录，不纳入此次接口重构 |
| 合并颜色与霍夫、质量与分类模块 | 保留算法职责；共享类型不意味着算法该并在同一文件 |
| 删除 MaterialSighting 的七个“零读取”字段 | 暂不删除；网络解析和诊断也属于消费者，先明确使用契约 |
| 把 calibration 与菜单 19 一起归档 | `vision/config.py` 仍直接使用其类型和解析逻辑，不能只简化 demo 或留几十行兼容类型绕过依赖 |
| 归档视觉 z 仅影响性能 | 仍有 `--align-z-visual`；`LiftScale` 还被悬停/抬升使用，属于业务能力取舍 |
| 未使用角度字段说明缺失安全抬升 | 当前默认先抬升；问题是字段说明与实际实现不一致，不能顺带新增阈值放宽保护 |
| 承诺某四/七文件方案且测试零修改 | 以无环依赖和真实职责为准，测试 import/patch 随定义迁移 |
| 汇总归档量、自测迁移量、移动量为“净减行数” | 删除重复计数；三种统计口径不同，数量不作验收 |

### 6.1 自测和 CLI 的处置原则

逐入口区分“已有 unit 覆盖的重复自测”“尚缺覆盖的自测”“真实图片/相机 CLI”。前两类去重或迁入 unit，第三类保留参数、用途和显式设备打开时机。

无参自测入口的去留与其场景覆盖是两件事；删除 `python -m` 的无参行为前需要明确用途。保留 CLI 时继续验证无 runpy 警告，不能以自测迁完为由删除全部入口检查。

## 7. 验收矩阵

| 编号 | 可以直接检查的结果 | 验证方式 |
|---|---|---|
| V1 | 仅导入公开 Vision 类即可按配置读取结构化报告 | 公开示例的离线假设备用例 + 已有定位回归 |
| V2 | 每会话只打开一次相机、加载一次模型；连续定位不回读默认配置 | 构造/读取次数断言，自定义配置贯穿测试 |
| V3 | 无物料、无画面、模型失败可区分 | 空报告与失败分支；失败不自动换模型或数据源 |
| V4 | 原有位置、分类、单位和时间戳含义不变 | 复用转换断言、定位与新帧测试 |
| G1 | 上层仅传配置，无需组装 arm/source/target/jacobian | 两个 Grasp 示例的完整调用用例 |
| G2 | 缺目标、缺/奇异雅可比时机械臂构造和动作次数均为零 | 在真实资源创建接缝处注入假件并检查次数 |
| G3 | 模型/相机失败时机械臂未使能 | 部分初始化失败用例 |
| G4 | `align_xy` 包含悬停，完整流程顺序不变 | 真实类方法调用和动作记录，不只 patch 一个空门面 |
| G5 | 下降前不满足对准条件时无下降；任一步失败不执行后续动作 | 保留失败路径和零动作断言 |
| R1 | 正常、异常、打开一半和清理报错时均尝试释放全部已取得资源 | 分阶段故障注入；重复 close 与关闭后调用用例 |
| I1 | 导入不读配置、不开设备、不预加载 CLI 后端 | 子进程导入检查和设备创建禁止断言 |
| I2 | 所有保留测试和 manual 入口被发现，真实 CLI 保留用途 | 发现守卫、菜单分发及 runpy 检查 |
| I3 | 只读/示教/标定工具保留各自前提，不被普通会话标定门禁误伤 | 各工具模式回归；只读模式不开机械臂 |

测试应 patch 实际读取依赖的位置：改为实例方法后，编排检查 patch 真实 Grasp 类/实例的方法；脉冲和 sleep 的检查指向 `servo_motion`。不能保留旧模块名 patch 再用转发补丁让测试看起来有效。

完整套件与基础命令见主方案 §6；新增会话测试归入现有 unittest 发现范围，不引入新框架或依赖。Python 3.8 未实跑、真机未核对的部分分别说明，不称全链路已验证。

## 8. 优先级与独立待定项

优先级为：**可调用的 Vision → 可调用的 Grasp → 迁移菜单/应用 → 清理重复自测与过时出口**。先交付真实易用的接口，再评估与此无关的历史删除。

已确认设备由对象管理；静态方法只是无状态操作的一种形式，不强制将连接操作静态化。

以下仍需独立决定，不在本轮悄悄执行：标定焦距是否正式接入抓取主线、独立存在性/测距工具是否保留、平面标定是否整体退出、视觉 z 工具是否退役，以及无参自测入口是否允许取消。现有安全字段说明与实际判据的差异应单列缺陷，不能靠删字段或加阈值完成“结构优化”。

## 9. 参考代码

- [本地相机装配与观测](../modules/grasp/drive/camera_source.py)
- [主线控制及数值对象](../modules/grasp/drive/eye_in_hand.py)
- [菜单 21 装配和资源收尾](../modules/grasp/test/manual/align_xy.py)
- [机械臂连接与控制](../modules/arm/drive/robot_arm.py)
- [视觉配置与路径](../modules/vision/config.py)
- [物料报告与定位器](../modules/vision/drive/material_locate.py)
- [模型惰性创建及物料配置读取](../modules/vision/drive/material_yolo.py)
- [已覆盖的观测换算](../modules/grasp/test/unit/test_camera_source.py)
