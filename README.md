# Jetson Orin Nano 电机串口驱动

此目录包含两套 Python 串口协议驱动，以及其上的麦轮小车控制层：

- `motor_drivers.bus_servo.BusServo`：幻尔总线舵机，115200 bps、半双工 UART。
- `motor_drivers.emm42.Emm42Driver`：Emm42 V5.0 闭环步进驱动器，115200 bps TTL UART。
- `mecanum.MecanumDrive`：四台 Emm42 驱动器组成的麦轮小车控制层，一组串口即可。

## 接线约束

步进驱动器使用一个 USB-TTL 模块：模块 `TX -> R/A/H`、`RX -> T/B/L`、`GND -> GND`；驱动器电源独立接 `12-32 V`。

总线舵机为单线、5 V 上拉的半双工 UART，波特率 115200。Jetson UART 直连时也需要按 `docs/幻尔科技总线舵机通信协议.pdf` 图 1 处理发送/接收方向与电平；USB 转接板只是另一种串口接法。普通全双工 UART 的 `TX` 与 `RX` 不能直接并接在舵机信号线上。

运行前在 Jetson 上确认设备名：

```powershell
Get-ChildItem /dev/ttyUSB*
```

如果使用 USB 转接板，预期看到独立的串口设备，例如 `/dev/ttyUSB0`（舵机）和 `/dev/ttyUSB1`（步进驱动）。使用 Jetson 板载 UART 时，按实际接线填写对应的 `/dev/ttyTHS*` 设备名。本代码在运行时需要 `pyserial`。

## 使用示例

```python
from motor_drivers import BusServo, Direction, Emm42Driver, HomeMode

servo = BusServo("/dev/ttyUSB0", servo_id=1).initialize()
servo.set_torque(True)
servo.set_position(position=500, duration_ms=1000)
servo.set_speed(-250)  # 电机模式，范围 -1000..1000
servo.stop()

stepper = Emm42Driver("/dev/ttyUSB1", motor_id=1).initialize()
stepper.enable(True)
stepper.set_speed(Direction.CW, rpm=300, acceleration=10)
stepper.move_relative(degrees=-360.0, rpm=300, acceleration=10)
stepper.move_relative_pulses(pulses=-3200, rpm=300, acceleration=10)
stepper.move_absolute(degrees=720.0, rpm=300, acceleration=10)
stepper.home(HomeMode.SINGLE_TURN_NEAREST)
stepper.stop()
```

`Emm42Driver` 默认按说明书中的 1.8 度、16 细分配置使用 `3200` 脉冲/圈。若驱动器的细分配置不同，初始化时传入实际的 `pulses_per_revolution`。

`move_relative` / `move_absolute` 收角度，`move_relative_pulses` / `move_absolute_pulses` 收脉冲数（正数 = CW）；后者不经过角度换算，机械臂和麦轮的位置模式都走这一条。

## 麦轮小车（MecanumDrive）

软件逻辑层固定使用 `FL/FR/RL/RR` 四个轮位，标准默认地址顺序为 `1/2/3/4`。
实际驱动器地址属于硬件配置，由 `config/mecanum_hardware.json` 注入；当前实车映射为
`FL=3 FR=4 RL=2 RR=1`。更换轮子位置时只需修改配置，不需要修改运动 API 或上层代码。

```python
from mecanum_config import create_chassis, load_profile

profile = load_profile("jetson")
car = create_chassis(profile).initialize()
car.enable_all()                      # 移动前必须先使能扭矩
car.forward(rpm=60)                   # 持续前进，直到 stop()
car.left(rpm=40, duration=2.0)        # 速度模式：向左平移 2 秒
car.strafe_left(rpm=40, duration=2.0) # 向左平移 2 秒后自动停
car.rotate_cw(rpm=30, duration=1.0)   # 原地顺时针转 1 秒
car.forward(rpm=200, pulses=3200)     # 位置模式：四轮各走 3200 脉冲（一圈）后返回
car.drive_pulses(1.0, 1.0, 0.0, pulses=1600, rpm=200)  # 任意方向走固定脉冲数
car.stop()
car.disable_all()                     # 松轴
car.close()
```

### 统一运动参数（机械臂 / 底盘一致）

所有整车动作和机械臂动作使用同一套参数名，含义相同（实现见 `motion.py`）：

| 参数 | 含义 | 机械臂 `RobotArm` | 底盘 `MecanumChassis` / `MecanumDrive` |
|---|---|---|---|
| `pulses` | 位置模式：本次走多少脉冲（正数，方向由动作方法决定） | 必填（第一个位置参数） | 可选关键字；给 `pulses` 就走位置模式 |
| `duration` | 速度模式：一条速度命令持续多少秒 | 不支持（机械臂只有相对位置模式） | 可选；与 `pulses` 只能给一个 |
| `rpm` | 目标转速，缺省用配置值 | 可选关键字 | 必填（第一个位置参数） |
| `acceleration` | 曲线加减速档位 `0..255`，缺省用配置值 | 可选关键字 | 可选 |
| `wait` | `True`（默认）等到动作走完再返回；`False` 只发命令 | 默认 `True` | 默认 `True`（位置模式用） |

- **`wait` 为什么重要**：Emm42 位置命令"发完就返回"，驱动器不会告诉上位机什么时候停。机械臂 5/6/7 号驱动器是 `Response=None`（连 ACK 都不回），如果不等待，连续调用会让后一条位置帧覆盖前一条的目标，看起来就是"上一个动作被跳过"。`wait=True` 按说明书的曲线加减速公式估算耗时并等待（`motion.estimate_move_seconds`），`MotionReport.estimated_seconds` 会把它带回来。
- 需要毫秒级精确时不能用估算：那要用 `0x3A` 读运动状态标志（`&0x02` 到位位）轮询，当前没有实现。
- 轮子位置模式默认走 Emm42 多机同步：四条 `0xFD` 帧缓存后发 `00 FF 66 6B` 同时启动；0 脉冲的轮子不下发。
- 上层整车 API：`MecanumChassis` 的 `forward` / `backward` / `left` / `right` / `rotate_cw` / `rotate_ccw` / `forward_left` / `forward_right` / `backward_left` / `backward_right`，都可带 `duration`（速度模式）、`pulses`（位置模式）或都不带（持续运动）。
- 四轮位置 API：`move_wheels_relative(fl, fr, rl, rr, rpm)` 按角度、`move_wheels_relative_pulses(fl, fr, rl, rr, rpm)` 按脉冲数（同样默认同步启动），以及对应的一组 `*_absolute*`。
- 底层单轮 API（供上层调用）：`enable_wheel` / `set_wheel_speed` / `stop_wheel` / `move_wheel_relative` / `move_wheel_absolute` / `home_wheel` / `get_wheel`。
- 整车低层：`drive(vx, vy, omega, rpm)`（麦轮逆运动学，可斜向移动）、`drive_pulses(vx, vy, omega, pulses, rpm)`（同矩阵，位移用脉冲数给）、`set_speeds(fl, fr, rl, rr)`。
- 当前实车使用标准麦轮坐标变换 `motion_transform="standard"`，四轮横移都会参与运动。
- 若装配后方向反了：整车方向反 → 把 `drive()` 的 `vx`/`vy` 取反；辊子排布不同 → 换 `pattern="O"`；单轮转向反 → 在 `wheel_signs` 给该轮 `-1`。

## 统一入口

在项目根目录运行 `python test.py` 打开交互菜单。选 1 运行离线单测；2–9 为机械臂、麦轮、电机诊断和视觉的手动真机测试；10 为激光障碍物扫描；11 为电机不动的激光固定测距；12 为总线舵机测试；13、14 为夹爪测试；15 为障碍物模型测试（在摄像头、验证集图片、图片文件夹或视频上试新训练的模型，全部按菜单问答选择，不需要命令行参数）；16 为 COIN-D6 激光雷达点云采集；17 为 COIN-D6 实时点云窗口（滑块或输入框调距离与角度窗口）。步进总线端口读取 `config/mecanum_hardware.json` 中当前平台的 `port`，无需每次输入 `--port`；激光测试使用独立的 `config/laser_scan.json`，扫描步长和停稳时间分别由 `scan.step_pulses`、`scan.settle_seconds` 设置；D6 激光雷达使用 `config/coin_d6.json`，端口、采集圈数、角度与距离窗口都在里面改。Windows 默认用 `windows`，Jetson 默认用 `jetson`；运行前先把配置中的端口改成实际设备名。真机动作仍会逐项请求确认。

```powershell
python test.py
```

预期：出现编号菜单；输入 `0` 直接退出，不会打开硬件设备。

菜单自动发现 `tests/manual/` 下声明 `MENU_ENTRY = (编号, "菜单名称")` 且提供 `main()` 的脚本。新增真机测试时，只需在新脚本中声明未占用的编号和名称，无需修改 `test.py`；展示菜单不会导入真机模块，选择后才导入并运行。

菜单 `12. 总线舵机测试` 走 PDF 中的 **115200 bps 直连总线舵机协议**，直接使用 `motor_drivers.bus_servo.BusServo`。端口、波特率和舵机 ID 从 `config/servo.json` 读取：平台串口取 `windows.port` / `jetson.port`，舵机参数取 `bus.baudrate` 和 `bus.servo_id`，菜单里不再要求输入 ID。测试先只读当前位置；只有查询成功、输入目标脉冲位置和时长，并输入大写 `YES`，才会发送一次运动指令，随后读回实际位置。当前 Windows 端口填 `COM14`，舵机 ID 为 `1`；Jetson 串口需按实际接线填写。

### 夹爪测试

菜单 `13. 夹爪测试` 和 `14. 夹爪直连调试` 都走**同一条 115200 bps 直连通路**，使用 `BusServo`，串口取 `config/servo.json` 的平台 `port`，舵机 ID 取 `gripper.servo_id`，位置和耗时取 `gripper` 下的 `open_position` / `closed_position` / `duration_ms`。两者都不再经过 9600 bps 控制板。菜单 13 先读位置，再对张开、闭合分别要求大写 `YES`，每次动作后读回实际位置；菜单 14 是同一条通路的快速试动入口。**300/500 只是随意选定的试动值，后续必须按真实夹爪行程修改，不能视为最终张开与夹紧位置。**`motor_drivers.servo_controller.ServoController` 是旧工程 9600 bps 多舵机控制板协议，当前没有任何菜单入口使用它。

完整说明、Jetson 迁移步骤及直连调试记录见 `夹爪控制板使用说明.md`。

## D6 激光雷达（lidar）

`lidar` 包封装国科光芯 COIN-D6 的串口指令与点云解析（协议见
`docs/国科光芯/1 D6激光雷达手册资料/`），并把采集结果画成点云图：

- `lidar.protocol`：帧编解码。启动 `AA 55 F0 0F`、停止 `AA 55 F5 0A`，应答为 12 字节
  `A5 5A ...`；上传帧 `A5 5A LEN CS TYPE DATA`（`0x01` 设备信息、`0x81` 点云）；点云包为
  `AA 55 M&T LSN FSA LSA CS` + 每点 3 字节（14 位距离、8 位强度、高反标志）。读取端会跳过噪声、
  坏校验包并自动重新同步，也同时接受带 `A5 5A` 外层封装与裸 `AA 55` 两种点云流。
- `lidar.CoinD6`：一次串口会话，`initialize()` / `start()` / `read_device_info()` /
  `collect_scan()` / `stop()` / `close()`；`scan(config.scan)` 一次跑完启停与采集。
- `lidar.PointCloud` / `CloudPoint`：点云数据（角度、距离、强度、高反），可 `save_csv()`。
- `lidar.visualize`：俯视点云图（按距离着色、等比例、带距离环），保存 PNG 并可弹窗。
- `lidar.simulate`：内置虚拟雷达，不接硬件就能跑通「串口 → 解析 → 出图」整条链路。

全部可调参数集中在 `config/coin_d6.json`，改完存盘即生效，菜单里不再输入端口：

- `profiles.windows.port` / `profiles.jetson.port`：串口设备名（Windows 看设备管理器，Jetson 用 `/dev/ttyUSB*`）。
- `device.baudrate`：波特率，D6/D2 出厂为 `230400`。
- `device.strict_checksum`：点云包校验不过时是否丢弃（默认 `true`；排查接线时可临时设 `false`）。
- `device.require_ack`：启停是否必须收到应答（手册写明有应答；若某台雷达不回应答可设 `false`）。
- `scan.revolutions`：采集几圈完整点云。
- `scan.angle_min_degrees` / `angle_max_degrees`：保留的角度窗口（以雷达 0° 为前方，逆时针为正）。
- `scan.min_distance_mm` / `max_distance_mm`：保留的距离窗口，`max_distance_mm` 最大 16383（14 位距离）。
- `scan.clockwise` / `angle_offset_degrees`：雷达旋转方向（D6 为顺时针，出图时镜像成逆时针）与零点补偿。
- `scan.timeout_seconds` / `max_points`：整次采集超时与点数上限。
- `visualization.range_mm` / `color_by` / `colormap` / `point_size` / `show_window`：出图范围、着色字段（`distance`/`intensity`/`angle`）与是否弹窗。

```python
from lidar import CoinD6, load_lidar_config, render_point_cloud

config = load_lidar_config()                       # 也可传 "windows" / "jetson"
with CoinD6(config.device.port, baudrate=config.device.baudrate) as lidar:
    cloud = lidar.scan(config.scan)                # 启动、采 N 圈、停止
cloud.save_csv("output/lidar_cloud/scan.csv")
render_point_cloud(cloud, config.view, "output/lidar_cloud/scan.png")
```

菜单 16 采集一次点云，写出 `output/lidar_cloud/cloud-<时间戳>.csv` 与 `.png`，并按
`visualization.show_window` 弹窗。没有硬件时用内置虚拟雷达验证整条链路：

```powershell
python -m tests.manual.coin_d6 --simulate --no-window
```

该命令不需要串口，会生成一张模拟房间（含圆柱障碍与高反条）的点云图。真机运行前先把
`config/coin_d6.json` 中的端口改成实际设备名，并确认雷达前方无遮挡。

第一次接真机（或采集一直失败）时先跑诊断：它只发一次启动指令，把**实际发送的原始字节**
和雷达回传的原始字节、解析统计一起打出来，用来区分接线/波特率问题和协议问题。

```powershell
python -m tests.manual.coin_d6 --diagnose
```

正常应看到 `已发送启动指令（原始字节，非文本）：AA 55 F0 0F`、一条 `A5 5A …` 应答和若干
点云包；若点云包为 0 且回传字节像乱码，先核对波特率（D6 出厂 230400）。

### 实时上位机窗口

菜单 17 是实时点云窗口：后台线程持续读取雷达，界面按固定间隔重画，**距离窗口、角度窗口
可以拖滑块，也可以直接在右侧输入框里打字**（输入超范围会自动夹到边界），改完立刻生效——
过滤作用在已缓存的点上，不是只影响后面新来的点，所以调窗口能马上看清效果。窗口里还有：

- **保留圈数**：缓存最近 1~5 圈，调大就能看到拖影/多次扫描叠加。
- **着色字段**：距离 / 强度 / 角度，颜色条范围跟着距离窗口走。
- **暂停显示 / 清空缓存 / 保存快照**：暂停只冻结画面（后台仍在采集）；保存快照把当前视图
  写成 `output/lidar_cloud/live-<时间戳>.csv`（点数据）和 `.png`（含界面的整张图）。
- 视图范围跟着"距离上限"缩放，距离环按窗口的 25/50/75/100% 画。

```powershell
python -m tests.manual.coin_d6_live --simulate     # 不接硬件先看界面
python -m tests.manual.coin_d6_live                # 真机（端口取 config/coin_d6.json）
```

关闭窗口即停止雷达转动并释放串口；真机报错会显示在右下角状态区，不会崩掉窗口。
`lidar/live.py` 里的 `LiveCloud` 是这套读数的模型（后台线程 + 滚动缓存 + 快照过滤），
与界面无关，可以直接在别的程序里复用。

离线单测覆盖帧解析、校验、断包重同步、角度换算、扫描窗口过滤与出图，全部不打开串口：

```powershell
python -m unittest tests.test_lidar_protocol tests.test_lidar_device tests.test_lidar_config `
  tests.test_lidar_cloud tests.test_lidar_visualize tests.test_lidar_menu -v
```

`python gripper.py` 本身不会执行动作；`gripper.Gripper` 类仍按 9600 bps 控制板协议封装，尚未改到直连通路，应用代码暂不要用它驱动当前硬件。夹爪直连调试可运行 `python -m tests.manual.gripper_direct`（或测试菜单 14），逐项输入 `YES` 后依次试动 300 和 500。

```python
import time
from gripper import load_gripper_config
from motor_drivers.bus_servo import BusServo, load_bus_config

config = load_gripper_config()
bus = load_bus_config()
servo = BusServo(bus.port, config.servo_id, baudrate=bus.baudrate, timeout=0.5).initialize()
try:
    servo.set_position(config.open_position, config.duration_ms)
    time.sleep(config.duration_ms / 1000 + 0.2)
    print("张开后读回：", servo.read_position())
finally:
    servo.close()  # 只关闭串口，不会改变夹爪位置
```

这段直接读写`config/servo.json` 中的串口与 `gripper` 参数，与菜单 13、14 是同一条通路；`gripper.Gripper` 仍是旧的 9600 控制板封装，暂未接入当前硬件。夹爪开合尚未与机械臂动作组成自动抓取流程。Jetson 使用前需填写 `config/servo.json` 中的 `jetson.port`。

`main.py` 是项目应用示例，依次调用底盘、机械臂和视觉。只需要其中一部分时，在 `main()` 中注释对应调用；总线舵机读取示例也在文件中，使用前填写独立的舵机串口。

```powershell
python main.py
```

预期：按程序中的调用顺序进入各模块演示，运动前请求确认；视觉窗口按 `q` 退出。运行它需要已连接的硬件及当前平台对应的模型文件。

## 本机协议测试

```powershell
Set-Location E:\SC_project\note\obsidian_note\fifth_semester\python_study\gong_chuang
python -m unittest discover -s tests -v
```

预期：所有测试通过，且无需连接 USB-TTL 或电机。

## 测试目录

- `tests/test_*.py`：离线单元测试，不会打开串口或驱动电机。
- `tests/manual/`：机械臂、麦轮、电机诊断及视觉的手动真机脚本。

真机测试从项目根目录运行菜单。开始前让车轮悬空，并确保急停措施可用：

```bash
python test.py
```

预期：选择 3 依次测试四轮正反转；选择 4 测试底盘各方向。两项均按当前平台配置取端口和速度。

所有常用硬件参数集中在 `config/mecanum_hardware.json`：

- `vehicle.motor_ids`：硬件实际地址到软件轮位的映射；软件 API 始终按 `FL/FR/RL/RR` 逻辑轮位工作。
- `vehicle`：单轮方向修正、加速度、麦轮 X/O 排布和旋转增益；两套平台共用。
- `vehicle.motion_transform`：运动坐标变换，当前实车为 `standard`。
- `profiles.jetson`：Jetson 串口和两类测试的默认速度、时长。
- `profiles.windows`：Windows `COM` 串口和两类测试的默认速度、时长。

Windows 上将 `profiles.windows.port` 改为设备管理器显示的实际 `COM` 号；Jetson 上核对 `profiles.jetson.port`。无论正常结束、异常或 `Ctrl+C`，麦轮脚本都会发送停止命令、失能扭矩并关闭串口。
## 视觉外设（vision）

USB 摄像头 + YOLOv8 检测，封装为 `vision` 包，供主程序和示例复用：

- `vision.create_detector(vision.default_model_path())`：Windows 用 ONNX，Jetson 用 TensorRT；都返回 `list[vision.Detection]`。
- `vision.USBCamera`：USB 摄像头封装（`initialize()/read()/release()`，支持 `with`）。
- `vision.open_preview()` / `vision.close_previews()`：HighGUI 预览窗口。**窗口名必须是 ASCII，并且要先显式 `namedWindow` 再 `imshow`**：本机实测中文窗口名会显示成 `??????`（Qt 后端里是 `椤圭洰瑙嗚…` 这样的乱码），不同发行版的 OpenCV 窗口后端（pip 的 Win32UI / conda-forge 的 Qt6）隐式建窗行为也不一样。所以预览入口统一用 `vision-preview` 这个名字。
- `vision.looks_blank(frame)` / `vision.warn_blank_preview(camera_id)`：`read()` 成功但画面是纯黑/纯白时（被别的程序占用、序号指向没有画面的设备、镜头盖关着）打印排查清单，而不是让用户对着一个纯色窗口猜。
- `vision.DistanceEstimator`：单目测距（针孔模型，需要已知类别真实宽度）。

菜单 9 和 `main.py` 的视觉演示从 `config/vision.json` 读取摄像头编号。Windows 当前设为 `1`（外接 USB 摄像头），Jetson 设为 `0`；如果设备枚举顺序不同，修改对应平台的 `camera_id`。指定编号无法打开时会报错，不会自动切换到前置摄像头。

### 依赖

Windows 使用 OpenCV ONNX 推理，需要 `opencv-python`、`numpy`。Jetson 在 `tensorrt` conda 环境执行，还需要 `tensorrt`、`pycuda`。

### 模型

模型文件放 `models/`（已 gitignore，不入库）：Windows 用 `models/best.onnx`，Jetson 用 `models/best.engine`。当前两份模型来自远端 `/home/pi/workspace/GC_robot_3G/yolov8n/`；ONNX 输入为 `1×3×640×640`，输出为 `1×14×8400`。
类别表在 `vision/yolov8.py::DEFAULT_CLASS_NAMES`，与模型训练标签顺序一致；换模型时同步修改即可。

自己训练障碍物模型：`models/data` 里的图片和标注用 `training/prepare_dataset.py` 切分，`training/train.py` 训练并导出 ONNX，完整流程（参数建议、预训练权重下载、上车注意事项）见 `docs/障碍物模型训练指南.md`。训练完在测试菜单选 `15. 障碍物模型测试`，按提示选择「哪个模型 / 怎么测 / 置信度阈值」即可；它会把类别表和模型绑在一起，选新模型时自动按单类 `obstacle` 解析，不会出现菜单 9 那种类名错位。

### 运行示例

```bash
cd /home/pi/gong_chuang
source /home/pi/miniforge3/bin/activate tensorrt

# 在菜单中选择 9；Windows 默认 best.onnx，Jetson 默认 best.engine
python test.py
```

预览窗口按 `q` 退出。`tests/manual/vision.py` 中的 `focal_px` 默认值（560px）
是占位符，正式使用前需按下方校准。

### 程序中使用

```python
import cv2
from vision import (DistanceEstimator, USBCamera, close_previews, create_detector,
                    default_model_path, open_preview)

window = open_preview()          # ASCII 窗口名 + 显式建窗
try:
    with USBCamera(0).initialize() as cam, \
            create_detector(default_model_path()) as detector:
        est = DistanceEstimator(focal_px=focal, real_widths={"target-1": 60.0})
        while True:
            frame = cam.read()
            if frame is None:
                break
            hits = detector.detect(frame)       # hit: vision.Detection
            for hit in hits:
                dist_mm = est.estimate(hit)     # 未知类别返回 None
                print(hit.name, hit.center, dist_mm)
            cv2.imshow(window, detector.draw(frame, hits))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
finally:
    close_previews()
```

### 校准焦距

水平焦距（像素）与水平 FOV 的关系：

```python
from vision import focal_length_from_fov
focal_px = focal_length_from_fov(fov_deg=90.0, image_width_px=640)  # ~320
```

实际应放一个已知宽度物体在已知距离处测量：`focal_px = 距离 × 像素宽 / 真实宽`，取多次平均。

### 测试

```bash
cd /home/pi/gong_chuang
python -m unittest tests.test_vision -v
```

离线单测不依赖 TensorRT 与摄像头，只验证预处理、解码、NMS、类名与测距纯计算逻辑。
