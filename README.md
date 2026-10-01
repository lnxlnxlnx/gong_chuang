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
stepper.move_absolute(degrees=720.0, rpm=300, acceleration=10)
stepper.home(HomeMode.SINGLE_TURN_NEAREST)
stepper.stop()
```

`Emm42Driver` 默认按说明书中的 1.8 度、16 细分配置使用 `3200` 脉冲/圈。若驱动器的细分配置不同，初始化时传入实际的 `pulses_per_revolution`。

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
car.left(rpm=40, duration=2.0)        # 向左平移 2 秒
car.strafe_left(rpm=40, duration=2.0) # 向左平移 2 秒后自动停
car.rotate_cw(rpm=30, duration=1.0)   # 原地顺时针转 1 秒
car.stop()
car.disable_all()                     # 松轴
car.close()
```

- 上层整车 API：`MecanumChassis` 的 `forward` / `backward` / `left` / `right` / `rotate_cw` / `rotate_ccw` / `forward_left` / `forward_right` / `backward_left` / `backward_right`，均可带可选 `duration` 自动限时停止。
- 四轮位置 API：`move_wheels_relative(fl, fr, rl, rr, rpm)` 和 `move_wheels_absolute(fl, fr, rl, rr, rpm)`，默认使用 Emm42 多机同步机制：四条 `0xFD` 位置帧缓存后，再发送 `00 FF 66 6B` 同时启动。
- 底层单轮 API（供上层调用）：`enable_wheel` / `set_wheel_speed` / `stop_wheel` / `move_wheel_relative` / `move_wheel_absolute` / `home_wheel` / `get_wheel`。
- 整车低层：`drive(vx, vy, omega, rpm)`（麦轮逆运动学，可斜向移动）、`set_speeds(fl, fr, rl, rr)`。
- 当前实车使用标准麦轮坐标变换 `motion_transform="standard"`，四轮横移都会参与运动。
- 若装配后方向反了：整车方向反 → 把 `drive()` 的 `vx`/`vy` 取反；辊子排布不同 → 换 `pattern="O"`；单轮转向反 → 在 `wheel_signs` 给该轮 `-1`。

## 统一入口

在项目根目录运行 `python test.py` 打开交互菜单。选 1 运行离线单测；2–9 为机械臂、麦轮、电机诊断和视觉的手动真机测试；10 为激光障碍物扫描；11 为电机不动的激光固定测距；12 为总线舵机测试；13、14 为夹爪测试；15 为障碍物模型测试（在摄像头、验证集图片、图片文件夹或视频上试新训练的模型，全部按菜单问答选择，不需要命令行参数）。步进总线端口读取 `config/mecanum_hardware.json` 中当前平台的 `port`，无需每次输入 `--port`；激光测试使用独立的 `config/laser_scan.json`，扫描步长和停稳时间分别由 `scan.step_pulses`、`scan.settle_seconds` 设置。Windows 默认用 `windows`，Jetson 默认用 `jetson`；运行前先把配置中的端口改成实际设备名。真机动作仍会逐项请求确认。

```powershell
python test.py
```

预期：出现编号菜单；输入 `0` 直接退出，不会打开硬件设备。

菜单自动发现 `tests/manual/` 下声明 `MENU_ENTRY = (编号, "菜单名称")` 且提供 `main()` 的脚本。新增真机测试时，只需在新脚本中声明未占用的编号和名称，无需修改 `test.py`；展示菜单不会导入真机模块，选择后才导入并运行。

菜单 `12. 总线舵机测试` 走 PDF 中的 **115200 bps 直连总线舵机协议**，直接使用 `motor_drivers.bus_servo.BusServo`。端口、波特率和舵机 ID 从 `config/servo.json` 读取：平台串口取 `windows.port` / `jetson.port`，舵机参数取 `bus.baudrate` 和 `bus.servo_id`，菜单里不再要求输入 ID。测试先只读当前位置；只有查询成功、输入目标脉冲位置和时长，并输入大写 `YES`，才会发送一次运动指令，随后读回实际位置。当前 Windows 端口填 `COM14`，舵机 ID 为 `1`；Jetson 串口需按实际接线填写。

### 夹爪测试

菜单 `13. 夹爪测试` 和 `14. 夹爪直连调试` 都走**同一条 115200 bps 直连通路**，使用 `BusServo`，串口取 `config/servo.json` 的平台 `port`，舵机 ID 取 `gripper.servo_id`，位置和耗时取 `gripper` 下的 `open_position` / `closed_position` / `duration_ms`。两者都不再经过 9600 bps 控制板。菜单 13 先读位置，再对张开、闭合分别要求大写 `YES`，每次动作后读回实际位置；菜单 14 是同一条通路的快速试动入口。**300/500 只是随意选定的试动值，后续必须按真实夹爪行程修改，不能视为最终张开与夹紧位置。**`motor_drivers.servo_controller.ServoController` 是旧工程 9600 bps 多舵机控制板协议，当前没有任何菜单入口使用它。

完整说明、Jetson 迁移步骤及直连调试记录见 `夹爪控制板使用说明.md`。

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
from vision import DistanceEstimator, USBCamera, create_detector, default_model_path

with USBCamera(0).initialize() as cam, \
        create_detector(default_model_path()) as detector:
    est = DistanceEstimator(focal_px=focal, real_widths={"target-1": 60.0})
    while True:
        frame = cam.read()
        if frame is None:
            break
        for hit in detector.detect(frame):      # hit: vision.Detection
            dist_mm = est.estimate(hit)         # 未知类别返回 None
            print(hit.name, hit.center, dist_mm)
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
