# Jetson Orin Nano 电机串口驱动

此目录包含两套 Python 串口协议驱动，以及其上的麦轮小车控制层：

- `motor_drivers.bus_servo.BusServo`：幻尔总线舵机，115200 bps、半双工 UART。
- `motor_drivers.emm42.Emm42Driver`：Emm42 V5.0 闭环步进驱动器，115200 bps TTL UART。
- `mecanum.MecanumDrive`：四台 Emm42 驱动器组成的麦轮小车控制层，一组串口即可。

## 接线约束

步进驱动器使用一个 USB-TTL 模块：模块 `TX -> R/A/H`、`RX -> T/B/L`、`GND -> GND`；驱动器电源独立接 `12-32 V`。

总线舵机为单线、5 V 上拉的半双工 UART。应使用总线舵机 USB 转接板，或具备半双工收发和 3.3 V/5 V 电平转换的 USB 串口模块。普通 USB-TTL 的 `TX` 与 `RX` 不能直接并接在舵机信号线上。

运行前在 Jetson 上确认设备名：

```powershell
Get-ChildItem /dev/ttyUSB*
```

预期看到两个独立设备，例如 `/dev/ttyUSB0`（舵机）和 `/dev/ttyUSB1`（步进驱动）。本代码在运行时需要 `pyserial`，请在 Jetson 的目标 Python 环境中自行安装。

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

四台 Emm42 驱动器共用一根 115200 TTL 总线（一个 USB-TTL 即可），默认主址 `FL=1 FR=2 RL=3 RR=4`。

```python
from mecanum import MecanumDrive

car = MecanumDrive("/dev/ttyUSB0").initialize()
car.enable_all()                      # 移动前必须先使能扭矩
car.forward(rpm=60)                   # 持续前进，直到 stop()
car.strafe_left(rpm=40, duration=2.0) # 向左平移 2 秒后自动停
car.rotate_cw(rpm=30, duration=1.0)   # 原地顺时针转 1 秒
car.stop()
car.disable_all()                     # 松轴
car.close()
```

- 高级方向方法：`forward` / `backward` / `strafe_left` / `strafe_right` / `rotate_cw` / `rotate_ccw`，均可带可选 `duration` 自动限时停止。
- 底层单轮 API（供上层调用）：`enable_wheel` / `set_wheel_speed` / `stop_wheel` / `move_wheel_relative` / `move_wheel_absolute` / `home_wheel` / `get_wheel`。
- 整车低层：`drive(vx, vy, omega, rpm)`（麦轮逆运动学，可斜向移动）、`set_speeds(fl, fr, rl, rr)`。
- 若装配后方向反了：整车方向反 → 把 `drive()` 的 `vx`/`vy` 取反；辊子排布不同 → 换 `pattern="O"`；单轮转向反 → 在 `wheel_signs` 给该轮 `-1`。

## 本机协议测试

```powershell
Set-Location E:\SC_project\note\obsidian_note\fifth_semester\python_study\gong_chuang
python -m unittest discover -s tests -v
```

预期：所有测试通过，且无需连接 USB-TTL 或电机。

## 测试目录

- `tests/`：离线单元测试，不会打开串口或驱动电机。
- `hardware_tests/emm42/`：单个 Emm42 驱动器的真机诊断脚本。
- `hardware_tests/mecanum/`：四麦轮底盘的真机测试脚本。

真机测试必须在 Jetson 上从项目根目录以模块方式执行。开始前让车轮悬空，并确保急停措施可用：

```bash
# 依次测试 FL、FR、RL、RR 的正反转；每个轮子开始前需要人工确认
python -m hardware_tests.mecanum.individual_wheels --profile jetson

# 依次测试前进、后退、左移、右移、顺时针旋转、逆时针旋转
python -m hardware_tests.mecanum.chassis_motion --profile jetson
```

所有常用硬件参数集中在 `config/mecanum_hardware.json`：

- `vehicle`：四轮地址、单轮方向修正、加速度、麦轮 X/O 排布和旋转增益；两套平台共用。
- `profiles.jetson`：Jetson 串口和两类测试的默认速度、时长。
- `profiles.windows`：Windows `COM` 串口和两类测试的默认速度、时长。

Windows 上将 `profiles.windows.port` 改为设备管理器显示的实际 `COM` 号后执行：

```powershell
py -m hardware_tests.mecanum.individual_wheels --profile windows
py -m hardware_tests.mecanum.chassis_motion --profile windows
```

两个脚本均支持 `--config` 指定其他 JSON 文件，也可通过 `--port`、`--rpm`、`--duration` 和 `--acceleration` 临时覆盖当前档案；整车脚本额外支持 `--pause`。无论正常结束、异常或 `Ctrl+C`，脚本都会发送四轮停止命令、失能扭矩并关闭串口。
## 视觉外设（vision）

USB 摄像头 + YOLOv8 TensorRT 检测，封装为 `vision` 包，供主程序和示例复用：

- `vision.YOLOv8Detector`：加载 `models/*.engine`，对 BGR 帧做端到端检测，返回 `list[vision.Detection]`。
- `vision.USBCamera`：USB 摄像头封装（`initialize()/read()/release()`，支持 `with`）。
- `vision.DistanceEstimator`：单目测距（针孔模型，需要已知类别真实宽度）。

### 依赖

在 Jetson 的 `tensorrt` conda 环境执行；需要 `tensorrt`、`pycuda`、`numpy`、`opencv-python`。
推理引擎用 ultralytics YOLOv8 导出（动态尺寸或固定尺寸均可）。

### 模型

引擎文件放 `models/`（已 gitignore，不入库）：`models/best.engine`、`models/best.onnx`。
类别表在 `vision/yolov8.py::DEFAULT_CLASS_NAMES`，与模型训练标签顺序一致；换模型时同步修改即可。

### 运行示例

```bash
cd /home/pi/gong_chuang
source /home/pi/miniforge3/bin/activate tensorrt

# 基础预览：默认 models/best.engine，摄像头 /dev/video0
python examples/demo_vision.py

# 调整阈值或改摄像头
python examples/demo_vision.py --conf 0.5 --camera 1

# 启用测距显示（obj-red 真实宽度 120mm、zhangai 80mm）
python examples/demo_vision.py --real-widths "obj-red=120,zhangai=80"
```

预览窗口按 `q` 退出。`--real-widths` 与 `demo_vision.py` 内的 `focal_px` 默认值（560px）
是占位符，正式使用前需按下方校准。

### 程序中使用

```python
from vision import DistanceEstimator, USBCamera, YOLOv8Detector

with USBCamera(0).initialize() as cam, \
        YOLOv8Detector("models/best.engine") as detector:
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