# Jetson Orin Nano 电机串口驱动

此目录包含两套 Python 串口协议驱动，以及其上的麦轮小车控制层：

- `modules.servo.drive.bus_servo.BusServo`：幻尔总线舵机，115200 bps、半双工 UART。
- `modules.motor.drive.emm42.Emm42Driver`：Emm42 V5.0 闭环步进驱动器，115200 bps TTL UART。
- `modules.chassis.drive.mecanum.MecanumDrive`：四台 Emm42 驱动器组成的麦轮小车控制层，一组串口即可。

## 接线约束

步进驱动器使用一个 USB-TTL 模块：模块 `TX -> R/A/H`、`RX -> T/B/L`、`GND -> GND`；驱动器电源独立接 `12-32 V`。

总线舵机为单线、5 V 上拉的半双工 UART，波特率 115200。Jetson UART 直连时也需要按 `modules/servo/docs/幻尔科技总线舵机通信协议.pdf` 图 1 处理发送/接收方向与电平；USB 转接板只是另一种串口接法。普通全双工 UART 的 `TX` 与 `RX` 不能直接并接在舵机信号线上。

运行前在 Jetson 上确认设备名：

```powershell
Get-ChildItem /dev/ttyUSB*
```

如果使用 USB 转接板，预期看到独立的串口设备，例如 `/dev/ttyUSB0`（舵机）和 `/dev/ttyUSB1`（步进驱动）。使用 Jetson 板载 UART 时，按实际接线填写对应的 `/dev/ttyTHS*` 设备名。本代码在运行时需要 `pyserial`。

## 使用示例

```python
from modules.motor.drive.emm42 import Direction, Emm42Driver, HomeMode
from modules.servo.drive.bus_servo import BusServo

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
实际驱动器地址属于硬件配置，由 `modules/chassis/config.json` 注入；当前实车映射为
`FL=3 FR=4 RL=2 RR=1`。更换轮子位置时只需修改配置，不需要修改运动 API 或上层代码。

```python
from modules.chassis.config import create_chassis, load_profile

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

所有整车动作和机械臂动作使用同一套参数名，含义相同（实现见 `modules/motor/drive/motion.py`）：

| 参数 | 含义 | 机械臂 `RobotArm` | 底盘 `MecanumChassis` / `MecanumDrive` |
|---|---|---|---|
| `pulses` | 位置模式：本次走多少脉冲（正数，方向由动作方法决定） | 必填（第一个位置参数） | 可选关键字；给 `pulses` 就走位置模式 |
| `duration` | 速度模式：一条速度命令持续多少秒 | 不支持（机械臂只有相对位置模式） | 可选；与 `pulses` 只能给一个 |
| `rpm` | 目标转速，缺省用配置值 | 可选关键字 | 必填（第一个位置参数） |
| `acceleration` | 曲线加减速档位 `0..255`，缺省用配置值 | 可选关键字 | 可选 |
| `wait` | `True`（默认）等到动作走完再返回；`False` 只发命令 | 默认 `True` | 默认 `True`（位置模式用） |

- **`wait` 为什么重要**：Emm42 位置命令"发完就返回"，驱动器不会告诉上位机什么时候停。机械臂 5/6/7 号驱动器是 `Response=None`（连 ACK 都不回），如果不等待，连续调用会让后一条位置帧覆盖前一条的目标，看起来就是"上一个动作被跳过"。`wait=True` 按说明书的曲线加减速公式估算耗时并等待（`modules.motor.drive.motion.estimate_move_seconds`），`MotionReport.estimated_seconds` 会把它带回来。
- 需要毫秒级精确时不能用估算：那要用 `0x3A` 读运动状态标志（`&0x02` 到位位）轮询，当前没有实现。
- 轮子位置模式默认走 Emm42 多机同步：四条 `0xFD` 帧缓存后发 `00 FF 66 6B` 同时启动；0 脉冲的轮子不下发。
- 上层整车 API：`MecanumChassis` 的 `forward` / `backward` / `left` / `right` / `rotate_cw` / `rotate_ccw` / `forward_left` / `forward_right` / `backward_left` / `backward_right`，都可带 `duration`（速度模式）、`pulses`（位置模式）或都不带（持续运动）。
- 四轮位置 API：`move_wheels_relative(fl, fr, rl, rr, rpm)` 按角度、`move_wheels_relative_pulses(fl, fr, rl, rr, rpm)` 按脉冲数（同样默认同步启动），以及对应的一组 `*_absolute*`。
- 底层单轮 API（供上层调用）：`enable_wheel` / `set_wheel_speed` / `stop_wheel` / `move_wheel_relative` / `move_wheel_absolute` / `home_wheel` / `get_wheel`。
- 整车低层：`drive(vx, vy, omega, rpm)`（麦轮逆运动学，可斜向移动）、`drive_pulses(vx, vy, omega, pulses, rpm)`（同矩阵，位移用脉冲数给）、`set_speeds(fl, fr, rl, rr)`。
- 当前实车使用标定后的麦轮坐标变换 `motion_transform="calibrated"`，四轮横移都会参与运动。
- 若装配后方向反了：整车方向反 → 把 `drive()` 的 `vx`/`vy` 取反；辊子排布不同 → 换 `pattern="O"`；单轮转向反 → 在 `wheel_signs` 给该轮 `-1`。

## 统一入口

在项目根目录运行 `python test.py` 打开交互菜单。选 1 运行离线单测；2–9 为机械臂、麦轮、电机诊断和视觉的手动真机测试；10 为激光障碍物扫描；11 为电机不动的激光固定测距；12 为总线舵机测试；13 为夹爪测试（原来的 14「夹爪直连调试」已合并进来，编号 14 不再使用）；15 为障碍物模型测试（在摄像头、验证集图片、图片文件夹或视频上试新训练的模型，全部按菜单问答选择，不需要命令行参数）；16 为 COIN-D6 激光雷达点云采集；17 为 COIN-D6 实时点云窗口（滑块或输入框调距离与角度窗口）；18 为物料顶部圆面定位（霍夫判断存在并测距，YOLO 补充类别，见 `modules/vision/docs/视觉物料定位.md`）；19 为**摄像头识别 + 机械臂抓取**（见 `modules/grasp/docs/抓取坐标链路.md`），它会在打开硬件前先列出还没做的标定项并拒绝运行，只想核对标定可以用 `--observe-only`；20 为**转盘测试**（三工位分度盘，见 `modules/servo/docs/转盘控制说明.md`）。步进总线端口读取 `modules/motor/config.json` 中当前平台的 `port`（底盘与机械臂共用这一处），舵机总线端口读取 `modules/servo/config.json`；激光测试使用独立的 `modules/laser_scan/config.json`，扫描步长和停稳时间分别由 `scan.step_pulses`、`scan.settle_seconds` 设置；D6 激光雷达使用 `modules/lidar/config.json`，端口、采集圈数、角度与距离窗口都在里面改。Windows 默认用 `windows`，Jetson 默认用 `jetson`；运行前先把配置中的端口改成实际设备名。真机动作仍会逐项请求确认。

```powershell
python test.py
```

预期：出现编号菜单；输入 `0` 直接退出，不会打开硬件设备。

菜单自动发现各模块 `modules/<模块>/test/manual/` 下声明 `MENU_ENTRY = (编号, "菜单名称")` 且提供 `main()` 的脚本（目录清单见 `test.py` 的 `MANUAL_TEST_DIRS`，编号全局唯一由 `tests/test_discovery_coverage.py` 守住）。新增真机测试时，只需在新脚本中声明未占用的编号和名称，无需修改 `test.py`；展示菜单不会导入真机模块，选择后才导入并运行。

菜单 `12. 总线舵机测试` 走 PDF 中的 **115200 bps 直连总线舵机协议**，直接使用 `modules.servo.drive.bus_servo.BusServo`。端口、波特率和舵机 ID 从 `modules/servo/config.json` 读取：平台串口取 `windows.port` / `jetson.port`，舵机参数取 `bus.baudrate` 和 `bus.servo_id`，菜单里不再要求输入 ID。测试先只读当前位置；只有查询成功、输入目标脉冲位置和时长，并输入大写 `YES`，才会发送一次运动指令，随后读回实际位置。当前 Windows 端口填 `COM14`，舵机 ID 为 `1`；Jetson 串口需按实际接线填写。

### 夹爪测试

菜单 `13. 夹爪测试` 走**同一条 115200 bps 直连通路**，使用 `BusServo`：串口与波特率取 `modules/servo/config.json` 的平台 `port` 与 `bus.baudrate`（也可用 `--port` 临时覆盖），舵机 ID 取 `gripper.servo_id`，位置和耗时取 `gripper` 下的 `open_position` / `closed_position` / `duration_ms`。它先读位置，再对张开、闭合分别要求确认（**按回车执行，输入 `q` 结束**），每次动作后读回实际位置。**300/500 只是随意选定的试动值，后续必须按真实夹爪行程修改，不能视为最终张开与夹紧位置。** 旧工程那套 9600 bps 多舵机控制板协议（`ServoController` 与 `Gripper` 类）已随本次重构删除，当前硬件只走直连通路。

完整说明、Jetson 迁移步骤及直连调试记录见 `夹爪控制板使用说明.md`。

### 转盘测试

菜单 `20. 转盘测试` 走**同一条舵机总线**，驱动的是三工位分度转盘（舵机 ID **2**，与夹爪 ID 1
区分）。位置、耗时和工位表取 `modules/servo/config.json` 的 `turntable` 段：

```json
"turntable": {
  "servo_id": 2,
  "duration_ms": 1500,
  "stations": { "home": 0, "station_2": 500, "station_3": 1000 }
}
```

`stations` 是**工位名 → 舵机位置**（0..1000 对应 0..240°），必须有一个叫 `home` 的工位，
`go_home()` 就是走到它。脚本先读当前位置，再按 `home` → 其余工位逐个要求确认：
**按回车执行，输入 `q` 结束，其他输入跳过这一项**。
每次动作后读回核对并打印"到位 / 未到位（差 ±N）"。

```python
from modules.servo.drive.turntable import TurntableServo

with TurntableServo("COM14") as turntable:      # 端口也可从总线配置取
    turntable.go_home()                          # 回零：走到 home 工位
    turntable.go_to_station("station_3")         # 转到三号物料放取位
    print(turntable.at_station("station_3"))     # 读回判断，不是"我发过命令"
```

**转盘的分度盘最怕累计误差，所以这一层只提供绝对位置命令**，没有"相对转多少下"的接口。
**总线舵机上没有"触发回零"这个动作**（它内部一直是绝对位置，不像 Emm42 的计数器掉电即失效），
所以"回零"就是走到 `home` 工位这一个动作；要改"哪是 0"得用舵机的偏差标定（指令 17/18），
属于装机时做一次的事，本层不封装。完整说明见 [modules/servo/docs/转盘控制说明.md](modules/servo/docs/转盘控制说明.md)。

> `0 / 500 / 1000` 是现场给出的三个工位位置（正好相隔 120°），**尚未在真机上逐点核对**。
> 装机后先用菜单 20 走一遍确认，再把实测值回填到 `stations`。

**转盘和机械臂的互锁不在本层**：转盘转动会撞到停在料盘上方的机械臂，而本层不知道机械臂在哪，
所以调用方必须在转动前保证机械臂已让开——和机械臂那边"伸出状态下不许转底座"是同一条规矩。

## D6 激光雷达（lidar）

`lidar` 包封装国科光芯 COIN-D6 的串口指令与点云解析（协议见
`modules/lidar/docs/国科光芯/1 D6激光雷达手册资料/`），并把采集结果画成点云图：

- `modules.lidar.drive.protocol`：帧编解码。启动 `AA 55 F0 0F`、停止 `AA 55 F5 0A`，应答为 12 字节
  `A5 5A ...`；上传帧 `A5 5A LEN CS TYPE DATA`（`0x01` 设备信息、`0x81` 点云）；点云包为
  `AA 55 M&T LSN FSA LSA CS` + 每点 3 字节（14 位距离、8 位强度、高反标志）。读取端会跳过噪声、
  坏校验包并自动重新同步，也同时接受带 `A5 5A` 外层封装与裸 `AA 55` 两种点云流。
- `modules.lidar.CoinD6`：一次串口会话，`initialize()` / `start()` / `read_device_info()` /
  `collect_scan()` / `stop()` / `close()`；`scan(config.scan)` 一次跑完启停与采集。
- `modules.lidar.PointCloud` / `CloudPoint`：点云数据（角度、距离、强度、高反），可 `save_csv()`。
- `modules.lidar.app.visualize`：俯视点云图（按距离着色、等比例、带距离环），保存 PNG 并可弹窗。
- `modules.lidar.app.simulate`：内置虚拟雷达，不接硬件就能跑通「串口 → 解析 → 出图」整条链路。

全部可调参数集中在 `modules/lidar/config.json`，改完存盘即生效，菜单里不再输入端口：

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
    cloud = modules.lidar.scan(config.scan)                # 启动、采 N 圈、停止
cloud.save_csv("output/lidar_cloud/scan.csv")
render_point_cloud(cloud, config.view, "output/lidar_cloud/scan.png")
```

菜单 16 采集一次点云，写出 `output/lidar_cloud/cloud-<时间戳>.csv` 与 `.png`，并按
`visualization.show_window` 弹窗。没有硬件时用内置虚拟雷达验证整条链路：

```powershell
python -m modules.lidar.test.manual.coin_d6 --simulate --no-window
```

该命令不需要串口，会生成一张模拟房间（含圆柱障碍与高反条）的点云图。真机运行前先把
`modules/lidar/config.json` 中的端口改成实际设备名，并确认雷达前方无遮挡。

第一次接真机（或采集一直失败）时先跑诊断：它只发一次启动指令，把**实际发送的原始字节**
和雷达回传的原始字节、解析统计一起打出来，用来区分接线/波特率问题和协议问题。

```powershell
python -m modules.lidar.test.manual.coin_d6 --diagnose
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
python -m modules.lidar.test.manual.coin_d6_live --simulate     # 不接硬件先看界面
python -m modules.lidar.test.manual.coin_d6_live                # 真机（端口取 modules/lidar/config.json）
```

关闭窗口即停止雷达转动并释放串口；真机报错会显示在右下角状态区，不会崩掉窗口。
`modules/lidar/app/live.py` 里的 `LiveCloud` 是这套读数的模型（后台线程 + 滚动缓存 + 快照过滤），
与界面无关，可以直接在别的程序里复用。

离线单测覆盖帧解析、校验、断包重同步、角度换算、扫描窗口过滤与出图，全部不打开串口：

```powershell
python -m unittest modules.lidar.test.unit.test_lidar_protocol modules.lidar.test.unit.test_lidar_device modules.lidar.test.unit.test_lidar_config `
  modules.lidar.test.unit.test_lidar_cloud modules.lidar.test.unit.test_lidar_visualize modules.lidar.test.unit.test_lidar_menu -v
```

`modules/servo/test/manual/gripper.py`（菜单 13）不会在导入时执行动作。夹爪直连调试可运行同一个脚本，逐项按回车确认后依次试动 300 和 500：

```python
import time
from modules.servo.config import load_gripper_config
from modules.servo.drive.bus_servo import BusServo, load_bus_config

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

这段直接读写 `modules/servo/config.json` 中的总线串口与 `gripper` 参数，与菜单 13 是同一条通路。**自动抓取流程**（菜单 19）走的是另一层：`modules/servo/drive/gripper.py` 的 `GripperServo` 把裸位置值收成 `open_jaws()` / `close_jaws()`，并且用位置读回判断"夹住了东西"——夹紧时读回没到闭合位就说明爪口之间有物料。菜单 13 是逐项确认的底层手动调试，两者共用同一条串口与同一份 `gripper` 参数配置。Jetson 使用前需填写 `modules/servo/config.json` 中的 `jetson.port`。

`main.py` 只是一个入口，默认测试物料定位：霍夫圆面判断是否有物料，YOLO 给类别和置信度，窗口显示相机坐标 `x/y/z` 与直线距离。演示代码按模块归位：底盘短动作与地图路线在 `modules/chassis/app/`，物料定位与普通视觉预览在 `modules/vision/app/demo.py`，机械臂在 `modules/arm/app/demo.py`，舵机在 `modules/servo/app/demo.py`；除了物料定位，其余都默认不调用。

```powershell
python main.py
```

预期：打开物料定位窗口，显示 `present`、物料类别、置信度和毫米坐标；按 `p` 在终端打印全部结果，按 `q` 退出。需要连接摄像头并放置当前平台对应的模型文件。

## 本机协议测试

```powershell
python test.py            # 菜单选 1：离线单测（同时覆盖 tests/ 与 modules/ 两个发现根）
```

也可以直接用 unittest，但离线测试分布在两个根目录下的 `test/unit/`，要用 `test.py` 的构建器才能一次跑全：

```powershell
python -c "import test; test.run_offline_tests()"
```

预期：所有测试通过，且无需连接 USB-TTL 或电机。

## 测试目录

测试跟着被测模块走，只有跨切面和入口测试留在顶层 `tests/`：

- `modules/<模块>/test/unit/test_*.py`：离线单元测试，不会打开串口或驱动电机。
- `modules/<模块>/test/manual/`：该模块的真机脚本，从 `test.py` 菜单选择后运行。
- `tests/`：入口测试（`test_main.py`、`test_entrypoints.py`）、跨模块守卫
  （`test_discovery_coverage.py`、`test_project_paths.py`、`test_vision_type_exports.py`、
  `test_python38_compatibility.py`）与共享辅助 `scratch.py`。

## 文档位置

**文档跟着它描述的模块走**，和测试、配置是同一条规矩：

- `modules/<模块>/docs/`：该模块自己的说明文档，以及它那套硬件的厂商手册与例程。
  例如 `modules/arm/docs/`、`modules/grasp/docs/抓取坐标链路.md`、
  `modules/motor/docs/Emm_V5.0步进闭环驱动说明书Rev1.3.pdf`。
- `docs/`（仓库根）：只放**不属于任何单个模块**的材料（跨模块的方案参考等）。

代码和文档里引用其他文档时写**仓库根相对路径**（`modules/grasp/docs/抓取坐标链路.md`），
这样从哪个文件点进去都指向同一个地方，也方便直接 grep。

真机测试从项目根目录运行菜单。开始前让车轮悬空，并确保急停措施可用：

```bash
python test.py
```

预期：选择 3 依次测试四轮正反转；选择 4 测试底盘各方向。两项均按当前平台配置取端口和速度。

所有常用硬件参数集中在 `modules/chassis/config.json`：

- `vehicle.motor_ids`：硬件实际地址到软件轮位的映射；软件 API 始终按 `FL/FR/RL/RR` 逻辑轮位工作。
- `vehicle`：单轮方向修正、加速度、麦轮 X/O 排布和旋转增益；两套平台共用。
- `vehicle.motion_transform`：运动坐标变换，当前实车为 `calibrated`。
- `profiles.jetson`：两类测试的默认速度、时长。
- `profiles.windows`：两类测试的默认速度、时长。

串口不在这里：`modules/chassis/config.json` 已不含 `port`，底盘与机械臂共用 `modules/motor/config.json` 的 `windows.port` / `jetson.port`（步进总线唯一来源）。Jetson 上核对 `modules/motor/config.json` 与 `modules/servo/config.json` 里的设备名。无论正常结束、异常或 `Ctrl+C`，麦轮脚本都会发送停止命令、失能扭矩并关闭串口。
## 视觉外设（vision）

USB 摄像头 + YOLOv8 检测，封装为 `modules.vision` 包，供主程序和示例复用：

- `modules.vision.create_detector(modules.vision.default_model_path())`：Windows 用 ONNX，Jetson 用 TensorRT；都返回 `list[modules.vision.Detection]`。
- `modules.vision.USBCamera`：USB 摄像头封装（`initialize()/read()/release()`，支持 `with`）。
- `modules.vision.open_preview()` / `modules.vision.close_previews()`：HighGUI 预览窗口。**窗口名必须是 ASCII，并且要先显式 `namedWindow` 再 `imshow`**：本机实测中文窗口名会显示成 `??????`（Qt 后端里是 `椤圭洰瑙嗚…` 这样的乱码），不同发行版的 OpenCV 窗口后端（pip 的 Win32UI / conda-forge 的 Qt6）隐式建窗行为也不一样。所以预览入口统一用 `vision-preview` 这个名字。
- `modules.vision.looks_blank(frame)` / `modules.vision.warn_blank_preview(camera_id)`：`read()` 成功但画面是纯黑/纯白时（被别的程序占用、序号指向没有画面的设备、镜头盖关着）打印排查清单，而不是让用户对着一个纯色窗口猜。
- `modules.vision.DistanceEstimator`：单目测距（针孔模型，需要已知类别真实宽度）。

### 物料顶部圆面定位

霍夫圆变换给出物料顶部圆面的圆心与像素半径。检出圆面就视为有物料；YOLO 只补充类别，
漏检时仍返回位置和距离。顶部内侧圆面的实际直径为 30 mm，距离按
`focal_px × 30 / (2 × radius_px)` 计算。未传焦距时，默认使用
`top_pic_2.jpg` 在约 145 mm 实测距离下得到的一次粗标定（1280×720 时约 1012 px）。
完整说明见 [modules/vision/docs/视觉物料定位.md](modules/vision/docs/视觉物料定位.md)。

```python
from vision import MaterialLocator, USBCamera

# 模型路径/类别表/阈值默认全部来自 modules/vision/config.json；
# 要临时换一个模型才传 model_path。
locator = MaterialLocator(model_path="models/best.onnx")
with locator, USBCamera(1, width=1280, height=720).initialize() as camera:
    report = locator.locate(camera.read())
    print(report.summary())  # 圆心、像素直径、约略距离及可选类别
```

菜单 18 是 720p 真机预览入口（`q` 退出，`p` 立刻打印结论）。需提高距离精度时，
先在已知距离处拍摄 30 mm 圆面，按 `focal_px = 已知距离(mm) × 圆面像素直径 / 30`
标定，再通过 `MaterialLocator(focal_px=...)` 或独立命令行入口的 `--focal` 传入。
**相机离物料很近（约 7–12 cm 这种机位）时要配两样东西**：`modules/vision/config.json`
的 `targets.working_distance_mm`（例如 `[55.0, 160.0]`，霍夫半径窗口按针孔模型算）
和 `targets.material_palette`（物料调色板，例如 `{"green": [165,252,114], "blue": [248,122,41]}`；
配了就改走颜色块定位，夹爪圆弧不再被当成物料，**换一块料只加一行**）；
菜单 18 可用 `--depth 60,85`、`--palette "green:165,252,114"` 临时覆盖。
取距离请用 `report.best`（没有通过校验的顶面时是 `None`），不要取 `report.regions[0]`。
详见 [modules/vision/docs/视觉物料定位.md](modules/vision/docs/视觉物料定位.md)。

菜单 9 和 `main.py` 的视觉演示从 `modules/vision/config.json` 读取摄像头编号。Windows 当前设为 `1`（外接 USB 摄像头），Jetson 设为 `0`；如果设备枚举顺序不同，修改对应平台的 `camera_id`。指定编号无法打开时会报错，不会自动切换到前置摄像头。

### 依赖

Windows 使用 OpenCV ONNX 推理，需要 `opencv-python`、`numpy`。Jetson 在 `tensorrt` conda 环境执行，还需要 `tensorrt`、`pycuda`。

### 模型

**用哪个模型由 `modules/vision/config.json` 的 `model` 块决定**，代码里不再写死文件名：

```json
"model": {
  "backend": "auto",
  "onnx": "models/best.onnx",
  "engine": "models/best.engine",
  "class_names": ["red", "yellow", "blue", "green"],
  "materials": null,
  "input_size": 320,
  "conf_thresh": 0.25,
  "iou_thresh": 0.45
}
```

- `backend`：`auto`（默认）按平台选——Windows 走 OpenCV ONNX，Jetson 走 TensorRT；也可以写死 `"onnx"` / `"tensorrt"` 强制一条通道。
- `class_names`：**顺序必须等于训练时的类别顺序**（也就是 ONNX 元数据里的 `names`）。框的坐标由模型给出、类名按这张表查，表错了框还是准的、类名会整体错位。
- `materials`：这张表里哪些算物料；`null` 表示全部算（当前四分类模型四类都是物料）。写进来的名字必须在 `class_names` 里，否则启动就报错。
- `input_size`：方形输入边长；`null` 表示让代码去问模型/引擎自己。**TensorRT 引擎问得到，ONNX 只有装了 `onnx` 才问得到**，所以这里建议显式写。
- 相对路径按**项目根**解析，不按当前工作目录。

模型文件放 `models/`（已 gitignore，不入库）。当前物料模型是四分类：ONNX 输入 `1×3×320×320`，输出 `1×8×2100`（4 个坐标 + 4 类），类别 `{0:red, 1:yellow, 2:blue, 3:green}`，即 1 红 2 黄 3 蓝 4 绿。

`models/best.engine` 是 TensorRT 引擎，**只能在 Jetson 本机生成**（它跟板子的 GPU 架构和 TensorRT 版本绑定，从 PC 拷过去反序列化会失败）。在板上：

```bash
cd ~/gong_chuang
/home/pi/miniforge3/envs/tensorrt/bin/python -m modules.vision.training.build_engine
```

它会读上面那个 `model` 块、用 `trtexec` 构建、然后自动跑一次验收（报张量与耗时）。完整步骤、实测数据和常见故障见 `modules/vision/docs/Jetson模型部署.md`。

换模型的完整流程：训练导出新 ONNX → 改 `model.class_names` / `input_size` → 拷 ONNX 到板上 → `build_engine --force` → `--check --image` 验收。

自己训练障碍物模型：`models/data` 里的图片和标注用 `modules/vision/training/prepare_dataset.py` 切分，`modules/vision/training/train.py` 训练并导出 ONNX，完整流程（参数建议、预训练权重下载、上车注意事项）见 `modules/vision/docs/障碍物模型训练指南.md`。训练完在测试菜单选 `15. 障碍物模型测试`，按提示选择「哪个模型 / 怎么测 / 置信度阈值」即可；它会把类别表和模型绑在一起，选新模型时自动按单类 `obstacle` 解析，不会出现菜单 9 那种类名错位。

### 运行示例

```bash
cd /home/pi/gong_chuang
source /home/pi/miniforge3/bin/activate tensorrt

# 在菜单中选择 9；用的模型读 modules/vision/config.json 的 model 块
# （Windows 是 best.onnx，Jetson 是板上生成的 best.engine）
python test.py
```

预览窗口按 `q` 退出。`modules/vision/test/manual/vision.py` 中的 `focal_px` 默认值（560px）
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
            hits = detector.detect(frame)       # hit: modules.vision.Detection
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
python -m unittest modules.vision.test.unit.test_vision -v
```

离线单测不依赖 TensorRT 与摄像头，只验证预处理、解码、NMS、类名与测距纯计算逻辑。

## 抓取（grasp）

摄像头识别 + 机械臂抓取：俯视相机测出**物料**与**爪上靶标**的位置，换算成机械臂的圆柱
关节量，再驱动三轴和夹爪完成一次抓取。完整说明见 [modules/grasp/docs/抓取坐标链路.md](modules/grasp/docs/抓取坐标链路.md)。

链路上每一级都独立成一个可离线自测的薄模块，依赖方向只能从上到下：

| 层 | 文件 | 职责 |
|---|---|---|
| 编排 | `modules/grasp/drive/pipeline.py` | 安全廊道、闭环微调、异常收尾 |
| 状态 | `modules/grasp/drive/arm_state.py` | 位姿与夹爪状态的唯一真相源（位姿 **+ 它是怎么来的**），跨模块共享 |
| 感知适配 | `modules/grasp/drive/perception.py` | 一帧 → 物料位姿 + 机械臂位姿 |
| 视觉 | `modules/vision/drive/cell_targets.py` | 同一帧里按颜色区分靶标与物料 |
| 针孔模型 | `modules/vision/drive/pinhole.py` | 像素 ↔ 相机系毫米 |
| 平面标定 | `modules/vision/drive/calibration.py` | 相机水平面 → 基座平面（θ 零点 + 半径） |
| 运动学 | `modules/arm/drive/kinematics.py` | 圆柱正逆解、最短转角、mm/度 ↔ 脉冲、可达性 |
| 夹爪 | `modules/servo/drive/gripper.py` | 开合与读回 |

几个刻意的设计决定：

- **中心只有三个标定参数**（基座轴像点 x/y + θ 零点），**没有缩放**——尺度由像素焦距承担，
  再放一个缩放会和焦距互相吞掉。
- **直线轴只需要 `mm_per_motor_rev`**，也就是丝杠导程 `L` 与减速比 `g` 的商；不必拆机量导程。
- **绝对位姿有两个来源，库只给原语，编排留给 main.py**：`0x9A` **回零**给出绝对参考
  （快、不依赖靶标），**相机测爪上靶标**给出实测位姿。什么时候回零、什么时候归位、
  要不要拿两者对一下，都由调用方决定——库里没有任何自动判断。
- **归位靠绝对模式回 0**：`RobotArm.move_all_to_zero()`。回零之后驱动器的位置计数器就在 0，
  不必在上位机累加反向位移。顺带修了驱动的一个限制：绝对模式原先也拒绝 0 脉冲
  （从相对模式抄来的检查），而"回原点"正是绝对模式 0。
- **回零落到"错误的圈"时驱动器不会报错**（单圈编码器，`0x36` 读回来两种情况都是 0）。
  库不做自动校验，只在 `homing_warnings()` 里把不可靠的配置组合提前列出来；
  真要兜住就由调用方拿 `measure_pose()` 和 `safe_pose` 比一下。
- **状态记的是"位姿 + 来源"**（`homed`/`measured`/`dead_reckoned`），不是"位姿 + 一个布尔值"。
  它只记事实、不做判断，但调用方据此就能决定下一步该不该重新对账。
- **夹爪读回值是白送的抓取检测**：夹紧时读回没到闭合位，说明爪口之间有东西
  （`PickResult.held_something`）。
- **转角度前必须先缩回抬升**：伸缩臂在伸出状态下转底座会把整条臂横扫过工作区。
  编排层强制经 `safe_pose` 走一段已知的安全廊道，并且只走最短转角。回零也照这条规矩来：
  顺序固定为**伸缩 → 升降 → 底座**。
- **失败一律不张爪**：中途异常只停轴和失能，松开爪子会把已经抓住的物料掉下去。

配置分成四份：机械常数（含回零模式）在 `modules/arm/config.json`，相机标定结果在
`modules/vision/calibration.json`，目标尺寸颜色在 `modules/vision/config.json`，
**策略参数**（余量、夹持比例、闭环次数与容差）在 `modules/grasp/config.json`。
换一台机器只重标，不改策略。

### 标定

抓取牵涉的标定最多，缺一项都会变成一句难懂的运行时报错，所以菜单 19 会在**打开硬件之前**
先把缺的项一次列全并拒绝运行。实验清单（A0 确认减速比 → A 平面标定 → A′ 重标焦距 →
B/C 标 `mm_per_motor_rev` → D 台面深度 → E 夹爪与 TCP → F 行程与安全位 → G 重复精度与背隙 →
**H 回零零点**（用回零方案时）→ I 到位回包）见 [modules/grasp/docs/抓取坐标链路.md](modules/grasp/docs/抓取坐标链路.md)。

```powershell
python test.py                                             # 菜单选 19
python -m modules.grasp.test.manual.grasp --observe-only   # 只看不动，核对标定
python -m modules.grasp.test.manual.grasp --home-only      # 只回零，验收实验 H
```

> 用"每次用完归位"这条路线时，断电重启后**再跑一次** `--home-only`：单圈回零的多义性只在
> "断电时轴停在别处"时才暴露。手动把底座转过半圈再上电，如果停的位置不对，说明你确实需要
> 限位开关。

### 自测

每层都能单独跑，不需要硬件；编排层用假机械臂 + 假夹爪 + 自洽的观测夹具跑通整条流程：

```powershell
python -m modules.vision.drive.pinhole
python -m modules.vision.drive.calibration
python -m modules.vision.drive.cell_targets
python -m modules.arm.drive.kinematics
python -m modules.servo.drive.gripper
python -m modules.servo.drive.turntable
python -m modules.grasp.drive.arm_state
python -m modules.grasp.drive.pipeline
```

