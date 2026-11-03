# Jetson Orin Nano 电机串口驱动

此目录包含两套 Python 串口协议驱动，以及其上的麦轮小车控制层：

- `modules.servo.drive.bus_servo.BusServo`：幻尔总线舵机，115200 bps、半双工 UART。
- `modules.motor.drive.emm42.Emm42Driver`：Emm42 V5.0 闭环步进驱动器，115200 bps TTL UART。
- `modules.chassis.drive.mecanum.MecanumDrive`：四台 Emm42 驱动器组成的麦轮小车控制层，一组串口即可。

## 配置在哪：硬件 vs 运动调参

每个涉及硬件的模块都有**两份**配置，规则只有一句话：

> **换机器、换接线、重新装机 → 改 `hardware.json`；调手感、调速度 → 改 `config.json`。**

| 模块 | 硬件事实（`hardware.json`） | 运动调参（`config.json`） |
|---|---|---|
| motor | 串口（按平台）、波特率、**脉冲/圈（细分）**、串口超时 | — |
| arm | 三轴电机地址、底座减速比、导程、`expect_ack`、初始机械角（`limits`/`safe_pose`/`homing` 仍被加载校验，但代码已不再使用，见下） | `rpm`、`acceleration`、`default_pulses`、`max_pulses` |
| chassis | 四轮电机地址、轮向 `wheel_signs`、`expect_ack`、`pattern`、`motion_transform` | `acceleration`、`rotation_gain`、各平台演示 profile |
| servo | 总线串口与波特率、两个舵机 ID、夹爪机械端位、转盘**输出行程 240°**、到位容差、工位表 | 夹爪/转盘的 `duration_ms`、转盘停稳余量 |

`pulses_per_revolution` 只有**一份**（`modules/motor/hardware.json` 的 `bus` 段），arm 与 chassis 都读它。

> `modules/arm/hardware.json` 里的 `limits` / `safe_pose` / `homing` 是**标定记录**：绝对坐标那套
> （规划、位姿推算、回零建参考）已经退出，代码不再用它们做运动保护——`limits` 只剩一个用途，
> 就是加载时校验 `initial_motor_degrees` 有没有超出行程；`safe_pose` / `homing` 只校验格式。
> 回零本身用驱动器面板的 `O_Set` 或 `0x9A` 帧（`Emm42Driver.home()`）做，不依赖这几个键。

**没有隐藏默认值**：硬件键缺失会直接报出"哪个文件、哪个键"，而不是退回代码里的常量——
那样"漏写一行"看起来会像"跑通了"。协议不变量（帧头、功能码、字段位宽、0..1000 位置域、
3000 RPM 场上限、50 µs/RPM/档 的加减速斜率）仍然留在代码里，因为它们不随本机变化。

`rpm` 与 `acceleration` 的物理含义：**加速度是斜坡速率**（越大越陡，`0` = 完全不加曲线），
`estimate` 等待模型用的公式是 `加速段 = rpm × (256 − acceleration) × 50µs`。所以在
`acceleration` 很低时，把 `rpm` 调高并不会让动作变快——只会让估算等待变长。

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
实际驱动器地址属于硬件配置，由 `modules/chassis/hardware.json` 的 `vehicle.motor_ids` 注入；当前实车映射为
`FL=3 FR=4 RL=2 RR=1`。更换轮子位置时只需修改配置，不需要修改运动 API 或上层代码。

```python
from modules.chassis import MecanumChassis

with MecanumChassis.open() as car:    # 读配置 + 打开（或借用）步进总线 + 使能四轮
    car.forward(rpm=60)                   # 持续前进，直到 stop()
    car.left(rpm=40, duration=2.0)        # 速度模式：向左平移 2 秒后自动停
    car.rotate_cw(rpm=30, duration=1.0)   # 原地顺时针转 1 秒
    car.forward(rpm=200, pulses=3200)     # 位置模式：四轮各走 3200 脉冲（一圈）后返回
    car.stop()
```

`MecanumChassis` 是上层的整车 API；下面那层 `MecanumDrive`（`car.drive`）持有四个
`Emm42Driver`、逆运动学矩阵和 `inter_frame_seconds`。要单独碰某一个轮子就用
`car.drive.get_wheel(WheelPosition.FL)`。

### 统一运动参数（机械臂 / 底盘一致）

所有整车动作和机械臂动作使用同一套参数名，含义相同（实现见 `modules/motor/drive/motion.py`）：

| 参数 | 含义 | 机械臂 `RobotArm` | 底盘 `MecanumChassis` / `MecanumDrive` |
|---|---|---|---|
| `pulses` | 位置模式：本次走多少脉冲（正数，方向由动作方法决定） | 必填（第一个位置参数） | 可选关键字；给 `pulses` 就走位置模式 |
| `duration` | 速度模式：一条速度命令持续多少秒 | 不支持（机械臂只有相对位置模式） | 可选；与 `pulses` 只能给一个 |
| `rpm` | 目标转速，缺省用配置值 | 可选关键字 | 可选（缺省用 `config.json` 的 `vehicle.rpm`） |
| `acceleration` | 曲线加减速档位 `0..255`，缺省用配置值 | 可选关键字 | 可选 |
| `wait` | `True`（默认）等到动作走完再返回报告；`False` 只发命令、返回**票据** | 默认 `True` | 默认 `True`（位置模式用） |

- **`wait=False` 返回的是票据不是报告**：报告的含义是"这次动作的结论"，而 `wait=False` 的时候还没有结论。票据（`modules.motion.MotionTicket`）的 `estimated_seconds` 就是过去那份报告里同名字段，`.wait()` 才给出报告：`ticket = arm.move_up(pulses=800, wait=False)` → `ticket.wait()`。不调用 `wait` 的话行为与以前完全一样（只发帧，`turns_state` 照旧记账）。
- 需要毫秒级精确时不能用估算：那要用 `0x3A` 读运动状态标志（`&0x02` 到位位）轮询（`RobotArm.open(wait_mode=WAIT_IN_POSITION)`；默认仍走估算，理由见模块文档）。
- 轮子位置模式默认走 Emm42 多机同步：四条 `0xFD` 帧缓存后发 `00 FF 66 6B` 同时启动；0 脉冲的轮子不下发。
- **帧间必须留间隔**（`DEFAULT_INTER_FRAME_SECONDS`，30 ms）：这台机器的 CH340 上背靠背发帧会让后面的帧失效。机械臂逐轴命令、底盘逐轮命令与同步广播都由代码自己插这个间隔，调用方不用管。
- 上层整车 API：`MecanumChassis` 的 `forward` / `backward` / `left` / `right` / `rotate_cw` / `rotate_ccw` / `forward_left` / `forward_right` / `backward_left` / `backward_right`，都可带 `duration`（速度模式）、`pulses`（位置模式，按峰值轮）或 `mm`（位置模式，按车心位移，需标定）或都不带（持续运动）。
- 底盘驱动层（`car.drive`）还有三个通用入口：`drive(vx, vy, omega, rpm)`（速度模式，任意方向）、`drive_pulses(vx, vy, omega, pulses, rpm)` / `drive_mm(vx, vy, mm, rpm)`（位置模式，任意方向）。
- 单轮维护：`car.drive.get_wheel(w)` 拿到 `Emm42Driver`；`enable_wheel` / `set_wheel_speed` / `stop_wheel` 是逐轮命令，`enable_all` / `disable_all` / `stop` 是四轮一起。
- 当前实车使用标定后的麦轮坐标变换 `motion_transform="calibrated"`，四轮横移都会参与运动。
- 若装配后方向反了：整车方向反 → 把 `drive()` 的 `vx`/`vy` 取反；辊子排布不同 → 换 `pattern="O"`；单轮转向反 → 在 `wheel_signs` 给该轮 `-1`。

### 多轴同时启动：`move_axes` 与动作组

机械臂三根轴可以**真的同时起步**（轮子早就是这么走的）。用的是 Emm42 的多机同步三步法：
逐轴发 `0xFD`（多机同步标志 = 01）缓存住 → 帧间留 `DEFAULT_INTER_FRAME_SECONDS`（30 ms，
CH340 上背靠背两条位置帧会双双失效）→ 最后一条广播 `00 FF 66 6B` 让它们一起起步。

```python
with Robot.open(arm=True) as robot:
    robot.arm.move_axes({"lift": 4000, "extension": -8000})                 # 同时起步，一起等
    robot.arm.move_axes({"base": 600, "extension": 200})                    # 底座小角度可与伸缩并行
    robot.arm.move_axes({"base": 5000, "extension": 200}, mode="serial")    # 不许并行时按安全顺序走
```

- `mode="same_start"`（默认）：一起起步，**各自到位**（谁走的路短谁先停）。想让它们同时停要按轴改 rpm（联动/插补），本模块刻意不做。
- `mode="serial"`：按 抬升 → 伸缩 → 底座 → 下降 的安全顺序逐轴走完（与 `_serial_order` 同序）；它只接受 `wait=True`。
- **并行有安全上限**：底座旋转超过 `DEFAULT_PARALLEL_ROTATION_LIMIT_PULSES`（600 脉冲）时不许和伸缩并行——伸出状态下横扫会把整条臂扫过工作区（与 `modules/grasp/drive/visual_servo_motion.py` 同一条规矩）。超了会**在发帧之前**抛错，并提示改用 `mode="serial"`。
- 发帧中途失败时会先补一条广播触发、再逐轴停止：广播触发帧会点燃总线上所有缓存命令，"缓存了却没触发"是个危险状态，不能留着。
- 三根轴是 `Response=None` 的驱动器，连 ACK 都不回，所以缓存帧发完不需要读回包——这也是"同时启动"能做得这么轻的原因。

不同模块之间也可以"各自发出、一起等"：Emm42 与舵机的命令都是发完就自己走，**不需要线程**。
`robot.motion.begin()` 拿一个动作组，`batch.wait()` 是唯一的 join 点：

```python
with Robot.open_all() as robot:
    batch = robot.motion.begin()
    batch.add(robot.arm.move_axes({"lift": 4000, "extension": -8000}, wait=False))
    batch.add(robot.gripper.set_position(40, wait=False))
    batch.add(robot.turntable.go_to_station("station_2", wait=False))
    report = batch.wait()          # 一项失败不阻断其它项；失败时异常上挂着完整报告
```

- 动作组只收**票据**，或带 `estimated_seconds` 的结果对象；两者都没有时用 `batch.add(结果, seconds=预计秒数)` 声明耗时（底盘那类只返回脉冲数的动作走这条）。
- 每张票据都可以单独 `.wait()`，也可以 `.cancel()`（尽力停住、**不收尾**：命令已经发出去了，停在哪里不知道）。
- 票据只负责"到点 + 补报告"，不推算任何位置；绝对位置一律以 `RobotArm.turns_state` 的命令脉冲里程表为准。
- 全程在调用线程上 join，**不建线程**（`modules.motion` 里没有后台线程，也没有锁）。真要"边动边看视觉"，再在 `MotionController` 上加后台 join，调用点不用改。

## 统一入口

在项目根目录运行 `python test.py` 打开交互菜单。选 1 运行离线单测；2–9 为机械臂、麦轮、电机诊断和视觉的手动真机测试；10 为激光障碍物扫描；11 为电机不动的激光固定测距；12 为总线舵机测试；13 为夹爪测试（原来的 14「夹爪直连调试」已合并进来，编号 14 不再使用）；15 为障碍物模型测试（在摄像头、验证集图片、图片文件夹或视频上试新训练的模型，全部按菜单问答选择，不需要命令行参数）；16 为 COIN-D6 激光雷达点云采集；17 为 COIN-D6 实时点云窗口（滑块或输入框调距离与角度窗口）；18 为物料顶部圆面定位（霍夫判断存在并测距，YOLO 补充类别，见 `modules/vision/docs/视觉物料定位.md`）；19 已**归档**（原「摄像头识别 + 机械臂抓取」走的是"相机固定 + 爪上靶标"路线，**本机不适用**，代码与测试移到 `legacy/grasp_fixed_camera/`，菜单不再显示 19）；20 为**转盘测试**（三工位分度盘，见 `modules/servo/docs/转盘控制说明.md`）；21 为**x/y 对准物块 + z 下降**（抓取第一步：先让爪子正下方对准物块，再按固定脉冲数下降到位）。**本机的相机装在机械臂上、跟着爪子动**，所以菜单 21 走的是 eye-in-hand 视觉伺服：把画面里的物块调到固定的目标读数，全程**脉冲驱动**，不需要 `mm_per_motor_rev`、不需要行程标定、也不需要给爪子贴靶标。第一次用先 `--set-target` 记目标读数、再 `--calibrate` 实测雅可比，`--status` 只看不动。平面稳定之后 `--align-z` 走抓取前最后两步：升降（6 号）下降 `z_descend_pulses` 脉冲、伸缩（7 号）后缩 `extension_retract_pulses` 脉冲——两个数都写在 `modules/grasp/eye_in_hand.json` 里，现场直接改、不用重标（`--z-pulses` / `--retract-pulses` 可以只覆盖这一次）；想一次跑完就用 `--align-and-descend`（悬停高度 → 安全高度 → x/y 对准 → 下降 → 后缩，每一段都报耗时）；起始悬停高度也在同一个文件里（`hover_z_mm`，默认与 `safe_lift_z_mm` 同值），就位时会先把爪子伺服到那一层——它决定了开环下降的起点，改它就要重测 `z_descend_pulses`。第 19 项走的是另一套（相机固定 + 爪上靶标），其假设**不适用于本机**；两套做法的分流说明见 `modules/grasp/docs/抓取坐标链路.md` 第 0 节与第 10 节。步进总线端口读取 `modules/motor/config.json` 中当前平台的 `port`（底盘与机械臂共用这一处），舵机总线端口读取 `modules/servo/config.json`；激光测试使用独立的 `modules/stp23l/config.json`，扫描步长和停稳时间分别由 `scan.step_pulses`、`scan.settle_seconds` 设置；D6 激光雷达使用 `modules/coin_d6/config.json`，端口、采集圈数、角度与距离窗口都在里面改。Windows 默认用 `windows`，Jetson 默认用 `jetson`；运行前先把配置中的端口改成实际设备名。真机动作仍会逐项请求确认。

```powershell
python test.py
```

预期：出现编号菜单；输入 `0` 直接退出，不会打开硬件设备。

菜单自动发现各模块 `modules/<模块>/test/manual/` 下声明 `MENU_ENTRY = (编号, "菜单名称")` 且提供 `main()` 的脚本（目录清单见 `test.py` 的 `MANUAL_TEST_DIRS`，编号全局唯一由 `tests/test_discovery_coverage.py` 守住）。新增真机测试时，只需在新脚本中声明未占用的编号和名称，无需修改 `test.py`；展示菜单不会导入真机模块，选择后才导入并运行。

菜单 `12. 总线舵机测试` 走 PDF 中的 **115200 bps 直连总线舵机协议**，直接使用 `modules.servo.drive.bus_servo.BusServo`。端口与波特率从 `modules/servo/hardware.json` 读取（平台串口取 `windows.port` / `jetson.port`，波特率取 `bus.baudrate`），舵机 ID 取 `gripper.servo_id`。测试先只读当前位置；只有查询成功、输入目标脉冲位置和时长，并输入大写 `YES`，才会发送一次运动指令，随后读回实际位置。当前 Windows 端口填 `COM35`；Jetson 串口需按实际接线填写。

### 夹爪测试

菜单 `13. 夹爪测试` 走**同一条 115200 bps 直连通路**，使用 `BusServo`：串口与波特率取 `modules/servo/hardware.json` 的平台 `port` 与 `bus.baudrate`（也可用 `--port` 临时覆盖），舵机 ID 与两个机械端位取同一份文件的 `gripper.servo_id` / `open_position` / `closed_position`，耗时取 `modules/servo/config.json` 的 `gripper.duration_ms`。它先读位置，再对张开、闭合分别要求确认（**按回车执行，输入 `q` 结束**），每次动作后读回实际位置。**当前 200/0 是现场给出的端位，换爪或改装后必须重新确认。** 旧工程那套 9600 bps 多舵机控制板协议（`ServoController` 与 `Gripper` 类）已随本次重构删除，当前硬件只走直连通路。

完整说明、Jetson 迁移步骤及直连调试记录见 `夹爪控制板使用说明.md`。

### 转盘测试

菜单 `20. 转盘测试` 走**同一条舵机总线**，驱动的是三工位分度转盘（舵机 ID **2**，与夹爪 ID 1
区分）。位置与工位表取 `modules/servo/hardware.json` 的 `turntable` 段，耗时取 `modules/servo/config.json`：

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

with TurntableServo.open() as turntable:        # 端口等参数取自 modules/servo/config.json
    turntable.go_home()                          # 回零：走到 home 工位
    turntable.go_to_station("station_3")         # 转到三号物料放取位
    print(turntable.at_station("station_3"))     # 读回判断，不是"我发过命令"
```

**转盘和夹爪共用一条舵机总线**（同一个 COM 口开两次会失败），所以由 `Robot` 开一次、
两个模块各自借用：

```python
with Robot.open(gripper=True, turntable=True) as robot:
    robot.turntable.go_to_station("station_2")
```

**转盘的分度盘最怕累计误差，所以这一层只提供绝对位置命令**，没有"相对转多少下"的接口。
**总线舵机上没有"触发回零"这个动作**（它内部一直是绝对位置，不像 Emm42 的计数器掉电即失效），
所以"回零"就是走到 `home` 工位这一个动作；要改"哪是 0"得用舵机的偏差标定（指令 17/18），
属于装机时做一次的事，本层不封装。完整说明见 [modules/servo/docs/转盘控制说明.md](modules/servo/docs/转盘控制说明.md)。

> `0 / 500 / 1000` 是现场给出的三个工位位置（正好相隔 120°），**尚未在真机上逐点核对**。
> 装机后先用菜单 20 走一遍确认，再把实测值回填到 `stations`。

**转盘和机械臂的互锁不在本层**：转盘转动会撞到停在料盘上方的机械臂，而本层不知道机械臂在哪，
所以调用方必须在转动前保证机械臂已让开——和机械臂那边"伸出状态下不许转底座"是同一条规矩。

## D6 激光雷达（coin_d6）

`coin_d6` 包封装国科光芯 COIN-D6 的串口指令与点云解析（协议见
`modules/coin_d6/docs/国科光芯/1 D6激光雷达手册资料/`），并把采集结果画成点云图：

- `modules.coin_d6.drive.protocol`：帧编解码。启动 `AA 55 F0 0F`、停止 `AA 55 F5 0A`，应答为 12 字节
  `A5 5A ...`；上传帧 `A5 5A LEN CS TYPE DATA`（`0x01` 设备信息、`0x81` 点云）；点云包为
  `AA 55 M&T LSN FSA LSA CS` + 每点 3 字节（14 位距离、8 位强度、高反标志）。读取端会跳过噪声、
  坏校验包并自动重新同步，也同时接受带 `A5 5A` 外层封装与裸 `AA 55` 两种点云流。
- `modules.coin_d6.CoinD6`：一次串口会话，`initialize()` / `start()` / `read_device_info()` /
  `collect_scan()` / `stop()` / `close()`；`scan(config.scan)` 一次跑完启停与采集。
- `modules.coin_d6.PointCloud` / `CloudPoint`：点云数据（角度、距离、强度、高反），可 `save_csv()`。
- `modules.coin_d6.app.visualize`：俯视点云图（按距离着色、等比例、带距离环），保存 PNG 并可弹窗。
- `modules.coin_d6.app.simulate`：内置虚拟雷达，不接硬件就能跑通「串口 → 解析 → 出图」整条链路。

全部可调参数集中在 `modules/coin_d6/config.json`，改完存盘即生效，菜单里不再输入端口：

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
from modules.coin_d6 import CoinD6, load_lidar_config, render_point_cloud

config = load_lidar_config()                       # 也可传 "windows" / "jetson"
with CoinD6(config.device.port, baudrate=config.device.baudrate) as lidar:
    cloud = lidar.scan(config.scan)                          # 启动、采 N 圈、停止
cloud.save_csv("output/lidar_cloud/scan.csv")
render_point_cloud(cloud, config.view, "output/lidar_cloud/scan.png")
```

菜单 16 采集一次点云，写出 `output/lidar_cloud/cloud-<时间戳>.csv` 与 `.png`，并按
`visualization.show_window` 弹窗。没有硬件时用内置虚拟雷达验证整条链路：

```powershell
python -m modules.coin_d6.test.manual.coin_d6 --simulate --no-window
```

该命令不需要串口，会生成一张模拟房间（含圆柱障碍与高反条）的点云图。真机运行前先把
`modules/coin_d6/config.json` 中的端口改成实际设备名，并确认雷达前方无遮挡。

第一次接真机（或采集一直失败）时先跑诊断：它只发一次启动指令，把**实际发送的原始字节**
和雷达回传的原始字节、解析统计一起打出来，用来区分接线/波特率问题和协议问题。

```powershell
python -m modules.coin_d6.test.manual.coin_d6 --diagnose
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
python -m modules.coin_d6.test.manual.coin_d6_live --simulate     # 不接硬件先看界面
python -m modules.coin_d6.test.manual.coin_d6_live                # 真机（端口取 modules/coin_d6/config.json）
```

关闭窗口即停止雷达转动并释放串口；真机报错会显示在右下角状态区，不会崩掉窗口。
`modules/coin_d6/app/live.py` 里的 `LiveCloud` 是这套读数的模型（后台线程 + 滚动缓存 + 快照过滤），
与界面无关，可以直接在别的程序里复用。

离线单测覆盖帧解析、校验、断包重同步、角度换算、扫描窗口过滤与出图，全部不打开串口：

```powershell
python -m unittest modules.coin_d6.test.unit.test_lidar_protocol modules.coin_d6.test.unit.test_lidar_device modules.coin_d6.test.unit.test_lidar_config `
  modules.coin_d6.test.unit.test_lidar_cloud modules.coin_d6.test.unit.test_lidar_visualize modules.coin_d6.test.unit.test_lidar_menu -v
```

`modules/servo/test/manual/gripper.py`（菜单 13）不会在导入时执行动作。夹爪直连调试可运行同一个脚本，逐项按回车确认后依次试动 300 和 500：

```python
import time
from modules.servo.config import load_gripper_config
from modules.servo.drive.bus_servo import BusServo, load_servo_bus_config

config = load_gripper_config()
bus = load_servo_bus_config()
servo = BusServo(bus.port, config.servo_id, baudrate=bus.baudrate, timeout=0.5).initialize()
try:
    servo.set_position(config.open_position, config.duration_ms)
    time.sleep(config.duration_ms / 1000 + 0.2)
    print("张开后读回：", servo.read_position())
finally:
    servo.close()  # 只关闭串口，不会改变夹爪位置
```

这段直接读写 `modules/servo/config.json` 中的总线串口与 `gripper` 参数，与菜单 13 是同一条通路。**抓取流程**走的是另一层：`modules/servo/runtime.py` 的 `GripperServo` 把裸位置值收成 `open_jaws()` / `close_jaws()` / `set_position()`，并且用位置读回判断"夹住了东西"——夹紧时读回没到闭合位就说明爪口之间有物料。菜单 13 是逐项确认的底层手动调试，两者共用同一条串口与同一份 `gripper` 参数配置。Jetson 使用前需填写 `modules/servo/config.json` 中的 `jetson.port`。

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
  `test_python38_compatibility.py`）、跨模块动作组验收（`test_motion_batch.py`）与共享辅助 `scratch.py`。

## 文档位置

**文档跟着它描述的模块走**，和测试、配置是同一条规矩：

- `modules/<模块>/docs/`：该模块自己的说明文档，以及它那套硬件的厂商手册与例程。
  例如 `modules/arm/docs/`、`modules/grasp/docs/抓取坐标链路.md`、
  `modules/motor/docs/Emm_V5.0步进闭环驱动说明书Rev1.3.pdf`。
- `docs/`（仓库根）：只放**不属于任何单个模块**的材料（跨模块的方案参考等）。
- **不新增文档，除非用户明确要求。** 能写进已有文档的就写进它对应的那一节，不要在
  `docs/` 或 `modules/*/docs/` 下另起一个文件来"补充说明"；改了代码就同步更新受影响
  的**已有**文档。新增文件（文档或说明）只有在用户明确要求时才做。

代码和文档里引用其他文档时写**仓库根相对路径**（`modules/grasp/docs/抓取坐标链路.md`），
这样从哪个文件点进去都指向同一个地方，也方便直接 grep。

真机测试从项目根目录运行菜单。开始前让车轮悬空，并确保急停措施可用：

```bash
python test.py
```

预期：选择 3 依次测试四轮正反转；选择 4 测试底盘各方向。两项均按当前平台配置取端口和速度。

所有**运动调参**集中在 `modules/chassis/config.json`（`vehicle.acceleration` / `rotation_gain` 与各平台演示 profile）；四轮地址、轮向、`expect_ack`、辊子排布与标定矩阵选择器在 `modules/chassis/hardware.json`：

- `vehicle.motor_ids`：硬件实际地址到软件轮位的映射；软件 API 始终按 `FL/FR/RL/RR` 逻辑轮位工作。
- `vehicle`：单轮方向修正、加速度、麦轮 X/O 排布和旋转增益；两套平台共用。
- `vehicle.motion_transform`：运动坐标变换，当前实车为 `calibrated`。
- `profiles.jetson`：两类测试的默认速度、时长。
- `profiles.windows`：两类测试的默认速度、时长。

串口不在这里：`modules/chassis/hardware.json` 不含 `port`，底盘与机械臂共用 `modules/motor/hardware.json` 的 `windows.port` / `jetson.port`（步进总线唯一来源）。Jetson 上核对 `modules/motor/hardware.json` 与 `modules/servo/hardware.json` 里的设备名。无论正常结束、异常或 `Ctrl+C`，麦轮脚本都会发送停止命令、失能扭矩并关闭串口。
## 视觉外设（vision）

USB 摄像头 + YOLOv8 检测，封装为 `modules.vision` 包，供主程序和示例复用：

- `modules.vision.create_detector(modules.vision.default_model_path())`：Windows 用 ONNX，Jetson 用 TensorRT；都返回 `list[modules.vision.Detection]`。
- `modules.vision.USBCamera`：USB 摄像头封装（`initialize()/read()/release()`，支持 `with`）。
- `modules.vision.run_preview(camera, 推理函数, 画图函数, window=...)`：**预览循环**。取帧一条线程、推理一条线程，窗口跑在摄像头帧率上；推理跟不上的帧被跳过而不是拖慢显示。窗口左下角写着 `cap / det / ms / age`（摄像头帧率 / 识别帧率 / 最近一次推理耗时 / 框比画面旧多少毫秒）——"画面卡"和"识别慢"从此是两个可分辨的数。需要"结果与画面严格同帧"的抓取链路请**不要**用它，直接同步调 `locate()` / `detect()`。
- `modules.vision.open_preview()` / `modules.vision.close_previews()`：HighGUI 预览窗口。**窗口名必须是 ASCII，并且要先显式 `namedWindow` 再 `imshow`**：本机实测中文窗口名会显示成 `??????`（Qt 后端里是 `椤圭洰瑙嗚…` 这样的乱码），不同发行版的 OpenCV 窗口后端（pip 的 Win32UI / conda-forge 的 Qt6）隐式建窗行为也不一样。所以预览入口统一用 `vision-preview` 这个名字。
- `modules.vision.looks_blank(frame)` / `modules.vision.warn_blank_preview(camera_id)`：`read()` 成功但画面是纯黑/纯白时（被别的程序占用、序号指向没有画面的设备、镜头盖关着）打印排查清单，而不是让用户对着一个纯色窗口猜。
- `modules.vision.DistanceEstimator`：单目测距（针孔模型，需要已知类别真实宽度）。

### 物料定位：YOLO 主算，圆面辅助

**主算**用 YOLO 的物料框：框中心当物料中心、框的等效半径 `(w+h)/4` 当顶面半径，
一个像素的霍夫都不跑（本机实测一帧 **27–29 ms**）。**辅助**是原来的"颜色块 / 霍夫找顶面圆"
那套（实测一帧 357–668 ms），**只在主算一个合格物料框都给不出来时才执行**；
它现在住在一个可以整个换掉的方法里（`MaterialLocator.locate_auxiliary`），主流程不关心它内部怎么做。

| 路 | 什么时候跑 | 读数从哪来 | 一帧实测 |
|---|---|---|---|
| 主算 | 有合格物料框（常态） | 框中心 + 框的等效半径 | 27–29 ms |
| 辅助 | 一个合格框都没有 | 顶面圆的圆心 + 像素半径（30 mm 直径按针孔模型换算） | 357–668 ms |

**框比顶面大**——同色桌面、可见的侧面都会进框（实测一帧里 30 mm 顶面是 253 px 直径，
模型给的框是 401×439 px），所以由框反推的深度偏近。这**不影响 x/y 对准**：视觉伺服在
同一套读数里闭环，尺度的系统偏差会被实测出来的雅可比吸收掉；但 `report.best.z_mm`
不是真实距离。报告里 `colour_source == "yolo"` 就表示这一帧的读数是框给的。

**画面里有多种物料时**，`locate()` 一个参数就能指定要哪一个，抓取侧同样按次传：

```python
report = vision.locate("red")     # 只要红的；也可以传编号，如 locate(2)
...
robot.grasp.align_xy(reading=(report.best.x_mm, report.best.y_mm), want="red")
robot.grasp.align_and_descend(want="red")     # 整轮就位（悬停 → x/y → 下降 → 后缩）
```

`want` 与 `vision.locate(want)` 是**同一套说法**（类别名，或从 1 开始的编号），按次生效——
一次程序里依次抓红、绿、蓝就靠它，不用改任何配置文件。指定的那一种不在画面里时
`report.best is None`（**不回退**到别的物料：拿蓝的顶上等于让机械臂朝错误的目标动）。

`eye_in_hand.json` 的 `object` 是**没传 `want` 时才用**的默认值，留给菜单/工具那些
没有按次传参的入口。两个都不给时，物块被定义成"画面里唯一的目标"；主算路径下这和
"每一个物料框都是一条观测"放在一起，就意味着画面里多于一个目标会直接报错。

```python
from modules.vision import Vision

# 配置路径、模型、类别表、阈值、摄像头序号全部来自 modules/vision/config.json；
# 要临时换一个模型/相机才传对应参数（只作用于这一次，不写回文件）。
with Vision.open(config_path="modules/vision/config.json") as vision:
    report = vision.locate()
    if report.best is not None:
        print(report.best.x_mm, report.best.y_mm, report.best.z_mm)
    print(report.summary())  # 圆心、像素直径、约略距离及可选类别
```

只想在一帧图像上跑（不打开相机）时，用帧级定位器：

```python
from modules.vision.drive.camera import USBCamera
from modules.vision.drive.material_locate import MaterialLocator

# 模型路径/类别表/阈值默认全部来自 modules/vision/config.json。
locator = MaterialLocator(model_path="models/best.onnx")
with locator, USBCamera(1, width=1280, height=720).initialize() as camera:
    report = locator.locate(camera.read())
    print(report.summary())
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

### 工位识别：三圆环、叠放位、两点测量

物料定位回答"料在哪"，这几项回答**别的**问题：工位摆正了没有、画面上两点多远、
转盘到底转过没有。它们共用同一个 `Vision` 会话与同一份 `config.json`，但判据与
物料定位**没有一处共用**，所以是独立检测器。完整说明见
[modules/vision/docs/工位识别.md](modules/vision/docs/工位识别.md)。

| 能力 | 会话入口 | 真机菜单 | 离线自测 |
|---|---|---|---|
| 三圆环工位识别 | `vision.detect_ring_station()` | 22 | `python -m modules.vision.drive.ring_station` |
| 叠放位识别 | `vision.detect_stacked_ring()` | 23 | `python -m modules.vision.drive.stacked_ring` |
| 两点像素测量 | `vision.pixel_distance()` / `pixel_distance_mm()` | 24 | `python -m modules.vision.drive.pixel_measure` |
| 停稳门控 | `vision.measure_settled(require_motion=True)` | — | `python -m modules.vision.drive.motion_gate` |
| 焦距 / 台面深度标定 | — | 25 | — |

```python
with Vision.open() as vision:
    station = vision.detect_ring_station()      # 圆心、偏角、毫米偏差
    print(station.summary())
    stacked = vision.detect_stacked_ring()      # 中环 + 上下基准线
    print(stacked.summary())
    print(vision.pixel_distance((100, 200), (400, 600)).summary())
```

**和 `locate()` 的语义差异是有意的**：`locate()` 读不到画面就抛 `RuntimeError`（业务失败）；
工位识别**画面里没有工位是正常结果**（`detected is False`，`detail` 说明原因），
而读不到画面仍然抛 `RuntimeError`——两种情况不该用同一个返回值表达。

三件值得知道的取舍：

- **没配比例尺时不编毫米**。`station.mm_per_px` 与 `station.image_to_station_affine`
  都空时只报像素，毫米字段是 `None` 而**不是 0**（0 看起来像"完全对准了"），
  而且 `within_tolerance` 恒为 `False`。
- **静止 ≠ 停稳**。`measure_settled(require_motion=True)` 要求先**看见它真的动过**，
  否则转盘还没转，上一轮的残留状态就会被当成这一轮的结果上报。
- **叠放位的尺度可以不可信**。没有标定时会用中环半径临时反推，结果的
  `scale_source` 会写成 `"ring"`、`scale_trustworthy` 为 `False`。

### 标定：先把 calibration.json 填上（菜单 25）

`modules/vision/calibration.json` 里 `focal_px`、`table_depth_mm`、`plane` 默认全是 `null`，
所以**所有毫米读数用的都是粗标定兜底值，不是真实距离**（物料定位那节的
"框比顶面大"也是同一个根源）。菜单 25 补前两项：把已知直径的圆片放在已知距离处，
按 `f` 点它的左右边缘，输入真实直径与距离即可；按 `c` 看还缺什么，按 `s` 保存焦距平均值，
按 `d` 直接设置台面深度。它**只写 `calibration.json`，不碰 `config.json`**。

### 辅助路径的六色调色板

主算的 YOLO 模型是七类（四色物料红/黄/蓝/绿 + 三个靶标 `tg1`/`tg2`/`tg3`），
其中**只有四种颜色算物料**；而 `targets.material_palette` 配了**六色**
（多出 `black` 与 `light_blue`）：主算认不出来的料，辅助路径按颜色块还能认。
换一块料只加一行调色板，不必重训模型——`material_class.DEFAULT_MATERIAL_LABELS`
里补上对应中文即可（缺标签会落到"未知物料"）。

### 预览帧率：慢的是经典 CV，不是 YOLO 也不是摄像头

本机实测（1280×720）：摄像头 27–30 fps、YOLO 推理 17–20 ms、`imshow` 3.5–7 ms，
而 `MaterialLocator.locate()` 在真实台面图上是 **0.5–3.3 s**——原来那条串行循环
（取帧→推理→显示）于是被最慢的一段拖到 0.3–2 fps，还因为取帧不停而越积越延迟。
现在取帧与推理各跑一条线程，窗口跑在摄像头帧率上；同时去掉了每帧的重复计算
（每帧转一次灰度图而不是每个候选转一次、模型一帧只问一次、轮廓抽样后再算外接圆），
`locate()` 本身快了 **2.9–7.9 倍**（例如 `top_pic.png` 954 ms → 131 ms）。
完整数据、还嫌慢时的四个调节开关、以及**刻意没动**的判据都在
[modules/vision/docs/预览帧率.md](modules/vision/docs/预览帧率.md)。

菜单 9、菜单 15、菜单 18 和 `main.py` 的物料预览都走同一个 `run_preview`：
窗口左下角那行 `cap / det / ms / age` 是取帧帧率、识别帧率、最近一次推理耗时、
以及画面上的框比当前画面旧了多少毫秒。**`det` 只有几 fps 而 `cap` 是 30 fps 时，
画面仍然是流畅的**——慢的是识别，不是显示；真要看识别够不够快，看 `det`。

菜单 9 和 `main.py` 的视觉演示从 `modules/vision/config.json` 读取摄像头编号，物料预览还会读上面那三个 `targets` 参数（与菜单 18 一致）。Windows 当前设为 `1`（外接 USB 摄像头），Jetson 设为 `0`；如果设备枚举顺序不同，修改对应平台的 `camera_id`。指定编号无法打开时会报错，不会自动切换到前置摄像头。

### 依赖

Windows 使用 OpenCV ONNX 推理，需要 `opencv-python`、`numpy`。Jetson 在 `tensorrt` conda 环境执行，还需要 `tensorrt`、`pycuda`。

### 模型

**用哪个模型由 `modules/vision/config.json` 的 `model` 块决定**，代码里不再写死文件名：

```json
"model": {
  "backend": "auto",
  "onnx": "models/best.onnx",
  "engine": "models/best.engine",
  "class_names": ["red", "yellow", "blue", "green", "tg1", "tg2", "tg3"],
  "materials": ["red", "yellow", "blue", "green"],
  "input_size": 320,
  "conf_thresh": 0.25,
  "iou_thresh": 0.45
}
```

- `backend`：`auto`（默认）按平台选——Windows 走 OpenCV ONNX，Jetson 走 TensorRT；也可以写死 `"onnx"` / `"tensorrt"` 强制一条通道。
- `class_names`：**顺序必须等于训练时的类别顺序**（也就是 ONNX 元数据里的 `names`）。框的坐标由模型给出、类名按这张表查，表错了框还是准的、类名会整体错位。
- `materials`：这张表里哪些算物料；`null` 表示全部算。**七类模型必须显式写这四个颜色**：`tg*` 是带数字的靶标，混进物料会让抓取链路去抓靶标。写进来的名字必须在 `class_names` 里，否则启动就报错。
- `input_size`：方形输入边长；`null` 表示让代码去问模型/引擎自己。**TensorRT 引擎问得到，ONNX 只有装了 `onnx` 才问得到**，所以这里建议显式写。
- 相对路径按**项目根**解析，不按当前工作目录。

模型文件放 `models/`（已 gitignore，不入库）。当前模型是**七类**（2026-10-20 训练，"四色 + 三个靶标"）：ONNX 输入 `1×3×320×320`，输出 `1×11×2100`（4 个坐标 + 7 类），类别 `{0:red, 1:yellow, 2:blue, 3:green, 4:tg1, 5:tg2, 6:tg3}`。`tg1`/`tg2`/`tg3` 是环纸上的三个靶标，菜单 26「环内数字环心」就是靠它们直接读出环心与环号。

> 旧的四类模型留了一份备份：`models/best_4color_20261013.onnx`。换回它在配置里把 `class_names` 改回四个颜色、`materials` 写成 `null` 即可。


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

**本机的相机装在机械臂上（eye-in-hand）**，所以走的是**脉冲闭环视觉伺服**：把画面里的
物块读到固定的目标读数，全程不需要毫米级世界坐标。世界坐标那条链（相机系 → 基座平面 →
圆柱关节量、爪上靶标）属于"相机固定"路线，**本机不适用**，代码在
`legacy/grasp_fixed_camera/`。

```python
from modules.grasp import Grasp

with Grasp.open() as grasp:        # 读配置 + 开机械臂 + 开视觉源（Vision 或 CameraLink）
    result = grasp.align_xy()      # 只动平面：把物块伺服到目标读数
```

| 层 | 文件 | 职责 |
|---|---|---|
| 会话 / 编排 | `modules/grasp/runtime.py` | `Grasp`：拥有机械臂与视觉源；悬停 → 安全抬升 → x/y → 下降 → 后缩 |
| 伺服算法 | `modules/grasp/drive/eye_in_hand.py` | 误差 → 脉冲步长，逐帧闭环 |
| 数值模型 | `modules/grasp/drive/visual_servo_model.py` | 配置、雅可比、结果与异常类型 |
| 观测判据 | `modules/grasp/drive/visual_servo_observation.py` | "这一帧够不够新、够不够稳" |
| 脉冲原语 | `modules/grasp/drive/visual_servo_motion.py` | 单轴 / 多轴脉冲下发 |
| 环心计划 | `modules/grasp/drive/ring_plan.py` | 环心 → 各轴**绝对电机角度**（只算不动） |

几个要点：

- **一个机器常数都不用标**：伺服要的是"每脉冲让画面挪多少毫米"（2×2 雅可比，实测），
  不是 `mm_per_motor_rev`。平面一旦对准，剩下的两步就是固定脉冲数
  （`z_descend_pulses` / `extension_retract_pulses`）。
- **雅可比是实测的**（`--calibrate`），相机怎么装、歪多少度都不用假设。它是线性的，
  所以每一步都用**新的一帧**重算，大位移靠闭环而不是"理论上一步到位"。
- **"新的一帧"是硬要求**：动作前记一个时刻，动作后只认晚于它的观测（`min_received_at`）。
  用"多久算旧"挡不住"动作很快、旧帧也还新"，那时读到的是动作**之前**的坐标。
- **绝对位置只到电机角度为止**：`ring_plan` 给的是"底座 / 伸缩走到哪个电机位置"
  （3200 脉冲 = 360°，不取模），口径与 `RobotArm.move_axis_to_motor_degrees` 一致；
  它用的是 `turns_state` 命令里程表，所以中间**不能回零**（回零会清零里程表）。
- **失败一律不张爪**：中途异常只停轴和失能，松开爪子会把已经抓住的物料掉下去。

### 现场调参与标定

所有参数在 `modules/grasp/eye_in_hand.json`：目标读数 `target`、实测雅可比 `jacobian`、
容差与步长上限、下降 / 后缩脉冲、安全抬升那一组。现场改这个文件，不改代码。

```powershell
python test.py                                                      # 菜单 21：x/y 对准（含一键就位）
python -m modules.grasp.test.manual.align_xy --set-target           # 记目标读数
python -m modules.grasp.test.manual.align_xy --calibrate            # 实测雅可比
python -m modules.grasp.test.manual.align_xy --status               # 只看不动
python -m modules.grasp.test.manual.align_xy --align-and-descend    # 悬停 → 对准 → 下降 → 后缩
```

环心 → 各轴绝对位置走菜单 27（`modules.grasp.test.manual.ring_angle`）；环内数字环心是
菜单 26（视觉侧）。两套都在开硬件之前把缺的标定一次列全并拒绝运行。

### 自测

```powershell
python -m modules.grasp.drive.ring_plan      # 环心 → 绝对角的解算，纯计算
```

其余各层由 `modules/grasp/test/unit/` 覆盖（假机械臂 + 假视觉源，不需要硬件）。
