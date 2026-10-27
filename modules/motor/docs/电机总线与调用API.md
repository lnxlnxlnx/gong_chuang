# 电机总线控制与调用 API 文档

> 适用对象：`gong_chuang/` 下两套串口驱动 —— **Emm42 V5.0 闭环步进驱动器**（麦轮底盘 + 机械臂用）
> 和 **幻尔总线舵机**（夹爪 / 转盘用）。
>
> 本文是"照着写代码"的技术文档：先讲总线如何区分电机、如何找电机 ID，
> 再给各层的完整函数签名与可直接复制的调用示例。
> 协议级帧格式速查见根目录 `电机舵机驱动命令速查.md`。

---

## 1. 总体架构：一条总线 + 三层代码

```
┌─ 应用代码（你写的）──────────────────────────────────────┐
│  from modules.chassis import MecanumChassis               │
│  with MecanumChassis.open() as car:                       │
│      car.forward(rpm=60)                                  │
└──────────────────────────┬───────────────────────────────┘
                           │ 第 3 层 底盘门面 MecanumChassis (modules/chassis/runtime.py)
┌──────────────────────────▼───────────────────────────────┐
│ 第 2 层 整车控制 MecanumDrive (modules/chassis/drive/mecanum.py)  │
│   · 四台 Emm42Driver 共享同一个 serial 对象（分时复用总线）  │
│   · drive() 逆运动学矩阵算出四轮转速，逐轮发 F6 帧             │
│   · 逐轮发帧之间留 inter_frame_seconds（CH340 的脾气）         │
└──────────────────────────┬───────────────────────────────┘
                           │ 第 1 层 协议驱动 (modules/{motor,servo}/drive/)
┌──────────────────────────▼───────────────────────────────┐
│ Emm42Driver ── 按 motor_id 组帧/解 ACK, 写共享串口          │
│ BusServo    ── 按 servo_id 组帧/解 ACK                     │
└──────────────────────────┬───────────────────────────────┘
                           │ 一根 USB-TTL (115200, 8N1)
                    Emm42 总线（全双工 TTL，多机并接）
```

- **第 1 层（协议）**：`modules/motor/drive/emm42.py` `Emm42Driver`、`modules/servo/drive/bus_servo.py` `BusServo`。
- **第 2 层（整车）**：`modules/chassis/drive/mecanum.py` `MecanumDrive`，四轮共享一根串口，含麦轮逆运动学。
- **第 3 层（门面）**：`modules/chassis/runtime.py` `MecanumChassis`，向上层应用提供的"整车动作" API。
- **共享总线**：arm 与 chassis 用**同一根** Emm42 总线，由 `Robot` 打开一次、两边借用
  （谁开的谁关）。所以"总线上有别人的帧"是常态，回包匹配规则见第 4 节。
- **配置**：串口在 `modules/motor/hardware.json`（`bus` 段 + 各平台 `port`）；轮位地址、
  轮向、标定矩阵选择器在 `modules/chassis/hardware.json`；速度/加速度档在
  `modules/chassis/config.json`。

> 总线舵机只到第 1 层（没有整车层），用 `BusServo` 直接控制；夹爪/转盘在它上面各自包了一层
> `GripperServo` / `TurntableServo`。

---

## 1.1 统一运动参数（机械臂与底盘一致）

第 2、3 层的运动方法都使用同一套参数名，含义完全相同（算法在 `modules/motor/drive/motion.py`）：

| 参数 | 含义 | 机械臂 `RobotArm` | 底盘 `MecanumChassis` / `MecanumDrive` |
|---|---|---|---|
| `pulses` | 位置模式：本次移动的脉冲数（正数，方向由动作方法决定） | 必填（第一个位置参数） | 可选关键字；给了就走位置模式 |
| `mm` | 位置模式：**车心**沿运动方向走多少毫米（底盘独有，需 `mm_per_wheel_rev` 标定） | — | 可选关键字；与 `pulses`/`duration` 互斥 |
| `duration` | 速度模式：一条速度命令持续多少秒 | 不支持（只发相对位置帧） | 可选；与 `pulses`/`mm` 互斥 |
| `rpm` | 目标转速，缺省用配置值 | 可选关键字 | 可选（缺省用 `config.json` 的 `vehicle.rpm`） |
| `acceleration` | 曲线加减速档位 `0..255`，缺省用配置值 | 可选关键字 | 可选 |
| `wait` | `True`（默认）等到动作走完再返回，`False` 只发命令并返回票据 | 默认 `True` | 位置模式默认 `True` |

**为什么需要 `wait`**：Emm42 位置命令是"发完就返回"，驱动器不会告诉上位机什么时候停。
不等待时，下一条位置帧会在上一条还没走完时改写目标，现象就是"上一个动作被跳过"。
`wait=True` 按说明书给的曲线加减速公式估算耗时后等待：

```
每档加速度让转速 +1 RPM 约需 (256 - acceleration) * 50µs
总耗时 ≈ rpm * (256 - 档位) * 50µs  +  脉冲数 / (rpm * 脉冲每圈 / 60)  +  到位余量(默认 0.15s)
```

这是估算（把减速段和串口往返折进到位余量），**不是编码器实测**。要实测就切轮询模式
（`RobotArm.open(wait_mode=WAIT_IN_POSITION)`，读 `0x3A` 的 `&0x02` 到位位），
`Emm42Driver.wait_until_in_position()` 就是那条路的原语。
`MotionReport.estimated_seconds` 会带回本次估算值，`wait=False` 时调用方可以自己拿它去等
（或者交给 `robot.motion.begin()` 的动作组）。

**帧间必须留间隔**：这台机器的 CH340 上背靠背发帧会让后面的帧失效，
所以机械臂逐轴命令、底盘逐轮命令、以及多机同步的缓存帧之间都是
`DEFAULT_INTER_FRAME_SECONDS`（30 ms）。这个间隔由代码自己插，调用方不用管。

---

## 2. 总线如何区分各台电机：帧首地址

Emm42 是**共享总线 + 帧内寻址**：

```
         ┌────── 同一根 TTL 总线（RX/TX 并接）──────┐
  USB-TTL │   ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐
  TX──R/A/H┤  │ 电机#1  │  │ 电机#2  │  │ 电机#3  │  │ 电机#4  │
  RX──T/B/L┤  └────────┘  └────────┘  └────────┘  └────────┘
         └──────────────────────────────────────────┘
  每帧: [地址] [功能码] [数据...] [0x6B]
```

- 总线上的每台电机都能收到**每一帧**；
- 帧首地址与某台电机自身 ID 相等时，**只有这台执行命令并回 ACK**，其余忽略；
- 速度帧结构（方向 + 速度就藏在这里）：

```
字段:   addr   func     dir      rpm(2B 大端)   accel    sync     tail
示例:   01     F6       01       05 DC(1500)    0A       00       6B
含义:   地址1  速度模式  CCW反转  1500 RPM       加速度档10  立即执行  帧尾/校验
```

| 字段 | 长度 | 说明 |
|---|---|---|
| `addr` | 1B | 驱动器地址 **1~255**；`0x00` = 广播（所有电机执行，**四台各自用自己的地址回话**） |
| `func` | 1B | `F6`=速度模式、`FD`=位置模式、`F3 AB`=使能、`FE 98`=停止、`FF 66`=同步触发 |
| `dir` | 1B | **`00`=CW（正转）`01`=CCW（反转）** |
| `rpm` | 2B 大端 | 0~3000 |
| `accel` | 1B | 0~255，`0`=无加减速曲线；档位越大加速越快 |
| `sync` | 1B | `0`=立即执行；`1`=缓存住，等 `00 FF 66 6B` 同步帧一起启动 |
| `tail` | 1B | 固定 `0x6B` |

控制命令的 ACK 固定 4 字节：`[地址] [功能码] [状态] [6B]`，状态 `02`=成功、`E2`=条件不满足
（堵转保护/未使能/零点无效）、`00 EE`=错误命令（帧比正常短，是 4 字节的 `ID 00 EE 6B`）。

> 广播帧的坑：`00 FF 66 6B` 会让四台**都**回一帧 `<自己的地址> FF 02 6B`，
> 驱动只读走第一帧，剩下的留在缓冲里。所以下一次单播必须**按地址 + 功能码**认回包，
> 否则会把上一轮留下的 `01 FF 02 6B` 当成自己这条 `0xFD` 的 ACK。

---

## 3. 如何找电机 ID（地址）

**方法一：面板看（最直接，推荐）**
Emm42 V5.0 驱动器的 OLED 菜单里有 `ID_Addr` 项，直接显示当前地址，默认 **1**。
多机前先把每台地址改成不重复的值。

**方法二：改地址命令**
```
格式: [addr] AE 4B [存储标志] [新ID] 6B
示例: 01 AE 4B 01 10 6B      # 把地址 1 的电机改成 16
```

**方法三：程序扫描（逐地址发无害读命令，看谁回包）**

```python
import serial

ser = serial.Serial("/dev/ttyUSB0", baudrate=115200, timeout=0.3)
found = []
for addr in range(1, 8):            # 扫描 1..7，可扩到 255
    ser.reset_input_buffer()
    ser.write(bytes((addr, 0x1F, 0x6B)))   # 0x1F = 读固件/硬件版本（无害）
    ser.flush()
    reply = ser.read(5)             # 期望 [addr][1F][fw][hw][6B]
    if len(reply) >= 4 and reply[0] == addr and reply[-1] == 0x6B:
        found.append(addr)
        print(f"ID={addr}  回复={reply.hex(' ')}")
ser.close()
print("扫描到的电机 ID:", found)
```

> `Emm42Driver` 也有两条读命令可以直接用：`read_status()`（`0x3A`）与 `read_position()`（`0x36`）。
> 扫描时**一次只接一台**更可靠；全部并接后扫描结果应与面板一致。

**本项目当前映射（`modules/chassis/hardware.json` 的 `vehicle.motor_ids`）**

| 软件轮位 | 实际驱动器地址 |
|---|---|
| FL | **1** |
| FR | **2** |
| RL | **3** |
| RR | **4** |

机械臂占 5 / 6 / 7（底座 / 升降 / 伸缩），与底盘 1~4 不冲突——`Robot.open`
在打开串口之前就会核对这件事。

> 软件永远按 `FL/FR/RL/RR` 逻辑轮位写代码；换装轮子只改配置，不动代码。

---

## 4. 第 1 层：协议驱动 `Emm42Driver`

文件：`modules/motor/drive/emm42.py`。**一台驱动一个对象**；同一个串口对象可以注入给多个驱动
（arm 与 chassis 就是这么共用总线的）。

```python
from modules.motor.drive.emm42 import Direction, Emm42Driver, HomeMode

m1 = Emm42Driver("/dev/ttyUSB0", motor_id=1).initialize()   # 自己开串口
m2 = Emm42Driver("/dev/ttyUSB0", motor_id=2, serial_port=bus)  # 借用已开的串口
```

### 构造函数

```python
Emm42Driver(
    port: str,                       # 串口设备：Jetson "/dev/ttyUSB0"、Windows "COM28"
    motor_id: int,                   # 1..255，必须与驱动器面板 ID 一致
    *,
    timeout: float = 0.1,            # 串口读超时（秒）
    pulses_per_revolution: int = 3200,  # 角度→脉冲换算；16细分 1.8° = 3200/圈
    expect_ack: bool = True,         # 这台驱动器的 Response 菜单会不会回 4 字节 ACK
    serial_port: SerialPort | None = None,  # 注入已开的串口/FakeSerial
    sleep: Callable[[float], None] = time.sleep,   # 测试注入
    clock: Callable[[], float] = time.monotonic,   # 测试注入
)
```

### 方法速查

| 方法 | 作用 | 帧 |
|---|---|---|
| `initialize()` / `close()` | 开/关串口（生命周期） | — |
| `enable(enabled: bool)` | `True` 锁轴使能 / `False` 松轴失能 | `F3 AB 01/00 00` |
| `set_speed(direction, rpm, acceleration)` | **持续正/反转**，速度 0~3000 | `F6 dir rpm accel sync` |
| `move_relative(degrees, rpm, acceleration, *, synchronized=False)` | 相对角度移动（正角=CW，0 度被拒） | `FD ...` |
| `move_absolute(degrees, ...)` | 绝对角度移动 | `FD ...` |
| `move_relative_pulses(pulses, ...)` | 相对脉冲数（正数=CW，不做角度换算，0 被拒） | `FD ...` |
| `move_absolute_pulses(pulses, ...)` | 绝对脉冲位置（**0 合法**，就是回原点） | `FD ...` |
| `trigger_sync_motion()` | 触发总线上所有"同步标志=1"的缓存命令 | `00 FF 66 6B` |
| `stop()` | 立即刹车（速度/位置模式通用，已停时驱动器回 `E2` 也算成功） | `FE 98 00` |
| `home(mode: HomeMode)` | 触发回零 | `9A mode` |
| `read_position()` | 读实时位置（带符号，`65536 = 一圈`） | `36` |
| `read_status()` | 读状态位：使能/到位/堵转/堵转保护 | `3A` |
| `read_homing_status()` | 读回零状态位 | `3B` |
| `read_encoder()` | 读单圈绝对编码器（0~65535，分不清圈数） | `31` |
| `wait_until_in_position(timeout, ...)` | 轮询到位位，返回最后读到的状态 | `3A` |

`Direction`：`CW = 0`（顺时针）、`CCW = 1`（逆时针）。`HomeMode`：`0` 单圈就近、`1` 单圈方向、
`2` 多圈碰撞、`3` 多圈限位开关（实际行为以驱动器面板配置为准，回零前要先用 `O_Set` 把
当前姿态设成零点）。

```python
m1.enable(True)                                   # 先使能才有扭矩
m1.set_speed(Direction.CW, rpm=300, acceleration=10)  # 300 RPM 正转
m1.stop()                                         # 刹车
m1.move_relative_pulses(pulses=-3200, rpm=300, acceleration=10)  # 逆时针转一圈
m1.move_absolute_pulses(pulses=0, rpm=300, acceleration=10)      # 回到计数器 0
print(m1.read_position(), m1.read_status())
m1.close()
```

### 4.1 回包匹配与重试（共用总线上的唯一正确姿势）

`_read_reply_for` 按**字节流重新对齐**，只有同时满足下面几条才算"这次交换的回包"：

- 长度正确、尾字节是 `0x6B`；
- 地址是自己的（广播帧没有地址，只认功能码）；
- **功能码对得上**（单播也一样——上一轮广播留下的 `NN FF 02 6B` 与本轮的 `NN FD 02 6B`
  只差这一个字节）；
- 它不是 `FD 9F` 到位包：`Response=Reached/Both` 的驱动器会在运动结束时异步发
  `ID FD 9F 6B`，那是**上一段动作的完成通知**，永远不是命令 ACK。

"驱动器回过话"和"什么都没等到"要分开处理：

- 回话（`02` / `E2` / `00 EE`）是**真结论**，立刻报错，重发没有意义；
- 什么都没等到，可能只是回包被打坏了，**幂等**命令重发（最多 `MAX_ATTEMPTS = 3` 次）；
  相对位置帧**永不重发**——它表达的是"再走这么多"，重发就是让轴多走一段。

发帧之前会先 `reset_input_buffer` 并且等总线安静（有字节在路上就等，最多
`QUIET_TIMEOUT_SECONDS`），因为读超时之后迟到的回包会毒到下一条命令。
`expect_ack=False` 的驱动器发完命令要留 `UNACKED_SETTLE_SECONDS` 静默——这就是底盘的
`inter_frame_seconds` 在处理机臂那一侧的对应物。

---

## 5. 第 2 层：整车控制 `MecanumDrive`

文件：`modules/chassis/drive/mecanum.py`。四台 Emm42 共用一个串口对象（总线分时复用），
提供麦轮运动学。

### 5.1 构造函数（真机建议走门面，见第 6 节）

```python
MecanumDrive(
    port: str,
    motor_ids: Mapping[WheelPosition, int] | None = None,   # 默认 FL1/FR2/RL3/RR4
    *,
    wheel_signs: Mapping[WheelPosition, int] | None = None, # 默认全 +1
    acceleration: int = 10,          # 默认加速度档
    pattern: str = "X",              # "X" 或 "O"（辊子排布）
    rotation_gain: float = 1.0,      # 自转速度倍率
    motion_transform: str = "standard",  # "standard" 或 "calibrated"（实车标定矩阵）
    expect_ack: bool = True,         # 四台驱动器是否回 ACK
    timeout: float = 0.1,
    pulses_per_revolution: int = 3200,
    serial_port: SerialPort | None = None,  # 注入已开的串口（借用，不由本对象关闭）
    sleep: Callable[[float], None] = time.sleep,
    settle_seconds: float = DEFAULT_SETTLE_SECONDS,
    inter_frame_seconds: float = DEFAULT_INTER_FRAME_SECONDS,  # 帧间静默，30 ms
    default_rpm: int | None = None,  # 调用方不传 rpm 时用它
    mm_per_wheel_rev: float | None = None,  # 轮子转一圈车走多少毫米；None = 拒绝毫米指令
)
```

### 5.2 常用 API

**生命周期 / 使能**

| 方法 | 说明 |
|---|---|
| `enable_all(enabled=True)` / `disable_all()` | 4 轮使能锁轴 / 松轴。**移动前必须先使能**（门面 `open` 已做过） |
| `enable_wheel(wheel, enabled=True)` | 单轮使能 |
| `close()` | 关**自己开的**共享串口（借来的只失效自身） |

**单轮 / 四轮**

| 方法 | 说明 |
|---|---|
| `get_wheel(wheel)` | 返回该轮底层 `Emm42Driver`，做回零等维护 |
| `set_wheel_speed(wheel, signed_rpm, acceleration=None)` | **正=前进、负=后退、0=停止** |
| `stop_wheel(wheel)` | 单轮立即刹车 |
| `stop()` | 4 轮全部刹车 |
| `move_wheels_relative_pulses(fl, fr, rl, rr, rpm, ...)` | 四轮位置模式（**脉冲数**，0 脉冲的轮子不下发，默认同步启动） |

**整车**

| 方法 | 说明 |
|---|---|
| `drive(vx, vy, omega, rpm, acceleration=None)` | 逆运动学入口：`vx>0`=前，`vy>0`=左，`omega>0`=逆时针；四轮转速归一化到 `rpm` |
| `drive_pulses(vx, vy, omega, pulses, rpm, acceleration=None, *, wait=True)` | 同一张矩阵，位移归一化到 `pulses`（量的是**峰值轮**） |
| `drive_mm(vx, vy, mm, rpm, acceleration=None, *, wait=True)` | 位移按**车心毫米**给；只接受纯平移（需要 `mm_per_wheel_rev`） |
| `forward/backward/left/right/rotate_cw/rotate_ccw/forward_left/.../backward_right(...)` | 方向方法，参数见 1.1 节 |

```python
car.forward(rpm=60, duration=2.0)     # 速度模式：跑 2 秒后自动停（阻塞 2 秒）
car.forward(rpm=200, pulses=3200)     # 位置模式：四轮各走 3200 脉冲后返回
car.forward(rpm=200, pulses=3200, wait=False)   # 只发命令，不等
car.drive_pulses(1.0, 1.0, 0.0, pulses=1600, rpm=200)  # 左前方，峰值轮 1600 脉冲
```

`WheelPosition` 枚举：`FL / FR / RL / RR`（本项目的轮位命名，无 BL/BR）。

---

## 6. 第 3 层：底盘门面 `MecanumChassis`（最推荐上层使用）

文件：`modules/chassis/runtime.py`。把配置装配、总线借用、使能与整车动作包成最简 API。

```python
from modules.chassis import MecanumChassis

with MecanumChassis.open() as car:        # 读配置 + 打开/借用总线 + 使能四轮
    car.forward(rpm=60)                    # 持续前进，直到 stop()
    car.left(rpm=40, duration=2.0)         # 向左平移 2 秒后自动停
    car.rotate_cw(rpm=30, duration=1.0)    # 顺时针 1 秒
```

门面方法就是方向方法本身：`forward / backward / left / right / rotate_cw / rotate_ccw /
forward_left / forward_right / backward_left / backward_right`，加 `enable() / disable() /
stop() / close()`；底层对象挂在 `car.drive` 上（需要 `drive()`/`drive_pulses()`/
`drive_mm()`/`get_wheel()` 时走它）。

`Robot.open()` 里装配时走的是 `MecanumChassis._from_settings(...)`，即"配置已经读好、总线已经
打开"的那条路；arm 与 chassis 因此共用同一条总线，各自 `close` 只失效自身。

> 底盘短动作演示见 `modules/chassis/app/demo.py`，地图路线见 `modules/chassis/app/map.py`，
> 交互式动作测试见 `modules/chassis/test/manual/chassis.py`；入口都在结束时停机、失能并关闭串口。

---

## 7. 配置加载：`modules/chassis/config.py`

只**读并校验**配置，不建驱动、不开串口（装配在 `MecanumChassis.open`）：

```python
from modules.chassis.config import HardwareProfile, load_profile

profile = load_profile("jetson")   # -> HardwareProfile 快照（合并两份文件）
```

**`modules/chassis/hardware.json`（硬件事实，换机器/换接线才改）**

| 字段 | 含义 | 当前实车 |
|---|---|---|
| `vehicle.motor_ids` | 硬件地址 → 软件轮位 | FL:1 FR:2 RL:3 RR:4 |
| `vehicle.wheel_signs` | 单轮方向修正（-1/1） | 全 +1 |
| `vehicle.expect_ack` | 四台驱动器是否回 4 字节 ACK | `true` |
| `vehicle.pattern` | 麦轮 X/O 排布 | X |
| `vehicle.motion_transform` | `standard`=标准矩阵 / `calibrated`=实车标定矩阵 | **calibrated** |
| `vehicle.mm_per_wheel_rev` | 轮子滚一圈车走多少毫米；`null` = 毫米指令会被明确拒绝 | `null` |

**`modules/chassis/config.json`（运动调参，调手感才改）**

| 字段 | 含义 | 当前值 |
|---|---|---|
| `vehicle.rpm` | 调用方不传 `rpm` 时的默认轮速 | 60 |
| `vehicle.acceleration` | 默认加速度档 | 150 |
| `vehicle.rotation_gain` | 自转速度倍率 | 1.0 |
| `profiles.<平台>.*` | 演示脚本的 rpm / 时长 / 停顿 | 见文件 |

串口**不在这两份文件里**：底盘与机械臂共用 `modules/motor/hardware.json` 的
`windows.port` / `jetson.port`（`auto` 会枚举 USB 串口逐个探测）。

---

## 8. 快速上手：按需求选一段直接抄

### 例 1：整车移动（前进 / 平移 / 自转）
```python
from modules.chassis import MecanumChassis

with MecanumChassis.open() as car:
    car.forward(rpm=60, duration=2.0)
    car.left(rpm=40, duration=1.5)
    car.rotate_ccw(rpm=30, duration=1.0)
    car.backward(rpm=60, duration=2.0)
```

### 例 2：底层整车 API 做斜向 / 组合运动
```python
with MecanumChassis.open() as car:
    car.drive(vx=1.0, vy=0.5, omega=0.0, rpm=50)   # 前斜左，四轮自动归一化
    time.sleep(1.0)
    car.stop()
    car.drive_pulses(1.0, 1.0, 0.0, pulses=1600, rpm=200)   # 位置模式：峰值轮 1600 脉冲
```

### 例 3：位置模式（相对 + 脉冲数 + 多机同步）
```python
with MecanumChassis.open() as car:
    # 四轮各走 180°/90° 之类就买脉冲：先缓存 4 条 FD，再发 00 FF 66 6B 同时启动
    car.drive.move_wheels_relative_pulses(1600, 1600, -1600, -1600, rpm=200)
    # 前进一圈 = 每轮 3200 脉冲，默认等动作走完
    car.forward(rpm=200, pulses=3200)
    # 任意方向走固定脉冲数：vx=1、vy=1（左前）
    car.drive.drive_pulses(1.0, 1.0, 0.0, pulses=1600, rpm=200)
```

> 两个位置动作**可以连着发**：回包按地址 + 功能码匹配、帧间有 30 ms 静默，
> 不需要自己插 `sleep`。

### 例 4：直接控制单台步进（不走整车层）
```python
from modules.motor.drive.emm42 import Direction, Emm42Driver

m = Emm42Driver("/dev/ttyUSB1", motor_id=1).initialize()
try:
    m.enable(True)
    m.set_speed(Direction.CW, rpm=150, acceleration=5)   # 单台正转 150 RPM
    time.sleep(2.0)
    m.stop()
finally:
    m.close()
```

### 例 5：总线舵机（夹爪 / 转盘）
```python
from modules.servo.drive.bus_servo import BusServo

servo = BusServo("/dev/ttyUSB0", servo_id=1).initialize()   # 需半双工转接板
try:
    servo.set_torque(True)
    servo.set_position(position=500, duration_ms=1000)   # 位置 0..1000 (0..240°)
    servo.set_speed(-250)                                # 电机模式，速度 -1000..1000
finally:
    servo.stop()
    servo.close()
```

---

## 9. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| 命令抛 `received no ACK` | motor_id 与面板地址不符；或 TX/RX 接反 / 断线；或帧被 CH340 打坏（帧间 30 ms 静默就是治它）。用第 3 节扫描确认地址 |
| 抛 `unexpected ACK: 01 ff 02 6b` | 同步广播的回包地址是**驱动器自己的**，不是 `00`。用当前版本代码（判据是"地址或广播 + 功能码"）；老版本会把这个当错误 |
| 抛 `unexpected ACK: ... FD 9F 6B` | 上一段动作的到位包撞进了读窗口。当前版本会丢掉它继续等；如果还报，检查是不是把 `expect_ack` 设错了 |
| 回包是 `... E2 ...` | 电机未 enable、堵转保护触发、或位置模式下零点无效 |
| 电机不动但 ACK 正常 | 忘了使能；或加速度档/转速组合太小；或这一帧被上一帧打坏（看报错里的 `discarded`） |
| 某个方向与预期相反 | 单轮反 → 配置 `wheel_signs` 给该轮 -1；整车反 → 换 `motion_transform`/`pattern`，不必改代码 |
| 一台串口能带几台 | 理论上 1~255 个地址；本项目 4 轮用 1~4、机械臂用 5/6/7，共用一条总线 |
| 需要多机同时启动 | 位置命令带 `sync=1`（`synchronized=True`），之后发 `trigger_sync_motion()`（`00 FF 66 6B`） |
| 舵机没反应 | 普通 USB-TTL 不能直连舵机信号线，必须用半双工转接板（见 `BusServo` 文档注释） |
| `pyserial is required on Jetson` | 目标环境没装 pyserial。所有开串口的地方都走 `modules/serial_link.py`，报错只有这一句 |
