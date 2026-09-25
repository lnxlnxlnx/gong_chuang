# 电机总线控制与调用 API 文档

> 适用对象：`gong_chuang/` 下两套串口驱动 —— **Emm42 V5.0 闭环步进驱动器**（麦轮底盘用）
> 和 **幻尔总线舵机**（机械臂/云台类执行器用）。
>
> 本文是"照着写代码"的技术文档：先讲总线如何区分电机、如何找电机 ID，
> 再给三层的完整函数签名与可直接复制的调用示例。
> 协议级帧格式速查见根目录 `电机舵机驱动命令速查.md`。

---

## 1. 总体架构：一条总线 + 三层代码

```
┌─ 应用代码（你写的）────────────────────────────────────┐
│  from hardware_tests.mecanum.config import create_chassis  │
│  car = create_chassis(profile).initialize()                │
│  car.forward(rpm=60)                                       │
└──────────────────────────┬───────────────────────────────┘
                           │ 第 3 层 底盘门面 MecanumChassis (chassis.py)
┌──────────────────────────▼───────────────────────────────┐
│ 第 2 层 整车控制 MecanumDrive (mecanum.py)                 │
│   · 四台 Emm42Driver 共享同一个 serial 对象                 │
│   · drive() 逆运动学矩阵算出四轮转速，逐轮发 F6 帧           │
└──────────────────────────┬───────────────────────────────┘
                           │ 第 1 层 协议驱动 (motor_drivers/)
┌──────────────────────────▼───────────────────────────────┐
│ Emm42Driver ── 按 motor_id 组帧/解 ACK, 写共享串口          │
│ BusServo    ── 按 servo_id 组帧/解 ACK                     │
└──────────────────────────┬───────────────────────────────┘
                           │ 一根 USB-TTL (115200, 8N1)
                    Emm42 总线（全双工 TTL，多机并接）
```

- **第 1 层（协议）**：`motor_drivers/emm42.py` `Emm42Driver`、`motor_drivers/bus_servo.py` `BusServo`。
- **第 2 层（整车）**：`mecanum.py` `MecanumDrive`，四轮共享一根串口，含麦轮逆运动学。
- **第 3 层（门面）**：`chassis.py` `MecanumChassis`，向上层应用提供的"整车动作" API。
- **配置**：`config/mecanum_hardware.json`，硬件地址、方向修正、串口都在这里，脚本里用
  `hardware_tests/mecanum/config.py` 读取。

> 总线舵机只到第 1 层（没有整车层），用 `BusServo` 直接控制。

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
| `addr` | 1B | 驱动器地址 **1~255**；`0x00` = 广播（所有电机执行，**只有地址 1 回复**） |
| `func` | 1B | `F6`=速度模式、`FD`=位置模式、`F3 AB`=使能、`FE 98`=停止、`FF 66`=同步触发 |
| `dir` | 1B | **`00`=CW（正转）`01`=CCW（反转）** |
| `rpm` | 2B 大端 | 0~3000 |
| `accel` | 1B | 0~255，`0`=无加减速曲线；档位越大加速越快 |
| `sync` | 1B | `0`=立即执行；`1`=缓存住，等 `00 FF 66 6B` 同步帧一起启动 |
| `tail` | 1B | 固定 `0x6B` |

ACK 回包固定 4 字节：`[地址] [功能码] [状态] [6B]`，状态 `02`=成功、`E2`=条件不满足
（堵转保护/未使能/零点无效）、`EE`=错误命令。

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

**方法三：程序扫描（逐地址发无害读命令，看谁回 ACK）**
`Emm42Driver` 只支持控制命令（不解析读命令回包），扫描用裸 pyserial 即可：

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

建议**一次只接一台**做扫描，逐台确认；全部并接后扫描结果应与面板一致。

**本项目当前映射（`config/mecanum_hardware.json`）**

| 软件轮位 | 实际驱动器地址 |
|---|---|
| FL | **3** |
| FR | **4** |
| RL | **2** |
| RR | **1** |

> 软件永远按 `FL/FR/RL/RR` 逻辑轮位写代码；换装轮子只改配置，不动代码。

---

## 4. 第 1 层：协议驱动 `Emm42Driver`

文件：`motor_drivers/emm42.py`。**一台驱动一个对象**，都传同一个串口设备名（各自按 motor_id 组帧）。

```python
from motor_drivers import Direction, Emm42Driver, HomeMode

# 每台电机一个对象；同一根总线可挂多台
m1 = Emm42Driver("/dev/ttyUSB0", motor_id=1).initialize()   # 打开串口
m2 = Emm42Driver("/dev/ttyUSB0", motor_id=2).initialize()
```

### 构造函数

```python
Emm42Driver(
    port: str,                       # 串口设备：Jetson "/dev/ttyUSB0"、Windows "COM28"
    motor_id: int,                   # 1..255，必须与驱动器面板 ID 一致
    *,
    timeout: float = 0.1,            # 串口读超时（秒）
    pulses_per_revolution: int = 3200,  # 角度→脉冲换算；16细分 1.8° = 3200/圈
    serial_port: SerialPort | None = None,  # 测试注入 FakeSerial 用
)
```

### 方法速查

| 方法 | 作用 | 帧 |
|---|---|---|
| `initialize()` / `close()` | 开/关串口（生命周期） | — |
| `enable(enabled: bool)` | `True` 锁轴使能 / `False` 松轴失能 | `F3 AB 01/00 00` |
| `set_speed(direction: Direction, rpm: int, acceleration: int)` | **持续正/反转**，速度 0~3000 | `F6 dir rpm accel sync` |
| `move_relative(degrees, rpm, acceleration, *, synchronized=False)` | 相对角度移动（正角=CW） | `FD ...` |
| `move_absolute(degrees, rpm, acceleration, *, synchronized=False)` | 绝对角度移动 | `FD ...` |
| `trigger_sync_motion()` | 触发总线所有"同步标志=1"的缓存命令 | `00 FF 66 6B` |
| `stop()` | 立即刹车（速度/位置模式通用） | `FE 98 00` |
| `home(mode: HomeMode)` | 触发回零 | `9A mode` |

`Direction`：`CW = 0`（顺时针）、`CCW = 1`（逆时针）。

```python
m1.enable(True)                                   # 先使能才有扭矩
m1.set_speed(Direction.CW, rpm=300, acceleration=10)  # 300 RPM 正转
m1.set_speed(Direction.CCW, rpm=300, acceleration=10) # 反转
m1.stop()                                         # 刹车
m1.move_relative(degrees=-360.0, rpm=300, acceleration=10)  # 逆时针转一圈
m1.move_absolute(degrees=720.0, rpm=300, acceleration=10)   # 到 720° 绝对位置
m1.close()
```

> 控制命令每帧都**等 ACK**：没回复抛 `RuntimeError "received no ACK"`（地址不对或接线断）；
> 回 `E2` 也抛异常（电机未使能/堵转保护）。

---

## 5. 第 2 层：整车控制 `MecanumDrive`

文件：`mecanum.py`。四台 Emm42 共用一个串口对象（总线分时复用），提供麦轮运动学。

### 5.1 构造函数（真机建议走配置，见第 7 节）

```python
MecanumDrive(
    port: str,
    motor_ids: Mapping[WheelPosition, int] | None = None,  # 默认 FL1/FR2/RL3/RR4
    *,
    wheel_signs: Mapping[WheelPosition, int] | None = None, # 默认全 +1
    acceleration: int = 10,      # 默认加速度档
    pattern: str = "X",          # "X" 或 "O"（辊子排布）
    rotation_gain: float = 1.0,  # 自转速度倍率
    motion_transform: str = "standard",  # "standard" 或 "calibrated"（实车标定）
    timeout: float = 0.1,
    pulses_per_revolution: int = 3200,
    serial_port: SerialPort | None = None,
)
```

### 5.2 常用 API 分组

**生命周期 / 使能**

| 方法 | 说明 |
|---|---|
| `initialize()` | 返回自身（链式）；共享串口构造时已开 |
| `enable_all(enabled=True)` / `disable_all()` | 4 轮使能锁轴 / 松轴。**移动前必须先 enable_all()** |
| `enable_wheel(wheel, enabled=True)` | 单轮使能 |
| `close()` | 关共享串口（一次即可） |

**单轮控制**（供上层调用）

| 方法 | 说明 |
|---|---|
| `get_wheel(wheel)` | 返回该轮底层 `Emm42Driver`，做回零等维护 |
| `set_wheel_speed(wheel, signed_rpm, acceleration=None)` | **正=前进、负=后退、0=停止** |
| `set_wheel_acceleration(wheel, accel)` | 设默认加速度档 |
| `stop_wheel(wheel)` | 单轮立即刹车 |
| `move_wheel_relative(degrees, rpm, ...)` / `move_wheel_absolute(...)` | 单轮位置模式 |
| `home_wheel(wheel, mode)` | 单轮回零 |

**整车控制**

| 方法 | 说明 |
|---|---|
| `drive(vx, vy, omega, rpm, acceleration=None)` | 逆运动学入口：`vx>0`=前，`vy>0`=左，`omega>0`=逆时针；四轮转速归一化到 `rpm` |
| `set_speeds(fl, fr, rl, rr, acceleration=None)` | 最低层整车入口：直接给 4 个带符号 RPM，绕过矩阵 |
| `stop()` | 4 轮全部刹车 |
| `move_wheels_relative(fl, fr, rl, rr, rpm, *, synchronized=True)` | 4 轮相对位置，默认**多机同步**（缓存 FD 后发 `00 FF 66 6B`） |
| `move_wheels_absolute(fl, fr, rl, rr, rpm, *, synchronized=True)` | 4 轮绝对位置 |

**高层方向方法**（都可由 `duration=` 限时自动停）

| 方法 | 方向 |
|---|---|
| `forward(rpm)` / `backward(rpm)` | 前进 / 后退 |
| `strafe_left(rpm)` / `strafe_right(rpm)` | 左平移 / 右平移 |
| `rotate_cw(rpm)` / `rotate_ccw(rpm)` | 俯视顺时针 / 逆时针原地自转 |
| `forward_left/forward_right/backward_left/backward_right(rpm)` | 四个斜向 |

`WheelPosition` 枚举：`FL / FR / RL / RR`（注意是本项目的轮位命名，无 BL/BR）。

---

## 6. 第 3 层：底盘门面 `MecanumChassis`（最推荐上层使用）

文件：`chassis.py`。把整车动作包成最简 API。**配合配置加载使用（第 7 节）**，这样
真机/Windows 切换、换装轮子都不用改应用代码。

```python
from hardware_tests.mecanum.config import create_chassis, load_profile

profile = load_profile("jetson")          # 或 "windows"
car = create_chassis(profile).initialize()

car.enable_all()                          # 移动前必须使能
car.forward(rpm=60)                       # 持续前进，直到 stop()
car.left(rpm=40, duration=2.0)            # 向左平移 2 秒后自动停
car.rotate_cw(rpm=30, duration=1.0)       # 顺时针 1 秒
car.stop()
car.disable_all()                         # 松轴
car.close()
```

门面方法 = `MecanumDrive` 同名方法的别名：`forward/backward/left/right/strafe_left/strafe_right/
rotate_cw/rotate_ccw/forward_left/forward_right/backward_left/backward_right`，
另外有 `enable()/enable_all()/disable()/disable_all()/stop()/set_wheel_speed()/
move_wheels_relative()/move_wheels_absolute()/wheel(WheelPosition)`（返回底层驱动对象）。

> 建议真机示例直接抄 `examples/demo_mecanum.py`，它用 try/finally 保证结束时必松轴关串口。

---

## 7. 配置加载：`hardware_tests/mecanum/config.py`

配置文件 `config/mecanum_hardware.json` 集中管理硬件参数：

```python
from hardware_tests.mecanum.config import (
    create_chassis,   # -> MecanumChassis（上层应用推荐）
    create_drive,     # -> MecanumDrive（需要底层 API 时）
    load_profile,     # -> HardwareProfile（只读配置，不建控制对象）
)

profile = load_profile("jetson")
car = create_chassis(profile)            # 可覆盖: create_chassis(profile, port="/dev/ttyUSB1", acceleration=8)
```

配置字段含义：

| 字段 | 含义 | 当前实车 |
|---|---|---|
| `vehicle.motor_ids` | 硬件地址 → 软件轮位 | FL:3 FR:4 RL:2 RR:1 |
| `vehicle.wheel_signs` | 单轮方向修正（-1/1） | 全 +1 |
| `vehicle.acceleration` | 默认加速度档 | 5 |
| `vehicle.pattern` | 麦轮 X/O 排布 | X |
| `vehicle.rotation_gain` | 自转速度倍率 | 1.0 |
| `vehicle.motion_transform` | `standard`=标准矩阵 / `calibrated`=实车标定矩阵 | **calibrated** |
| `profiles.jetson.port` | Jetson 串口 | /dev/ttyUSB0 |
| `profiles.windows.port` | Windows 串口 | COM28 |
| `profiles.*.chassis_motion` / `individual_wheels` | 测试脚本默认 rpm/时长 | 见文件 |

---

## 8. 快速上手：按需求选一段直接抄

### 例 1：整车移动（前进 / 平移 / 自转）
```python
car = create_chassis(load_profile("jetson")).initialize()
try:
    car.enable_all()
    car.forward(rpm=60, duration=2.0)
    car.strafe_left(rpm=40, duration=1.5)
    car.rotate_ccw(rpm=30, duration=1.0)
    car.backward(rpm=60, duration=2.0)
finally:
    car.stop(); car.disable_all(); car.close()
```

### 例 2：底层整车 API 做斜向 / 组合运动
```python
drive = create_drive(load_profile("jetson")).initialize()
drive.enable_all()
drive.drive(vx=1.0, vy=0.5, omega=0.0, rpm=50)  # 前斜左，四轮自动归一化
time.sleep(1.0)
drive.stop()
drive.set_speeds(30, -30, 30, -30, acceleration=8)  # 绕过矩阵直接赋值
time.sleep(0.5)
drive.stop(); drive.disable_all(); drive.close()
```

### 例 3：位置模式（相对 + 多机同步）
```python
car = create_chassis(load_profile("jetson")).initialize()
try:
    car.enable_all()
    # 四轮同步各转 180°：先缓存 4 条 FD，再发 00 FF 66 6B 同时启动
    car.move_wheels_relative(fl=180.0, fr=180.0, rl=-180.0, rr=-180.0, rpm=200, synchronized=True)
    # 绝对位置
    car.move_wheels_absolute(fl=0.0, fr=0.0, rl=0.0, rr=0.0, rpm=200, synchronized=True)
finally:
    car.disable_all(); car.close()
```

### 例 4：直接控制单台步进（不走整车层）
```python
from motor_drivers import Direction, Emm42Driver

m = Emm42Driver("/dev/ttyUSB0", motor_id=1).initialize()
m.enable(True)
m.set_speed(Direction.CW, rpm=150, acceleration=5)   # 单台正转 150 RPM
time.sleep(2.0)
m.stop()
m.close()
```

### 例 5：总线舵机（机械臂/云台类）
```python
from motor_drivers import BusServo

servo = BusServo("/dev/ttyUSB0", servo_id=1).initialize()   # 需半双工转接板
servo.set_torque(True)
servo.set_position(position=500, duration_ms=1000)   # 位置 0..1000 (0..240°)
servo.set_speed(-250)                                # 电机模式，速度 -1000..1000
servo.stop()
servo.close()
```

---

## 9. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| 命令抛 `received no ACK` | motor_id 与面板地址不符；或 TX/RX 接反 / 断线。用第 3 节扫描确认地址 |
| 回包是 `... E2 ...` | 电机未 enable、堵转保护触发、或位置模式下零点无效 |
| 电机不动但 ACK 正常 | 忘了 `enable_all(True)`；或加速度档/转速组合太小 |
| 某个方向与预期相反 | 单轮反 → 配置 `wheel_signs` 给该轮 -1；整车反 → 换 `motion_transform`/`pattern`，不必改代码 |
| 一台串口能带几台 | 理论上 1~255 个地址；总线负载按实际接线 4 台就用地址 1~4（本项目配置见第 7 节） |
| 需要多机同时启动 | 位置/速度命令带 `sync=1` 或 `synchronized=True`，之后发 `trigger_sync_motion()`（`00 FF 66 6B`） |
| 舵机没反应 | 普通 USB-TTL 不能直连舵机信号线，必须用半双工转接板（见 BusServo 文档注释） |