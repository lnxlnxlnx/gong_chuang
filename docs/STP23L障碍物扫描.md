# STP-23L 障碍物扫描

8 号 ZDT 电机带动 STP-23L 在约 90° 扇区内逐点扫描。启动位置由现场人员确认；程序把该位置记为 0°，逆时针累计 800 脉冲记为约 90°。当前无减速器，也没有零位开关或到位反馈，因此角度是基于 800 脉冲约等于 90° 的标定估计，不是传感器测得的绝对方位。

## 模块与数据流

```text
config/laser_scan.json
    -> laser_scanner.load_scan_config()  参数校验
    -> tests/manual/laser_scan.py       打开 COM28/COM26，人工确认
    -> laser_scanner.scan_sector()      位置模式步进、采样、返回
       -> motor_drivers.emm42           8 号电机，115200 baud
       -> motor_drivers.stp23l          激光数据，230400 baud
    -> CSV + SVG 雷达扇区图
```

STP-23L 开发手册第 2–3 页：测距帧以 `AA AA AA AA` 开头，后随地址 `00`、命令 `02`、偏移 `0000`、长度 `B8 00`、12 个各 15 字节的测量点、4 字节时间戳和校验和。总帧长 195 字节。校验和为从地址开始直到数据末尾的字节和低 8 位；距离是小端有符号整数，单位 mm。模块上电后持续发送，120 点/秒，即默认一帧约 0.1 秒。12 点是在同一停稳位置附近连续取得的时间样本，不是 12 个不同扫描角。每站丢弃运动期间的旧数据后，取新一帧中有效距离的中位数。

## 配置

只修改 `config/laser_scan.json`：

- `motor.id=8`，`motor.port=COM28`，`lidar.port=COM26` 是目前确认的接线。
- `motor.expect_ack=true`：8 号驱动器的 Response 设置尚未确认。若面板实测为 `Response=None`，改成 `false`；否则无 ACK 会在使能时报错。
- `motor.pulses_per_revolution=3200`：按 1.8° 电机、16 细分计算；如果 8 号驱动器细分不同，按实际值修改。
- `motor.rpm=10`、`motor.acceleration=5`：位置模式的低速测试值。
- `scan.max_ccw_pulses=800`、`scan.step_pulses=40`、`scan.sweep_degrees=90`：从起点扫到约 90°。步长改为 30 时，最后一步自动使用剩余 20 脉冲，不越过 800。
- `scan.settle_seconds=0.5`：在按速度计算的运动时间之外，额外等待的稳定时间。没有到位反馈，应按实测运动情况调整。
- `scan.obstacle_distance_mm=1000`：不大于 1 米的有效点标为障碍物；`min_distance_mm=70`、`max_distance_mm=7500` 取自数据手册的典型测距范围。

只有 `FD` 相对位置帧用于扫描。正常完成后，电机 CW 回走累计脉冲数，等待人工确认到达起点，再停止、失能。若采样失败或 Ctrl+C，中断时停止并失能，不推测剩余角度，也不自动执行可能越过机械边界的回程。

## 真机测试

在 Python 3.9 环境中从项目根目录运行：

```powershell
python -m tests.manual.laser_scan
```

预期：先校验 STP-23L 帧，再提示确认扫描头位于顺时针起点；共采集 21 个角度（0°, 4.5°, ... 90°），回程确认后在 `output/laser_scan/` 写入同名 `.csv` 和 `.svg`。CSV 记录脉冲、估算角度、毫米距离和障碍判定；SVG 是 0–90° 扇区图，红点为阈值内障碍，青色为有效远距点。未获得有效距离时 CSV 距离为空，不把它判定为无障碍。

数据手册：`docs/stp23l激光模块/3.STP-23L相关手册/1.LDROBOT_STP-23L_Datasheet_CN_v1.2.pdf`。协议：同目录的 `2.LDROBOT_STP-23L_Development_Manual_CN_v0.1.pdf`。供应商 Windows Python 示例只检查起始字节、未校验整帧，故本实现依照开发手册校验完整包与校验和。
