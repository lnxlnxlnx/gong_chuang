# STP-23L 障碍物扫描

8 号 ZDT 电机带动 STP-23L 在约 90° 扇区内逐点扫描。启动位置由现场人员确认；程序把该位置记为 0°，逆时针累计 800 脉冲记为约 90°。当前无减速器，也没有零位开关或到位反馈，因此角度是基于 800 脉冲约等于 90° 的标定估计，不是传感器测得的绝对方位。

## 模块与数据流

```text
modules/laser_scan/config.json
    -> modules.laser_scan.drive.laser_scanner.load_scan_config()  参数校验
    -> modules/laser_scan/test/manual/laser_scan.py       打开 COM28/COM26，人工确认
    -> modules.laser_scan.drive.laser_scanner.scan_sector()      位置模式步进、采样、返回
       -> modules.motor.drive.emm42           8 号电机，115200 baud
       -> modules.laser_scan.drive.stp23l          激光数据，230400 baud
    -> CSV + SVG 雷达扇区图
```

STP-23L 开发手册第 2–3 页：测距帧以 `AA AA AA AA` 开头，后随地址 `00`、命令 `02`、偏移 `0000`、长度 `B8 00`、12 个各 15 字节的测量点、4 字节时间戳和校验和。总帧长 195 字节。校验和为从地址开始直到数据末尾的字节和低 8 位；距离是小端有符号整数，单位 mm。模块上电后持续发送，120 点/秒，即默认一帧约 0.1 秒。12 点是在同一停稳位置附近连续取得的时间样本，不是 12 个不同扫描角。每站丢弃运动期间的旧数据后，取新一帧中有效距离的中位数。

## 配置

只修改 `modules/laser_scan/config.json`：

- `motor.id=8`，`motor.port=COM28`，`modules.lidar.port=COM26` 是目前确认的接线。
- `motor.expect_ack=false`：当前 8 号驱动器不等待应答；若面板 Response 已开启，应按实际设置改成 `true`。
- `motor.pulses_per_revolution=3200`：按 1.8° 电机、16 细分计算；如果 8 号驱动器细分不同，按实际值修改。
- `motor.rpm=10`、`motor.acceleration=5`：位置模式的低速测试值。
- `scan.max_ccw_pulses`、`scan.step_pulses`、`scan.sweep_degrees`：扫描总脉冲、每次步进脉冲和对应的估算角度。最后一步自动使用剩余脉冲，不越过总脉冲上限。
- `scan.settle_seconds`：每步按速度估算的运动时间之外，额外等待的停稳时间。可在同一 JSON 中修改，不必改代码；没有到位反馈，缩短前须确认扫描头已停稳，否则距离会被标到错误角度。
- `scan.obstacle_distance_mm`：有效距离不大于此阈值时标为障碍物；`scan.min_distance_mm` 和 `scan.max_distance_mm` 限定有效测距范围。具体数值以当前 JSON 为准。

只有 `FD` 相对位置帧用于扫描。正常完成后，电机 CW 回走累计脉冲数，等待人工确认到达起点，再停止、失能。若采样失败或 Ctrl+C，中断时停止并失能，不推测剩余角度，也不自动执行可能越过机械边界的回程。

## 真机测试

在 Python 3.9 环境中从项目根目录运行：

```powershell
python test.py
```

预期：选择菜单 `10. 激光障碍物扫描` 后，先校验 STP-23L 帧，再提示确认扫描头位于顺时针起点；采样起点和每个步进位置，直到约 90° 终点。回程确认后在 `output/laser_scan/` 写入同名 `.csv` 和 `.svg`。CSV 记录脉冲、估算角度、毫米距离和障碍判定。SVG 是完整圆形雷达坐标图，但只有左下第三象限被扫描：左侧相对 0° 是起点，沿逆时针方向到下方相对 90° 是终点；其他象限留空表示未扫描。圆环标注按配置的最大距离分为四档，橙色虚线是 `scan.obstacle_distance_mm` 阈值；红点为阈值内障碍，青色为有效远距点。未获得有效距离时 CSV 距离为空，图上不画点，不把它判定为无障碍。

选择菜单 `11. 激光固定测距`：只打开 `modules.lidar.port`，不打开电机串口、不使能或移动电机。控制台持续输出每个校验通过的测距帧中有效点的中位距离、有效点数和障碍判定；按 `Ctrl+C` 关闭串口并返回菜单。没有有效点时显示“无有效数据/未知”，不会误报为无障碍。此测试可先观察静止状态下的读数稳定性；它不能单独证明电机运动已停稳。

数据手册：`docs/stp23l激光模块/3.STP-23L相关手册/1.LDROBOT_STP-23L_Datasheet_CN_v1.2.pdf`。协议：同目录的 `2.LDROBOT_STP-23L_Development_Manual_CN_v0.1.pdf`。供应商 Windows Python 示例只检查起始字节、未校验整帧，故本实现依照开发手册校验完整包与校验和。
