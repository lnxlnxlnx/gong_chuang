# only_for_test —— 真机只读探针

这一目录放的是**上机用的只读诊断脚本**。它们只发读取命令
（`0x31` / `0x36` / `0x3A` / `0x3B` / `0x42 6C`），**不使能力矩、不发任何运动帧**，
所以机械臂不会动、也不会加电锁轴。

配套的背景见 `docs/Emm42步进闭环驱动器命令手册.md` 与 `modules/grasp/docs/抓取坐标链路.md`。

---

## 一条铁律：别用菜单里的机械臂项做这件事

`modules/arm/test/manual/arm.py`（菜单项）走的是 `RobotArm.open()`，而它在装配时就会调
`enable_all()`，**给三轴加力矩**。要做"纯读"，直接用这些脚本，它们绕开 `RobotArm`，
自己开总线、自己按地址建 `Emm42Driver`。

三个脚本都用 `modules/motor/config.json` 里当前平台的 `port`（Windows 是 `COM35`）。
要换端口就改那个文件，不要在脚本里硬编码。

---

## 脚本

| 脚本 | 做什么 |
|---|---|
| `read_axes.py` | 读三轴的 `0x36` 实时位置 / `0x31` 单圈编码器 / `0x3A` 状态 / `0x3B` 回零状态 |
| `read_driver_config.py` | 读三轴的 `0x42 6C` 驱动配置帧，逐字段解析 21 个参数 |
| `scan_bus.py` | 扫地址，确认总线上到底挂着谁（排"读不到"用） |

```powershell
python only_for_test/read_axes.py
python only_for_test/read_axes.py --axes 5,7
python only_for_test/read_driver_config.py
python only_for_test/scan_bus.py --first 1 --last 8
```

---

## 第 1 步 · 读驱动配置（先做这个，1 分钟）

```powershell
python only_for_test/read_driver_config.py
```

四列必须看清：

| 列 | 为什么要看 |
|---|---|
| `MSTEP` | 不是 `0x10`（16 细分）的话，`modules/arm/config.json` 里的 `pulses_per_revolution: 3200` 就是错的，**所有脉冲↔毫米换算全废**。这个假设从第一天起没验证过 |
| `RESPONSE` | 顺带**实测验证**了"读命令不受 `Response` 影响"——这个脚本能读出东西本身就是证据 |
| `STALL_PROTECT` | **ON 表示撞到东西时该轴会直接掉力矩**。做任何碰撞实验之前必须先看到这一列 |
| `in_position_win` | "到位"判据有多严（手册 8.1 说默认 0.3°，示例帧却是 `0x0001` = 0.1°，两处打架） |

---

## 第 2 步 · 掉电实验：`0x36` 的多圈计数掉电后还在不在

手册**没有写**这件事，而它决定了整套方案：保持 → 锚定一次管一个上电周期；不保持 →
每次上电都得重新锚定。

### protocol

```
1. 上电，跑 read_axes.py，确认三轴的 rev_idx（圈数索引）都不为 0
       ↑ 如果某轴 rev_idx 是 0，先用手把它转过 2 个电机圈以上，再读一次
2. python only_for_test/read_axes.py --save before.json
3. 断驱动器电源
4. 用手把每根轴都转过 2 个电机圈以上（转多少不用精确，但要明显）
5. 上电
6. python only_for_test/read_axes.py --compare before.json
```

`--compare` 会直接给每根轴打判定。

### 判定怎么来的

有判别力的是**圈数索引**（`position // 65536`），不是原始值。因为驱动器若把多圈计数
存了 flash，上电时必须把小数部分重新对齐到当前编码器读数——所以小数部分会随"断电期间
手转的角度"变化，而整数部分保留下来。

| 判定 | 含义 | 后果 |
|---|---|---|
| `HELD` | 圈数索引扛过了掉电 | 锚定**做一次就够**，之后每次上电就知道自己在哪 |
| `RESET` | 索引回到 0，计数从单圈角重新开始 | **每次上电都要重新锚定** |
| `INCONCLUSIVE` | 两边索引都是 0，分不出来 | 掉电前先把手转 2 圈以上再来 |
| `UNKNOWN` | 索引变了但不是归零 | 把原始数字记下来人工判断 |

### 顺带看到的

- `delta` 列 = `0x36 mod 65536` 减 `0x31`。两者都是电机轴角度，正常情况下很小。
  三根轴都是**同一个**较大固定值 = 只是零点参考不同；**各轴不一样**才需要查解码。
- `encoder_ready=False`（屏幕上是不是 `Not Cal`？）→ 读数还不可信，先做编码器校准。

---

## 第 3 步 · 读不到东西时

```powershell
python only_for_test/scan_bus.py --first 1 --last 8
```

- 地址 1–4 应答、5–7 不应答 → **机械臂不在你插的这条总线上**。
  旧的 `docs/机械臂Emm42三轴控制说明.md` 记的是 COM29，而 `modules/motor/config.json`
  里现在是 COM35，这两个对不上就要先查线。
- 有地址回**不成帧**的数据 → 多半是**两个驱动器共用一个地址**，两条应答叠在总线上了。
  逐个拔掉再扫可以定位。

---

## 三个"不要"

1. **不要**用 `test.py` 菜单里的机械臂项做只读诊断（会 `enable_all()`）
2. **不要**发 `0x9A`——那是回零命令，机械臂会**动**
3. **不要**在跑完第 1 步、看到 `STALL_PROTECT` 之前做任何碰撞测试

---

## 跑测试要注意

仓库有两道守卫会扫到本目录，改这里的文件时别踩：

- `tests/test_python38_compatibility.py` 会遍历**全仓库每个 `.py`**，所以这里也必须
  **Python 3.8 兼容**（保留 `from __future__ import annotations`；别用 `X | Y` 注解、
  `str.removeprefix()` 之类）。
- `tests/test_discovery_coverage.py` 要求所有 `test_*.py` 被离线跑器发现，
  所以这里的文件名**不能以 `test_` 开头**。
