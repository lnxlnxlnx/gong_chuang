"""解析镜头端发来的「一条物料观测」文本（纯解析，不碰串口 / socket / cv2）。

镜头端（Jetson 上的圆检测进程）每条物料发一行，形如：

```
物料 4: @(566,376) 绿色物料（green），confidence=0.88，圆直径 268 px，内部σ=20.2，边界支撑=83%（36 个方向），xyz=(-8.3,1.8,113.2) mm，直线距离约 113.5 mm
```

## 为什么这一层只做解析

这条协议里**最值钱的是 ``xyz``**：它已经是毫米，说明镜头端做完了针孔投影与直径测距。
本仓库的 :mod:`modules.vision.drive.pinhole` 也能算这一步，但那是"只拿到像素"时的
退路；既然对面直接给了毫米，这里就**不再重复换算一遍**——重复换算只会引入两套
互相打架的尺度。

所以这一层只回答一个问题：**这行文本说的是什么？** 它不认识机械臂，也不认识圆柱坐标。
坐标变换是 :mod:`modules.vision.drive.sighting_plane` 的事。

## 一行里哪些字段是必须的

必须：物料编号、``xyz``。可选：像素坐标（只有诊断价值）、颜色名、置信度、圆直径、
``直线距离``。**可选的字段缺失时不报错**——镜头端在调参阶段会加减字段，
让"少打印一个 confidence"直接炸掉整条抓取链路是不划算的。

``xyz`` 缺失则**必须报错**：没有它就换算不出平面坐标，静默地拿默认值会让机械臂
照着一个编出来的位置动，那比报错危险得多。

## 坐标系

``xyz`` 的原点是**镜头光心**，``z`` 沿光轴向前（俯视时就是向下）。单位毫米。

光轴竖直向下时 ``x/y`` 就是水平毫米量，这正是
:class:`modules.vision.drive.calibration.PlaneCalibration` 需要的输入。
相机一旦倾斜，水平坐标会带上与高度相关的偏置（见 ``modules/grasp/docs/抓取坐标链路.md``
第 2 节），所以 ``z`` 被照实保留下来：它不参与平面换算，但用来核对"台面深度"标定对不对。

## 像素半径与 ``(u, v)``

``@(566,376)`` 是物料圆心在画面里的像素坐标。这里只当作**诊断量**保存：
真正用于定位的是 ``xyz``。保留它是为了在真机上能一眼看出"镜头报的点"和"画面上的圆"
是不是同一个东西。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

#: 「物料 N:」——编号是物料的稳定标识，用来判断"还是不是同一个物料"。
#: 冒号可以省略，但**后面必须有空白**：`.` 不能匹配任意字符，否则 "物料4xyz=(...)"
#: 这种没有分隔的写法会把编号吃掉一半。
_MATERIAL_ID = re.compile(r"物料\s*(\d+)\s*(?:[:：]|\s)")

#: 「@(566,376)」——像素圆心；半角/全角括号与空格都容忍。
_PIXEL_CENTER = re.compile(r"@\s*[（(]\s*(-?\d+(?:\.\d+)?)\s*[,，]\s*(-?\d+(?:\.\d+)?)\s*[)）]")

#: 「xyz=(-8.3,1.8,113.2) mm」——镜头系毫米坐标。**唯一的必填几何量。**
_XYZ = re.compile(
    r"xyz\s*[=＝]\s*[（(]\s*"
    r"(-?\d+(?:\.\d+)?)\s*[,，]\s*"
    r"(-?\d+(?:\.\d+)?)\s*[,，]\s*"
    r"(-?\d+(?:\.\d+)?)\s*[)）]",
    re.IGNORECASE,
)

#: 「confidence=0.88」——检测置信度。
_CONFIDENCE = re.compile(r"confidence\s*[=＝]\s*(\d+(?:\.\d+)?)", re.IGNORECASE)

#: 「圆直径 268 px」——像素直径，用来和 xyz 互相印证比例尺。
_DIAMETER_PX = re.compile(r"圆直径\s*(\d+(?:\.\d+)?)\s*px")

#: 「内部σ=20.2」——颜色一致性指标，越小越像纯色圆面。
_INNER_SIGMA = re.compile(r"内部\s*σ\s*[=＝]\s*(\d+(?:\.\d+)?)")

#: 「边界支撑=83%（36 个方向）」——圆边界被支持的比例。
_EDGE_SUPPORT = re.compile(r"边界支撑\s*[=＝]\s*(\d+(?:\.\d+)?)\s*%")
_EDGE_DIRECTIONS = re.compile(r"(\d+)\s*个方向")

#: 「直线距离约 113.5 mm」——镜头到物料的斜距，等于 ``hypot(x, y, z)``，是自洽性检查。
_STRAIGHT_DISTANCE = re.compile(r"直线距离(?:约)?\s*(\d+(?:\.\d+)?)\s*mm")

#: 「绿色物料（green）」——中文色名 + 括号里的英文色名。
_COLOR = re.compile(r"([\u4e00-\u9fff]+)\s*物料\s*[（(]\s*([A-Za-z_]+)\s*[)）]")
#: 兜底：只给英文色名的写法，如「green 物料」。
_COLOR_EN = re.compile(r"\b(red|green|blue|yellow|black|white|orange|purple)\b", re.IGNORECASE)

#: 颜色名 -> 本仓库统一的英文键（与 ``modules/vision/config.json`` 的
#: ``targets.material_palette`` 同域）。
_COLOR_NAMES = {
    "红": "red",
    "红色": "red",
    "黄": "yellow",
    "黄色": "yellow",
    "蓝": "blue",
    "蓝色": "blue",
    "绿": "green",
    "绿色": "green",
    "黑": "black",
    "黑色": "black",
    "白": "white",
    "白色": "white",
}


class SightingParseError(ValueError):
    """这一行不是一条能用的物料观测。"""


@dataclass(frozen=True)
class MaterialSighting:
    """镜头端报的一条物料观测（解析结果，未做任何坐标变换）。

    - ``material_id``：物料编号，来自「物料 N:」；
    - ``x_mm`` / ``y_mm``：**镜头水平面**上的位置（毫米）。光轴竖直向下时就是水平真值；
    - ``z_mm``：沿光轴的深度（毫米），俯视时向下为正。不参与平面换算，用于核对台面深度；
    - ``pixel_u`` / ``pixel_v``：画面像素圆心，诊断用，可能为 ``None``；
    - ``color``：归一化后的颜色键（``green`` / ``blue`` / …），识别不出时为 ``None``；
    - ``confidence``：检测置信度，可能为 ``None``；
    - ``diameter_px`` / ``inner_sigma`` / ``edge_support`` / ``edge_directions``：
      圆检测的质量指标，全部可选，用来判断这一条观测值不值得信；
    - ``straight_distance_mm``：镜头到物料的斜距，与 ``xyz`` 冗余，用来对账。
    """

    material_id: int
    x_mm: float
    y_mm: float
    z_mm: float
    pixel_u: Optional[float] = None
    pixel_v: Optional[float] = None
    color: Optional[str] = None
    confidence: Optional[float] = None
    diameter_px: Optional[float] = None
    inner_sigma: Optional[float] = None
    edge_support: Optional[float] = None
    edge_directions: Optional[int] = None
    straight_distance_mm: Optional[float] = None
    #: 原始行，原样保留：真机上出问题时最想看到的就是"到底收到了什么"。
    raw: str = ""

    @property
    def distance_mm(self) -> float:
        """镜头到物料的斜距，由 ``xyz`` 算出（不是解析来的那个约数）。"""
        return (self.x_mm ** 2 + self.y_mm ** 2 + self.z_mm ** 2) ** 0.5

    @property
    def planar_distance_mm(self) -> float:
        """镜头光轴到物料的水平距离；光轴竖直时就是水平面里的距离。"""
        return (self.x_mm ** 2 + self.y_mm ** 2) ** 0.5

    def summary(self) -> str:
        parts = [
            f"物料 {self.material_id}",
            f"镜头系 xyz=({self.x_mm:.1f},{self.y_mm:.1f},{self.z_mm:.1f}) mm",
        ]
        if self.color is not None:
            parts.append(self.color)
        if self.confidence is not None:
            parts.append(f"conf={self.confidence:.2f}")
        if self.pixel_u is not None and self.pixel_v is not None:
            parts.append(f"像素({self.pixel_u:.0f},{self.pixel_v:.0f})")
        return "；".join(parts)


def parse_sighting(line: str) -> MaterialSighting:
    """把一行镜头文本解析成 :class:`MaterialSighting`。

    :raises SightingParseError: 没有「物料 N:」编号，或者没有 ``xyz=`` 坐标。
        这两种情况下拿不到可用的目标位置，**必须**响亮地失败。
    """
    if not isinstance(line, str):
        raise SightingParseError(f"期望一行文本，收到 {type(line).__name__}")
    text = line.strip()
    if not text:
        raise SightingParseError("空行不是一条物料观测")

    material_match = _MATERIAL_ID.search(text)
    if material_match is None:
        raise SightingParseError(f"找不到「物料 N:」编号：{text!r}")

    xyz_match = _XYZ.search(text)
    if xyz_match is None:
        raise SightingParseError(
            f"找不到 xyz=(x,y,z) 毫米坐标：{text!r}；"
            "没有镜头系坐标就无法定位，请让镜头端打开 xyz 输出"
        )

    diameter = _number(_DIAMETER_PX, text)
    directions = _number(_EDGE_DIRECTIONS, text)
    return MaterialSighting(
        material_id=int(material_match.group(1)),
        x_mm=float(xyz_match.group(1)),
        y_mm=float(xyz_match.group(2)),
        z_mm=float(xyz_match.group(3)),
        pixel_u=_group(_PIXEL_CENTER, text, 1),
        pixel_v=_group(_PIXEL_CENTER, text, 2),
        color=_color(text),
        confidence=_number(_CONFIDENCE, text),
        diameter_px=diameter,
        inner_sigma=_number(_INNER_SIGMA, text),
        edge_support=_number(_EDGE_SUPPORT, text),
        edge_directions=None if directions is None else int(directions),
        straight_distance_mm=_number(_STRAIGHT_DISTANCE, text),
        raw=text,
    )


def try_parse_sighting(line: str) -> Optional[MaterialSighting]:
    """解析成功返回观测，失败返回 ``None``。

    接收线程用它：串口 / socket 上混着日志行是常态，**不该让一行无关输出把整条链路打断**。
    需要知道"为什么没解析出来"的地方（标定时、调试时）请直接用 :func:`parse_sighting`。
    """
    try:
        return parse_sighting(line)
    except SightingParseError:
        return None


def _group(pattern: re.Pattern, text: str, index: int) -> Optional[float]:
    match = pattern.search(text)
    return None if match is None else float(match.group(index))


def _number(pattern: re.Pattern, text: str) -> Optional[float]:
    return _group(pattern, text, 1)


def _color(text: str) -> Optional[str]:
    """取颜色键：先认中文色名，再退回英文色名。"""
    match = _COLOR.search(text)
    if match is not None:
        chinese, english = match.group(1), match.group(2)
        return _COLOR_NAMES.get(chinese, english.lower())
    fallback = _COLOR_EN.search(text)
    return None if fallback is None else fallback.group(1).lower()


# ---------------------------------------------------------------------- 自测
_SAMPLE = (
    "物料 4: @(566,376) 绿色物料（green），confidence=0.88，圆直径 268 px，"
    "内部σ=20.2，边界支撑=83%（36 个方向），xyz=(-8.3,1.8,113.2) mm，直线距离约 113.5 mm"
)


def _self_test() -> int:
    """不需要镜头：用几个真实形态的样例行验证解析。"""
    sighting = parse_sighting(_SAMPLE)
    assert sighting.material_id == 4, sighting
    assert (sighting.x_mm, sighting.y_mm, sighting.z_mm) == (-8.3, 1.8, 113.2), sighting
    assert (sighting.pixel_u, sighting.pixel_v) == (566.0, 376.0), sighting
    assert sighting.color == "green", sighting
    assert sighting.confidence == 0.88, sighting
    assert sighting.diameter_px == 268.0, sighting
    assert sighting.inner_sigma == 20.2, sighting
    assert sighting.edge_support == 83.0, sighting
    assert sighting.edge_directions == 36, sighting
    assert sighting.straight_distance_mm == 113.5, sighting
    # 冗余的斜距要和 xyz 对得上：对不上说明镜头端拿错了深度。
    assert abs(sighting.distance_mm - sighting.straight_distance_mm) < 1.0, sighting.distance_mm

    # 半角标点、多余空格、缺可选字段都要能解析。
    loose = parse_sighting("物料12:@(1,2) xyz=(3.5, -4, 5) mm")
    assert loose.material_id == 12, loose
    assert (loose.x_mm, loose.y_mm, loose.z_mm) == (3.5, -4.0, 5.0), loose
    assert loose.confidence is None and loose.color is None, loose
    assert loose.diameter_px is None and loose.edge_directions is None, loose
    assert (loose.pixel_u, loose.pixel_v) == (1.0, 2.0), loose

    # 中文全角等号、全角逗号、大写 XYZ 也要认。
    assert parse_sighting("物料 1 @（3，4） XYZ＝（1，2，3）mm").z_mm == 3.0

    # 少了 xyz 必须报错：编一个位置比报错危险得多。
    try:
        parse_sighting("物料 4: @(566,376) 绿色物料（green），confidence=0.88")
    except SightingParseError as error:
        assert "xyz" in str(error), error
    else:
        raise AssertionError("缺 xyz 必须报 SightingParseError")

    # 少了编号也必须报错。
    try:
        parse_sighting("xyz=(1,2,3) mm")
    except SightingParseError as error:
        assert "物料" in str(error), error
    else:
        raise AssertionError("缺物料编号必须报 SightingParseError")

    # 空行和无关日志行不是观测。
    assert try_parse_sighting("") is None
    assert try_parse_sighting("[INFO] 服务已启动") is None
    assert try_parse_sighting(_SAMPLE) is not None

    print(f"物料观测解析自测通过：{sighting.summary()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
