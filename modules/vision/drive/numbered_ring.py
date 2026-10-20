"""视野里那个环：环心在哪、它是几号（模型给的，不是代码猜的）。

    python test.py   # 选 26

相机装在臂上、镜头离纸很近时，画面里通常**只有一个环**——"同时看到三个"那种检测
在这个取景下必然失败，而"只看到一个环"本来就不是错误。本模块只为这种取景服务。

## 两件事都由模型给

随包的 ``models/best.onnx`` 有 7 类：四色物料 + ``tg1`` / ``tg2`` / ``tg3``。后三个就是
带数字的靶标，所以一帧前向同时给出：

- **环心** = 命中框的中心（模型把整个靶标图案框住，框心即环心）；
- **环号** = 命中的类别，查 ``config.json`` 的 ``ring_target.classes``。

因此这里**没有**整帧 Canny → 椭圆观测 → 聚类 → 按支撑数排序那一套，也**没有**
"环里墨渍占比够不够"的墨迹判据：那些是已经训练进模型的能力，代码里再算一遍只会
多出一套会和模型结论打架的规则。

## 框贴边就**不是**环心

模型给的是框，框心只有在"整个靶标都落在画面里"时才等于环心：框被画面裁掉一截时，
中心会跟着往画面内侧挪（实测偏 80 px 以上）。所以贴边的框照样返回（``clipped=True``），
但 :meth:`RingTarget.reading` 返回 ``None``——宁可说"这一帧不能对准"，也不给一个
差十几毫米、却看起来完全正常的数。

## 比例尺只有两种来源

优先 ``table_depth_mm / focal_px``（台面的针孔标定，与 ``RegionReport`` 的毫米是同一套
单位），没有就退回 ``station.mm_per_px``；两者都没有就**不报毫米**。刻意不拿"模型框的
像素宽度"当已知直径去凑：框的大小由回归决定，实测同一条靶标能从 449 px 变到 580 px，
拿它当比例尺等于把一个不稳定的数引进读数。

毫米只影响伺服每一步走多远，不影响"最后对准没对准"——``align_xy`` 每步都用新的一帧
重算（见 ``modules/grasp/drive/eye_in_hand.py``），所以比例尺稍粗是可以接受的。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np

from modules.vision.config import RingTargetSettings, StationGeometrySettings

if TYPE_CHECKING:
    from modules.vision.drive.yolov8 import Detection

#: 框贴到画面边缘多少像素以内就当作"被裁掉"。
BORDER_TOLERANCE_PX = 1.0


@dataclass(frozen=True)
class RingTarget:
    """**视野里的那一个环**：环心在哪、几号、能不能直接拿去对准。"""

    #: 环心（像素）：命中的检测框中心。
    center_px: Tuple[float, float]
    #: 环的等效半径（像素）：框长边的一半，只用于画图。
    radius_px: float
    #: 环号（来自 ``ring_target.classes``）。
    digit: int
    #: 模型给这个框的置信度。
    score: float
    #: 框是否被画面边缘裁掉。裁掉时框心**不是**环心。
    clipped: bool
    #: 环心在**相机系**的毫米坐标；没有比例尺时是 ``None``。
    x_mm: Optional[float]
    y_mm: Optional[float]
    #: 本帧实际用的比例尺（毫米/像素）与来源（``depth/focal`` / ``station.mm_per_px``）。
    scale_mm_per_px: Optional[float] = None
    scale_source: str = ""
    #: 人话说明：这一帧为什么是它、比例尺从哪来。
    detail: str = ""

    @property
    def usable(self) -> bool:
        """能不能拿去对准：框没被裁掉，而且毫米坐标算出来了。"""
        return not self.clipped and self.x_mm is not None and self.y_mm is not None

    def reading(self) -> Optional[Tuple[float, float]]:
        """相机系 ``(x_mm, y_mm)``，可直接喂给 ``align_xy(reading=...)``。"""
        if not self.usable:
            return None
        return (float(self.x_mm), float(self.y_mm))

    def summary(self) -> str:
        where = (
            f"xyz=({self.x_mm:+.1f},{self.y_mm:+.1f})mm"
            if self.x_mm is not None and self.y_mm is not None
            else "无毫米（缺比例尺）"
        )
        shape = "**框被画面裁掉，框心不是环心**" if self.clipped else "整框在画面里"
        scale = (
            f"{self.scale_mm_per_px:.4f} mm/px（{self.scale_source}）"
            if self.scale_mm_per_px is not None
            else "无比例尺"
        )
        return (
            f"环 {self.digit}（模型 {self.score:.2f}）："
            f"@({self.center_px[0]:.0f},{self.center_px[1]:.0f}) r={self.radius_px:.0f}px "
            f"{where}，{shape}，{scale}"
        )


class RingTargetLocator:
    """一帧 BGR → **视野里那一个环**的环心（像素 + 相机系毫米）。

    本类**不拥有检测器**：一个会话里的模型只加载一次，靶标与物料共用同一个检测器
    （见 :meth:`modules.vision.runtime.Vision._ring_target_locator`）；单独跑一张图时
    由调用方自己建一个。

    选哪一个框：优先"没被画面裁掉"的（那才可能是真环心），再比置信度。

    Parameters:
        detector: 任何一个 ``detect(frame) -> list[Detection]`` 的检测器。
        settings: ``config.json`` 的 ``ring_target`` 块（类别名 → 环号）。
        geometry: 兜底比例尺（``station.mm_per_px``）。
        focal_px / table_depth_mm: 台面的针孔标定；两个都有时优先用它换算毫米。
    """

    def __init__(
        self,
        detector,
        *,
        settings: Optional[RingTargetSettings] = None,
        geometry: Optional[StationGeometrySettings] = None,
        focal_px: Optional[float] = None,
        table_depth_mm: Optional[float] = None,
    ) -> None:
        if detector is None:
            raise ValueError("RingTargetLocator 需要检测器；模型只该由会话建一次")
        self.detector = detector
        self.settings = settings if settings is not None else RingTargetSettings()
        self.geometry = geometry if geometry is not None else StationGeometrySettings()
        self.focal_px = focal_px
        self.table_depth_mm = table_depth_mm
        #: 环号表（类别名 → 环号）；构造时定死，逐帧不再回读配置。
        self.digits: Dict[str, int] = dict(self.settings.classes)
        #: 最近一次 ``detect`` 走到了哪一步（给真机入口打印"为什么没找到"）。
        self.last_diagnosis = ""
        self._check_class_table()

    def _check_class_table(self) -> None:
        """配置里的环类别必须真的在检测器的类别表里，否则当场报错。

        名字写错（``tg_1``）不会让程序挂，只会一直"画面里没有环"——那是现场最费时间的
        故障：看起来像模型没认出来，其实是配置少打了一个下划线。所以这里拒绝启动。
        """
        table = getattr(self.detector, "class_names", None)
        if not table:
            return
        missing = [name for name in self.digits if name not in set(table)]
        if missing:
            raise ValueError(
                "ring_target.classes 里的 "
                + "、".join(missing)
                + " 不在模型类别表里（"
                + "、".join(str(name) for name in table)
                + "）；检查 config.json 或模型文件的 names 元数据"
            )

    # ------------------------------------------------------------- 主流程
    def detect(self, frame: np.ndarray) -> Optional[RingTarget]:
        """找出视野里最可能是"那个环"的框；一个都没有时返回 ``None``。"""
        if frame is None:
            raise ValueError("frame 不能是 None")
        height, width = frame.shape[:2]
        hits = [hit for hit in self.detector.detect(frame) if hit.name in self.digits]
        if not hits:
            self.last_diagnosis = (
                "画面里没有环：模型给的框里没有 " + " / ".join(self.digits)
            )
            return None
        chosen = max(hits, key=lambda hit: self._rank(hit, width, height))
        return self._build(chosen, width, height, len(hits))

    # ------------------------------------------------------------- 内部
    @staticmethod
    def _rank(hit: "Detection", width: int, height: int) -> Tuple[int, float]:
        """先看框有没有被裁掉，再比置信度。"""
        return (0 if _touches_border(hit, width, height) else 1, float(hit.score))

    def _build(
        self, hit: "Detection", width: int, height: int, count: int
    ) -> RingTarget:
        clipped = _touches_border(hit, width, height)
        center_x, center_y = hit.center
        radius = max(0.0, max(hit.width, hit.height) * 0.5)
        scale, source = self._millimetres_per_pixel()
        x_mm = y_mm = None
        if scale is not None:
            x_mm = (float(center_x) - width / 2.0) * float(scale)
            y_mm = (float(center_y) - height / 2.0) * float(scale)
        detail = f"模型给了 {count} 个环框"
        if clipped:
            detail += "；选中的框贴画面边，框心不是环心，不给可对准读数"
        if scale is None:
            detail += "；没有比例尺（depth+focal / station.mm_per_px），给不出毫米"
        else:
            detail += f"；比例尺 {scale:.4f} mm/px 来自 {source}"
        self.last_diagnosis = (
            f"{count} 个环框，选中环 {self.digits[hit.name]}"
            f"（{hit.name} {float(hit.score):.2f}，"
            f"框 ({hit.x1:.0f},{hit.y1:.0f})-({hit.x2:.0f},{hit.y2:.0f})）"
            f"{'，贴边' if clipped else ''}"
        )
        return RingTarget(
            center_px=(float(center_x), float(center_y)),
            radius_px=radius,
            digit=int(self.digits[hit.name]),
            score=float(hit.score),
            clipped=clipped,
            x_mm=x_mm,
            y_mm=y_mm,
            scale_mm_per_px=None if scale is None else float(scale),
            scale_source=source,
            detail=detail,
        )

    def _millimetres_per_pixel(self) -> Tuple[Optional[float], str]:
        """本帧的比例尺与来源，优先级：针孔标定 > 配置固定值。"""
        focal = self.focal_px
        depth = self.table_depth_mm
        if focal is not None and depth is not None and focal > 0 and depth > 0:
            return (float(depth) / float(focal), "depth/focal")
        if self.geometry.mm_per_px is not None:
            return (float(self.geometry.mm_per_px), "station.mm_per_px")
        return (None, "none")


def _touches_border(hit: "Detection", width: int, height: int) -> bool:
    """框有没有被画面边缘裁掉——被裁掉的框，中心已经不是环心了。"""
    return (
        hit.x1 <= BORDER_TOLERANCE_PX
        or hit.y1 <= BORDER_TOLERANCE_PX
        or hit.x2 >= width - BORDER_TOLERANCE_PX
        or hit.y2 >= height - BORDER_TOLERANCE_PX
    )


# ---------------------------------------------------------------------- 自测
def _self_test() -> int:
    """假检测器：验证"选哪个框、框心怎么变成读数、贴边为什么不给读数"。

    不需要模型也不需要摄像头：``FakeDetector`` 按给定框返回 ``Detection``，
    正是"一帧 → 几个框"那一层接口。
    """
    from modules.vision.drive.yolov8 import Detection

    def box(name, score, x1, y1, x2, y2):
        return Detection(x1=x1, y1=y1, x2=x2, y2=y2, score=score, class_id=4, name=name)

    size = (1280, 720)

    class FakeDetector:
        def __init__(self, hits, *, class_names=("red", "tg1", "tg2", "tg3")):
            self.hits = list(hits)
            self.class_names = list(class_names)
            self.calls = 0

        def detect(self, _frame):
            self.calls += 1
            return list(self.hits)

    problems = []

    def run(hits, **kwargs):
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        return RingTargetLocator(FakeDetector(hits), **kwargs).detect(frame)

    # 1) 画面正中、整框可见：环心在光轴上，环号来自类别
    target = run([box("tg2", 0.95, 640 - 260, 360 - 258, 640 + 260, 360 + 258)],
                 geometry=StationGeometrySettings(mm_per_px=0.5))
    if target is None:
        problems.append("画面里有一个整框可见的环，应当给出环心")
    else:
        reading = target.reading()
        if reading is None or abs(reading[0]) > 0.5 or abs(reading[1]) > 0.5:
            problems.append(f"环心应当在光轴上，实际 {reading}")
        if target.digit != 2:
            problems.append(f"tg2 应当是 2 号环，实际 {target.digit}")
        if target.clipped:
            problems.append("整框在画面里，不该标成贴边")
        print(target.summary())

    # 2) 框贴边：照样返回，但**没有**可对准读数
    clipped = run([box("tg1", 0.9, -40.0, 60.0, 220.0, 620.0)],
                  geometry=StationGeometrySettings(mm_per_px=0.5))
    if clipped is None:
        problems.append("贴边的环也要返回（好让人看到「看到了但用不了」）")
    elif not clipped.clipped or clipped.reading() is not None:
        problems.append(f"贴边的框不该给出可对准读数：{clipped.summary()}")

    # 3) 整框可见的低分 vs 贴边的高分：选整框那个
    chosen = run(
        [
            box("tg1", 0.99, -10.0, 60.0, 240.0, 620.0),          # 贴边、分更高
            box("tg3", 0.40, 600.0, 100.0, 900.0, 400.0),          # 整框、分低
        ],
        geometry=StationGeometrySettings(mm_per_px=0.5),
    )
    if chosen is None or chosen.digit != 3:
        problems.append(f"应当选整框可见的那个环，实际 {chosen and chosen.digit}")

    # 4) 没有环框：返回 None，并把原因写进诊断
    locator = RingTargetLocator(FakeDetector([box("red", 0.9, 0.0, 0.0, 50.0, 50.0)]))
    if locator.detect(np.zeros((size[1], size[0], 3), dtype=np.uint8)) is not None:
        problems.append("只有物料框时不该报出一个环")
    if "没有环" not in locator.last_diagnosis:
        problems.append(f"没找到环时诊断要说清楚，实际 {locator.last_diagnosis!r}")

    # 5) 比例尺来源：针孔标定优先于配置固定值
    target = run(
        [box("tg2", 0.9, 290.0, 260.0, 590.0, 460.0)],   # 框心 (440, 360)，偏光轴 200 px
        geometry=StationGeometrySettings(mm_per_px=0.4),
        focal_px=800.0,
        table_depth_mm=400.0,       # 0.5 mm/px，而不是配置里的 0.4
    )
    if target is None or target.scale_source != "depth/focal":
        problems.append(f"有针孔标定时应当用它，实际 {target and target.scale_source}")
    elif abs(target.reading()[0] + 100.0) > 1.0:
        problems.append(f"0.5 mm/px 下偏 200 px 应当是 -100 mm，实际 {target.reading()}")

    # 6) 没配比例尺：环照样给出，但没有毫米
    target = run([box("tg2", 0.9, 290.0, 260.0, 590.0, 460.0)],
                 geometry=StationGeometrySettings(mm_per_px=None))
    if target is None or target.reading() is not None:
        problems.append("没有比例尺时不该给出可对准读数")

    # 7) 类别名写错：构造时就报错，而不是一直"没找到环"
    try:
        RingTargetLocator(
            FakeDetector([]),
            settings=RingTargetSettings(classes={"tg_1": 1}),
        )
    except ValueError as error:
        if "tg_1" not in str(error):
            problems.append(f"报错要指出错的那个类别名，实际 {error}")
    else:
        problems.append("环类别名不在模型类别表里时，构造就该失败")

    if problems:
        print("\n自测失败：")
        for problem in problems:
            print(f"  · {problem}")
        return 1
    print("\n自测通过：选框、框心换算、贴边拒绝、类别表校验都按约定处理。")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
