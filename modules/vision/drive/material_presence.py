"""是否有物料：在锁定区域里判断"到底有没有物料"。

输入是 YOLO 的检测框（``modules.vision.drive.material_yolo`` 的输出，任何带
``name`` / ``score`` / ``x1,y1,x2,y2`` / ``width`` / ``height`` / ``center`` 的对象都行），
输出一个明确的"是/否"+ 数量 + 代表框 + 人话理由。

按顺序过滤：类别 → 置信度 → 面积 → 是否落在锁定区域内。
``reason`` 会告诉你被哪一条挡住了，避免"没识别出来"这种没法排查的结论。

命令行自测::

    python -m modules.vision.drive.material_presence     # 合成检测框自测，不需要摄像头/模型
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, Sequence

from modules.vision.drive.material_yolo import MATERIAL_CLASS_NAMES

#: 判定"没有物料"时给出的理由。
REASON_OK = "ok"
REASON_NO_HIT = "no-hit"
REASON_NON_MATERIAL = "non-material-class"
REASON_LOW_SCORE = "below-score"
REASON_TOO_SMALL = "box-too-small"
REASON_OUTSIDE = "outside-region"


@dataclass(frozen=True)
class PresenceResult:
    """是否检到物料，以及为什么。"""

    present: bool
    count: int
    best: object | None
    reason: str

    def __bool__(self) -> bool:
        return bool(self.present)


def _box_area(detection) -> float:
    return float(detection.width) * float(detection.height)


class MaterialPresence:
    """Turn raw detections into a yes/no answer with a reason.

    Parameters:
        materials: class names that count as materials.
        min_score: minimum confidence kept.
        min_area_px: minimum box area; filters out speckle detections.
        require_material_class: when ``False``, any class counts (useful when
            testing a single-class obstacle model).
        region_mode: how a locked region is applied —
            ``"center"`` (box centre inside the circle, +``region_margin`` px),
            ``"intersect"`` (box overlaps the circle's bounding box),
            ``"any"`` (ignore the region).
        pick: which accepted hit becomes ``best`` —
            ``"highest_score"`` or ``"nearest"`` (closest centre to the region).
    """

    def __init__(
        self,
        *,
        materials: Sequence[str] = MATERIAL_CLASS_NAMES,
        min_score: float = 0.25,
        min_area_px: float = 200.0,
        require_material_class: bool = True,
        region_mode: str = "center",
        region_margin: float = 0.0,
        pick: str = "highest_score",
    ) -> None:
        if region_mode not in ("center", "intersect", "any"):
            raise ValueError("region_mode must be 'center', 'intersect' or 'any'")
        if pick not in ("highest_score", "nearest"):
            raise ValueError("pick must be 'highest_score' or 'nearest'")
        if min_score < 0 or min_area_px < 0:
            raise ValueError("min_score and min_area_px must not be negative")
        self.materials = tuple(materials)
        self.min_score = float(min_score)
        self.min_area_px = float(min_area_px)
        self.require_material_class = bool(require_material_class)
        self.region_mode = region_mode
        self.region_margin = float(region_margin)
        self.pick = pick

    def check(self, hits: Iterable, region=None) -> PresenceResult:
        """Decide whether any detection counts as a material in ``region``."""
        hits = list(hits)
        if not hits:
            return PresenceResult(False, 0, None, REASON_NO_HIT)

        accepted: list = []
        rejected: dict[str, int] = {}
        for hit in hits:
            reason = self._reject_reason(hit, region)
            if reason is None:
                accepted.append(hit)
            else:
                rejected[reason] = rejected.get(reason, 0) + 1

        if not accepted:
            reason = max(rejected, key=rejected.get)
            return PresenceResult(False, 0, None, reason)
        return PresenceResult(True, len(accepted), self._pick_best(accepted, region), REASON_OK)

    # ------------------------------------------------------------------ 内部
    def _reject_reason(self, hit, region) -> str | None:
        """Return the reason to reject ``hit``, or ``None`` to accept it."""
        if self.require_material_class and hit.name not in set(self.materials):
            return REASON_NON_MATERIAL
        if hit.score < self.min_score:
            return REASON_LOW_SCORE
        if _box_area(hit) < self.min_area_px:
            return REASON_TOO_SMALL
        if region is not None and not self._inside(hit, region):
            return REASON_OUTSIDE
        return None

    def _inside(self, hit, region) -> bool:
        if self.region_mode == "any":
            return True
        if self.region_mode == "center":
            center_x, center_y = hit.center
            return region.contains(center_x, center_y, margin=self.region_margin)
        # intersect：框和圆的外接矩形有重叠就算
        x1, y1, x2, y2 = region.bbox
        margin = self.region_margin
        return not (
            hit.x2 < x1 - margin or hit.x1 > x2 + margin or hit.y2 < y1 - margin or hit.y1 > y2 + margin
        )

    def _pick_best(self, accepted: list, region):
        if self.pick == "highest_score" or region is None:
            return max(accepted, key=lambda hit: hit.score)
        region_x, region_y = region.center
        return min(
            accepted,
            key=lambda hit: (hit.center[0] - region_x) ** 2 + (hit.center[1] - region_y) ** 2,
        )


# ---------------------------------------------------------------------- 自测
def _synthetic_hits():
    from modules.vision.drive.yolov8 import Detection

    return {
        "good": Detection(x1=280.0, y1=200.0, x2=360.0, y2=280.0, score=0.88, class_id=0, name="red"),
        "weak": Detection(x1=280.0, y1=200.0, x2=360.0, y2=280.0, score=0.10, class_id=3, name="green"),
        "tiny": Detection(x1=300.0, y1=220.0, x2=306.0, y2=226.0, score=0.70, class_id=2, name="blue"),
        "other": Detection(x1=280.0, y1=200.0, x2=360.0, y2=280.0, score=0.90, class_id=6, name="target-1"),
        "far": Detection(x1=10.0, y1=10.0, x2=90.0, y2=90.0, score=0.85, class_id=1, name="yellow"),
    }


def _self_test() -> int:
    from modules.vision.drive.hough_lock import LockedRegion

    hits = _synthetic_hits()
    region = LockedRegion(center_x=320.0, center_y=240.0, radius=80.0)
    presence = MaterialPresence()
    problems: list[str] = []

    def expect(label: str, result: PresenceResult, *, present: bool, reason: str) -> None:
        if result.present is not present or result.reason != reason:
            problems.append(
                f"{label}: 期望 present={present} reason={reason}，"
                f"实际 present={result.present} reason={result.reason}"
            )

    expect("范围内有物料", presence.check([hits["good"]], region), present=True, reason=REASON_OK)
    expect("空输入", presence.check([], region), present=False, reason=REASON_NO_HIT)
    expect("置信度太低", presence.check([hits["weak"]], region), present=False, reason=REASON_LOW_SCORE)
    expect("框太小", presence.check([hits["tiny"]], region), present=False, reason=REASON_TOO_SMALL)
    expect("不是物料类", presence.check([hits["other"]], region), present=False, reason=REASON_NON_MATERIAL)
    expect("在范围外", presence.check([hits["far"]], region), present=False, reason=REASON_OUTSIDE)

    mixed = presence.check([hits["good"], hits["weak"], hits["other"]], region)
    if not mixed.present or mixed.count != 1:
        problems.append(f"混合输入应只算 1 个物料，实际 present={mixed.present} count={mixed.count}")
    if mixed.best is not hits["good"]:
        problems.append("best 应指向被接受的那个物料框")

    nearest = MaterialPresence(pick="nearest").check([hits["good"], hits["far"]], region, )
    if nearest.present and nearest.best is not hits["good"]:
        problems.append("pick='nearest' 时应选离锁定圆心最近的那个框")

    everything_ok = MaterialPresence(require_material_class=False).check([hits["other"]], region)
    if not everything_ok.present:
        problems.append("require_material_class=False 时 target-1 也应算命中")

    no_region = presence.check([hits["far"]], None)
    if not no_region.present:
        problems.append("不给锁定区域时不应按位置过滤")

    print(f"合成检测框：{len(hits)} 个；判定结果 reason 分布已逐条检查")
    if problems:
        print("\n自测失败：")
        for problem in problems:
            print(f"  · {problem}")
        return 1
    print("\n自测通过：类别 / 置信度 / 面积 / 锁定区域四种过滤与 best 选择都正确。")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="是否有物料：对检测框做存在性判定")
    parser.parse_args(argv)
    return _self_test()


if __name__ == "__main__":
    raise SystemExit(main())
