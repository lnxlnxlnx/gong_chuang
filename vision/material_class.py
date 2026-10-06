"""是哪一种物料：把检测框翻译成"哪一种物料"+ 中文标签 + 置信度。

职责单一：只回答"是哪一种"。"有没有"在 ``vision.material_presence``，
框从哪来在 ``vision.material_yolo``，多远在 ``vision.material_distance``。

类别名来自 ``vision.yolov8.DEFAULT_CLASS_NAMES`` 里的六个 ``obj-*``；
中文标签可以用 :data:`DEFAULT_MATERIAL_LABELS`，也可以自己传 ``labels``
（换模型、换类别表时只改这里）。

命令行自测::

    python -m vision.material_class     # 合成检测框自测，不需要摄像头/模型
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from vision.material_yolo import MATERIAL_CLASS_NAMES

#: 比赛六种物料的中文标签，用于打印 / 画在画面上。
DEFAULT_MATERIAL_LABELS: dict[str, str] = {
    "obj-red": "红色物料",
    "obj-green": "绿色物料",
    "obj-blue": "蓝色物料",
    "obj-yellow": "黄色物料",
    "obj-light blue": "浅蓝物料",
    "obj-black": "黑色物料",
}

#: 类别不在标签表里时用的中文名。
UNKNOWN_MATERIAL_LABEL = "未知物料"


@dataclass(frozen=True)
class MaterialCandidate:
    """一个候选物料：类别名、中文标签、置信度。"""

    name: str
    label: str
    score: float


@dataclass(frozen=True)
class ClassResult:
    """分类结果：最像的那一种 + 全部候选（按置信度从高到低）。"""

    found: bool
    name: str | None
    label: str | None
    score: float
    candidates: tuple[MaterialCandidate, ...] = ()

    @property
    def names(self) -> tuple[str, ...]:
        """Every distinct candidate class in confidence order."""
        seen: list[str] = []
        for candidate in self.candidates:
            if candidate.name not in seen:
                seen.append(candidate.name)
        return tuple(seen)


class MaterialClassifier:
    """Pick the material class of a detection set.

    Parameters:
        materials: class names that count as materials.
        labels: class name -> Chinese label; defaults to
            :data:`DEFAULT_MATERIAL_LABELS`.
        min_score / min_area_px: same filters as :class:`vision.material_presence`
            so both modules agree on what a valid box is.
        pick: how the "top" candidate is chosen — currently always by confidence
            (kept as a parameter so a nearest-to-ring rule can be added later).
    """

    def __init__(
        self,
        *,
        materials: Sequence[str] = MATERIAL_CLASS_NAMES,
        labels: Mapping[str, str] | None = None,
        min_score: float = 0.25,
        min_area_px: float = 200.0,
    ) -> None:
        self.materials = tuple(materials)
        self.labels = dict(DEFAULT_MATERIAL_LABELS if labels is None else labels)
        self.min_score = float(min_score)
        self.min_area_px = float(min_area_px)

    def label_for(self, name: str) -> str:
        """Return the Chinese label of a class name (``未知物料`` when unknown)."""
        return self.labels.get(name, UNKNOWN_MATERIAL_LABEL)

    def describe(self, hit) -> MaterialCandidate:
        """Turn one detection into a candidate (no filtering)."""
        return MaterialCandidate(name=hit.name, label=self.label_for(hit.name), score=float(hit.score))

    def classify(self, hits: Iterable) -> ClassResult:
        """Return the best material class among ``hits`` (or an empty result)."""
        candidates = sorted(
            (
                self.describe(hit)
                for hit in hits
                if hit.name in set(self.materials)
                and hit.score >= self.min_score
                and float(hit.width) * float(hit.height) >= self.min_area_px
            ),
            key=lambda candidate: candidate.score,
            reverse=True,
        )
        if not candidates:
            return ClassResult(found=False, name=None, label=None, score=0.0, candidates=())
        best = candidates[0]
        return ClassResult(
            found=True,
            name=best.name,
            label=best.label,
            score=best.score,
            candidates=tuple(candidates),
        )


# ---------------------------------------------------------------------- 自测
def _synthetic_hits():
    from vision.yolov8 import Detection

    return {
        "red": Detection(x1=280.0, y1=200.0, x2=360.0, y2=280.0, score=0.90, class_id=0, name="obj-red"),
        "blue": Detection(x1=100.0, y1=200.0, x2=180.0, y2=280.0, score=0.80, class_id=2, name="obj-blue"),
        "weak": Detection(x1=280.0, y1=200.0, x2=360.0, y2=280.0, score=0.05, class_id=1, name="obj-green"),
        "other": Detection(x1=280.0, y1=200.0, x2=360.0, y2=280.0, score=0.95, class_id=6, name="target-1"),
        "alien": Detection(x1=280.0, y1=200.0, x2=360.0, y2=280.0, score=0.70, class_id=3, name="obj-purple"),
    }


def _self_test() -> int:
    hits = _synthetic_hits()
    classifier = MaterialClassifier()
    problems: list[str] = []

    best = classifier.classify([hits["red"], hits["blue"], hits["weak"], hits["other"]])
    if not best.found or best.name != "obj-red" or best.label != "红色物料":
        problems.append(f"应判为 obj-red/红色物料，实际 {best.name}/{best.label}")
    if abs(best.score - 0.90) > 1e-9:
        problems.append(f"置信度应为 0.90，实际 {best.score}")
    if best.names != ("obj-red", "obj-blue"):
        problems.append(f"候选类别应为 (obj-red, obj-blue)，实际 {best.names}")
    if len(best.candidates) != 2:
        problems.append(f"候选应过滤掉低置信度与非物料，实际 {len(best.candidates)} 个")

    empty = classifier.classify([])
    if empty.found or empty.name is not None or empty.candidates:
        problems.append("空输入的 classify 应返回 found=False 且没有任何候选")

    only_other = classifier.classify([hits["other"]])
    if only_other.found:
        problems.append("非物料类别不应被当成物料")

    # 换过类别表时，未知类别要能落到"未知物料"而不是崩掉
    alien = MaterialClassifier(materials=("obj-purple",)).classify([hits["alien"]])
    if not alien.found or alien.name != "obj-purple" or alien.label != UNKNOWN_MATERIAL_LABEL:
        problems.append(f"未知类别应落到 {UNKNOWN_MATERIAL_LABEL}，实际 {alien.label}")

    described = classifier.describe(hits["blue"])
    if described.label != "蓝色物料" or described.score != 0.80:
        problems.append("describe() 应直接给出候选（不做过滤）")

    print(f"合成检测框：{[hit.name for hit in hits.values()]}")
    print(f"最高置信度物料：{best.label}（{best.name}）score={best.score:.2f}；候选 {best.names}")
    if problems:
        print("\n自测失败：")
        for problem in problems:
            print(f"  · {problem}")
        return 1
    print("\n自测通过：类别判定、中文标签、候选排序与未知类别回退都正确。")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="是哪一种物料：检测框 -> 类别 + 中文标签")
    parser.parse_args(argv)
    return _self_test()


if __name__ == "__main__":
    raise SystemExit(main())
