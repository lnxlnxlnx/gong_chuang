"""项目应用入口：默认预览物料定位结果。"""

from __future__ import annotations

from modules.vision.app.demo import demo_material


def main() -> None:
    """Run the material preview.

    The other demos live with their module: ``modules.arm.app.demo``,
    ``modules.servo.app.demo`` and ``modules.chassis.app`` (wheel routes).
    """
    demo_material()


if __name__ == "__main__":
    main()
