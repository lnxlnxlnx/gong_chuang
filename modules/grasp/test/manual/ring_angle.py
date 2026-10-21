"""把视野里的环心解算成**各电机的绝对位置**（只看不动）。

    python test.py   # 选 27

这是 ``robot.grasp.plan_ring()`` 的真机入口：定位环心 → 用实测雅可比与命令里程表
解算"底座/伸缩各该走到哪个绝对位置"，**一帧都不驱动电机**。打印出来的是：

- 环号、模型分数、环心读数（相机系毫米）；
- **各轴绝对角度**（电机轴口径，与 ``move_axis_to_motor_degrees`` 一致）与脉冲；
- 预测残差，以及"存下这份 plan、晚点再走"的用法。

典型流程是"先算、后走"：在这个入口（或程序里）算出 plan → 中间去抓物料、摆料 →
最后 ``plan.apply(arm)`` 或逐轴发 ``move_axis_to_motor_degrees``。

## 为什么这次可能要 ``--scale-mm-per-px``

环心读数的比例尺必须是**物理毫米**（雅可比是按物理毫米标的）。正常来源有两种：
针孔标定 ``table_depth_mm/focal_px``，或 ``ring_target.outer_diameter_mm``
（已知外径 / 模型框，随包配置里已经填了 95 mm）。两者都没有时读数会退回
``station.mm_per_px``——那是给**固定相机**标的值，装在臂上会差好几倍，本入口会照实
拒绝并说明；明确知道换算关系时用 ``--scale-mm-per-px`` 显式给一次::

    python -m modules.grasp.test.manual.ring_angle --scale-mm-per-px 0.183

## 它不是闭环

一次解算用的是**一帧**的线性化雅可比。转完大角度之后雅可比会变，所以要复核：
再调一次 ``plan_ring``，残差收敛了就到位了。真正的闭环是菜单 21（``align_xy``）。
"""

from __future__ import annotations

MENU_ENTRY = (27, "环心 → 各轴绝对位置")

import argparse

from modules.grasp.drive.servo_model import ServoError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="定位环心，解算各轴绝对位置（只看不动）"
    )
    parser.add_argument(
        "--scale-mm-per-px", type=float, default=None,
        help="与 jacobian 一致的毫米/像素；环心比例尺与它不同时用它换算读数",
    )
    parser.add_argument(
        "--digit", type=int, default=None,
        help="只接受这个环号（1/2/3）；画面里是别的号就报错",
    )
    return parser.parse_args()


def _report(plan) -> None:
    """把一份计划打成现场能照着做的几行。"""
    print(plan.summary())
    print(
        f"  各电机绝对位置：底座 {plan.pulses['base']:+d} 脉冲"
        f"（{plan.degrees['base']:+.1f}° 电机轴 / "
        f"{plan.base_output_degrees:+.1f}° 输出轴）；"
        f"伸缩 {plan.pulses['extension']:+d} 脉冲"
        f"（{plan.degrees['extension']:+.1f}°）"
    )
    print(f"  {plan.note}")
    print("  稍后执行（先存下这份 plan，中间去干别的）：plan.apply(arm)；只转底座："
          'plan.apply(arm, axes=("base",))')


def _print_state(arm) -> None:
    """当前里程表：解算用的就是这两个数，现场核对用。"""
    base = arm.turns_state.of("base")
    radial = arm.turns_state.of("extension")
    print(
        f"  当前里程表：底座 {base.pulses:+d} 脉冲（{base.motor_degrees:+.1f}°），"
        f"伸缩 {radial.pulses:+d} 脉冲（{radial.motor_degrees:+.1f}°）"
    )


def run(scale_mm_per_px=None, require_digit=None) -> int:
    from modules.grasp.runtime import Grasp

    with Grasp.open() as grasp:
        arm = grasp._require_resources()[0]
        print("环心 → 各轴绝对位置（只看不动）；回车再测一次，q 退出")
        _print_state(arm)
        while True:
            choice = input("回车=再测一次，q=退出：").strip().lower()
            if choice in ("q", "quit"):
                return 0
            try:
                plan = grasp.plan_ring(
                    require_digit=require_digit, scale_mm_per_px=scale_mm_per_px
                )
            except ServoError as error:
                print(f"  解算不了：{error}")
                continue
            _report(plan)
            _print_state(arm)


def main() -> None:
    args = parse_args()
    raise SystemExit(
        run(
            scale_mm_per_px=args.scale_mm_per_px,
            require_digit=args.digit,
        )
    )


if __name__ == "__main__":
    main()
