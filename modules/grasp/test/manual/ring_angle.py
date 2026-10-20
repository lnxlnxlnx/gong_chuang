"""把视野里的环心解算成底座的**绝对角度**（只看不动）。

    python test.py   # 选 27

这是 ``robot.grasp.plan_ring()`` 的真机入口：定位环心 → 用实测雅可比与命令里程表
解算"底座该转到哪"，**一帧都不驱动电机**。打印出来的是：

- 环号、模型分数、环心读数（相机系毫米）；
- **底座绝对角度**（电机轴口径，与 ``move_axis_to_motor_degrees`` 一致）与输出轴角度；
- 伸缩的建议脉冲（只转底座时，爪子到了环的角度但不在同一条半径上）；
- 预测残差，以及可以直接复制运行的一条命令。

## 为什么这次可能要 ``--scale-mm-per-px``

环心读数的比例尺必须与标定雅可比时用的那把尺一致，否则角度会整体差一个倍数。
``calibration.json`` 里没有 ``focal_px`` / ``table_depth_mm`` 时，环心读数只能退回
``station.mm_per_px``——那是给**固定相机**标的值，和雅可比不是同一把尺。这时本入口
会照实拒绝并说明；知道两者关系时用 ``--scale-mm-per-px`` 显式换算一次::

    python -m modules.grasp.test.manual.ring_angle --scale-mm-per-px 0.111

标定好 ``focal_px`` / ``table_depth_mm``（菜单 25）之后就不需要这个参数了。

## 它不是闭环

一次解算用的是**一帧**的线性化雅可比。转完大角度之后雅可比会变，所以要复核：
再跑一次本入口（或再调一次 ``plan_ring``），残差收敛了就到位了。真正的闭环是
菜单 21（``align_xy``，逐步伺服到目标读数）。
"""

from __future__ import annotations

MENU_ENTRY = (27, "环心 → 底座绝对角度")

import argparse

from modules.grasp.drive.servo_model import ServoError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="定位环心，解算底座绝对角度（只看不动）"
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
        f"  底座绝对角度：电机轴 {plan.base_motor_degrees:+.1f}°"
        f"（{plan.base_motor_pulses:+d} 脉冲）"
        f"，输出轴 {plan.base_output_degrees:+.1f}°"
    )
    print(
        f"  伸缩：{plan.radial_pulses:+d} 脉冲 → 绝对值 {plan.radial_motor_pulses:+d}"
        f"（{plan.radial_motor_degrees:+.1f}°）"
    )
    print(f"  {plan.note}")
    print(
        '  命令：arm.move_axis_to_motor_degrees("base", '
        f'{plan.base_motor_degrees:.1f})'
    )


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
        print("环心 → 底座绝对角度（只看不动）；回车再测一次，q 退出")
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
