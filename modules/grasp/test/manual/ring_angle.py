"""把视野里的环心解算成**各电机的绝对位置**，或者直接闭环对准过去。

    python test.py   # 选 27

两种用法（进去按键选）：

- ``p``：**只算不动**——定位环心 → 用实测雅可比与命令里程表解算"底座/伸缩各该走到
  哪个绝对位置"，一帧都不驱动电机。打印环号、各轴绝对角度/脉冲、预测残差。
- ``a``：**闭环对准**——``grasp.align_ring()``：算一步、走一步、再看一帧，直到进容差。
  每一步都会打印"实测挪了多少 / 按雅可比预测多少 / 响应比"。

**响应比**是这里最有用的一个数：``1`` 说明配置的雅可比与实际相符；接近 ``0.5`` 或
``2`` 说明雅可比过期（换高度、换装法）或环心读数的比例尺与它不是同一把尺——那就是
"转一次再算一次还是偏"的根源，闭环会在线折算过去，但该重标还得重标（菜单 21 的
``--calibrate``）。

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

from modules.grasp.drive.visual_servo_model import VisualServoError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="定位环心：只算各轴绝对位置，或闭环对准过去"
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
    """真机循环：``p`` 只算、``a`` 闭环对准（会动）。"""
    from modules.grasp.runtime import Grasp

    with Grasp.open() as grasp:
        arm = grasp._require_resources()[0]
        print("环心 → 各轴绝对位置：p = 只算（不动），a = 闭环对准（会动），q 退出")
        _print_state(arm)
        while True:
            choice = input("p=算一次  a=闭环对准  q=退出：").strip().lower()
            if choice in ("q", "quit"):
                return 0
            if choice == "a":
                try:
                    result = grasp.align_ring(
                        require_digit=require_digit, scale_mm_per_px=scale_mm_per_px
                    )
                except VisualServoError as error:
                    print(f"  对准不了：{error}")
                    continue
                for step in result.steps:
                    print("  " + step.line())
                print(result.summary())
                if result.plan is not None:
                    print(
                        "  存下这份结果就能之后回来："
                        f"result.pulses = {result.pulses}；"
                        '恢复：result.apply(arm)'
                    )
                _print_state(arm)
                continue
            try:
                plan = grasp.plan_ring(
                    require_digit=require_digit, scale_mm_per_px=scale_mm_per_px
                )
            except VisualServoError as error:
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
