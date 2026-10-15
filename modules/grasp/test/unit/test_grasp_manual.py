"""真机抓取入口：标定清单必须一次列全，且没标定就不许打开硬件。"""

from __future__ import annotations

import unittest
from unittest.mock import Mock

from modules.arm.config import ArmConfig, AxisConfig
from modules.arm.drive.kinematics import CylindricalPose, TravelLimit
from modules.grasp.config import GraspConfig
from modules.grasp.drive.perception import CellObservation
from modules.grasp.test.manual.grasp import (
    MENU_ENTRY,
    calibration_report,
    run_grasp_once,
    run_observe_only,
)
from modules.vision.config import TargetSettings, VisionConfig
from modules.vision.drive.calibration import PlaneCalibration
from modules.servo.runtime import CLOSE_COMMAND, GripperReport


def calibrated_vision() -> VisionConfig:
    return VisionConfig(
        camera_ids={"windows": 1},
        targets=TargetSettings(
            material_diameter_mm=30.0,
            marker_diameter_mm=20.0,
            material_bgr=(200.0, 120.0, 60.0),
            marker_bgr=(245.0, 245.0, 245.0),
        ),
        focal_px=1012.0,
        table_depth_mm=300.0,
        plane=PlaneCalibration(
            center_x_mm=120.0, center_y_mm=-45.0, rotation_degrees=37.0, rms_mm=1.2, samples=12
        ),
    )


def calibrated_arm() -> ArmConfig:
    return ArmConfig(
        pulses_per_revolution=3200,
        rpm=30,
        acceleration=5,
        default_pulses=800,
        max_pulses=5000,
        axes={
            "base": AxisConfig(motor_id=5, reduction_ratio=5),
            "lift": AxisConfig(motor_id=6, mm_per_motor_rev=40.0),
            "extension": AxisConfig(motor_id=7, mm_per_motor_rev=12.5),
        },
        base_limit=TravelLimit(-110.0, 110.0),
        radial_limit=TravelLimit(40.0, 160.0),
        height_limit=TravelLimit(-5.0, 120.0),
        safe_pose=CylindricalPose(0.0, 40.0, 60.0),
    )


def calibrated_grasp() -> GraspConfig:
    return GraspConfig(marker_to_grasp_mm=-12.0)


class CalibrationReportTests(unittest.TestCase):
    def test_the_uncalibrated_repository_state_lists_every_missing_item(self) -> None:
        from modules.arm.config import load_arm_config
        from modules.grasp.config import load_grasp_config
        from modules.vision.config import load_vision_config

        missing = calibration_report(
            vision=load_vision_config(), arm=load_arm_config(), grasp=load_grasp_config()
        )
        self.assertGreater(len(missing), 5, missing)
        joined = "\n".join(missing)
        for keyword in ("平面标定", "focal_px", "table_depth_mm", "marker_diameter_mm",
                        "mm_per_motor_rev", "safe_pose", "limits", "marker_to_grasp_mm"):
            self.assertIn(keyword, joined, keyword)

    def test_a_fully_calibrated_setup_reports_nothing(self) -> None:
        self.assertEqual(
            calibration_report(
                vision=calibrated_vision(), arm=calibrated_arm(), grasp=calibrated_grasp()
            ),
            [],
        )

    def test_a_coarse_plane_fit_is_called_out_even_though_it_exists(self) -> None:
        import dataclasses

        coarse = dataclasses.replace(
            calibrated_vision(),
            plane=PlaneCalibration(
                center_x_mm=120.0, center_y_mm=-45.0, rotation_degrees=37.0, rms_mm=9.0, samples=4
            ),
        )
        missing = calibration_report(
            vision=coarse, arm=calibrated_arm(), grasp=calibrated_grasp()
        )
        self.assertEqual(len(missing), 1, missing)
        self.assertIn("残差", missing[0])

    def test_a_zero_marker_offset_is_flagged_because_it_is_almost_never_true(self) -> None:
        missing = calibration_report(
            vision=calibrated_vision(),
            arm=calibrated_arm(),
            grasp=GraspConfig(marker_to_grasp_mm=0.0),
        )
        self.assertEqual(len(missing), 1, missing)
        self.assertIn("marker_to_grasp_mm", missing[0])

    def test_the_menu_entry_is_a_valid_declaration(self) -> None:
        number, label = MENU_ENTRY
        self.assertEqual(number, 19)
        self.assertTrue(label.strip())


class ObserveOnlyTests(unittest.TestCase):
    def test_the_loop_prints_one_line_per_frame_and_stops_on_q(self) -> None:
        perception = Mock()
        perception.observe.side_effect = [
            CellObservation(marker=None, material=None),
            CellObservation(marker=None, material=None),
        ]
        lines: list = []
        answers = iter(["", "q"])

        run_observe_only(
            perception=perception,
            frames=lambda: object(),
            prompt=lambda _: next(answers),
            print_line=lines.append,
        )

        self.assertEqual(len(lines), 2)
        self.assertEqual(perception.observe.call_count, 2)


class GraspOnceTests(unittest.TestCase):
    def make_pipeline(self, observation: CellObservation) -> Mock:
        pipeline = Mock()
        pipeline.observe.return_value = observation
        return pipeline

    def test_a_frame_without_material_stops_before_asking_for_confirmation(self) -> None:
        pipeline = self.make_pipeline(CellObservation(marker=None, material=None))
        prompted: list = []

        gripped = run_grasp_once(
            pipeline=pipeline,
            gripper=Mock(),
            prompt=lambda message: prompted.append(message) or "YES",
            print_line=lambda _: None,
        )

        self.assertFalse(gripped)
        pipeline.pick_up.assert_not_called()
        # 没有物料就根本不该弹确认框：先看清楚，再问要不要动。
        self.assertEqual(prompted, [])

    def test_a_frame_without_marker_stops_and_says_why(self) -> None:
        pipeline = self.make_pipeline(
            CellObservation(marker=None, material=CylindricalPose(0.0, 60.0, 20.0))
        )
        lines: list = []

        gripped = run_grasp_once(
            pipeline=pipeline,
            gripper=Mock(),
            prompt=lambda _: "YES",
            print_line=lines.append,
        )

        self.assertFalse(gripped)
        pipeline.pick_up.assert_not_called()
        self.assertTrue(any("靶标" in line for line in lines), lines)

    def test_anything_other_than_uppercase_yes_cancels(self) -> None:
        pipeline = self.make_pipeline(
            CellObservation(
                marker=CylindricalPose(0.0, 40.0, 60.0),
                material=CylindricalPose(10.0, 70.0, 20.0),
            )
        )

        for answer in ("yes", "Y", "", "y"):
            with self.subTest(answer=answer):
                pipeline.reset_mock()
                gripped = run_grasp_once(
                    pipeline=pipeline,
                    gripper=Mock(),
                    prompt=lambda _, value=answer: value,
                    print_line=lambda _: None,
                )
                self.assertFalse(gripped)
                pipeline.pick_up.assert_not_called()

    def test_a_confirmed_cycle_reports_what_the_gripper_felt(self) -> None:
        pipeline = self.make_pipeline(
            CellObservation(
                marker=CylindricalPose(0.0, 40.0, 60.0),
                material=CylindricalPose(10.0, 70.0, 20.0),
            )
        )
        blocked = GripperReport(
            command=CLOSE_COMMAND, target=500, position=430, reached=False
        )
        pipeline.pick_up.return_value = Mock(
            held_something=True, summary=lambda: "fine", gripper=blocked
        )

        gripped = run_grasp_once(
            pipeline=pipeline,
            gripper=Mock(),
            prompt=lambda _: "YES",
            print_line=lambda _: None,
        )

        self.assertTrue(gripped)
        pipeline.pick_up.assert_called_once_with()

    def test_an_empty_grasp_is_warned_about_explicitly(self) -> None:
        pipeline = self.make_pipeline(
            CellObservation(
                marker=CylindricalPose(0.0, 40.0, 60.0),
                material=CylindricalPose(10.0, 70.0, 20.0),
            )
        )
        pipeline.pick_up.return_value = Mock(
            held_something=False, summary=lambda: "empty", gripper=Mock()
        )
        lines: list = []

        gripped = run_grasp_once(
            pipeline=pipeline,
            gripper=Mock(),
            prompt=lambda _: "YES",
            print_line=lines.append,
        )

        self.assertFalse(gripped)
        self.assertTrue(any("空夹" in line for line in lines), lines)


if __name__ == "__main__":
    unittest.main()
