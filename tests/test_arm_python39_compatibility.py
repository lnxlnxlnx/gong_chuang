"""Ensure the robot-arm test entry point imports under Python 3.9."""

import subprocess
import sys
import unittest


class ArmPython39CompatibilityTests(unittest.TestCase):
    def test_position_motion_module_imports(self):
        result = subprocess.run(
            [sys.executable, "-c", "import tests.manual.arm"],
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
