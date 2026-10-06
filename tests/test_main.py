"""The application entry point starts the material-coordinate preview."""

import unittest
from unittest.mock import patch

import main


class MainDemoTests(unittest.TestCase):
    def test_main_opens_material_preview(self) -> None:
        with patch.object(main, "demo_material") as material, patch.object(main, "demo_vision") as vision:
            main.main()
        material.assert_called_once_with()
        vision.assert_not_called()


if __name__ == "__main__":
    unittest.main()
