"""The application entry point starts the material-coordinate preview."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import main


class MainDemoTests(unittest.TestCase):
    def test_main_opens_material_preview(self) -> None:
        with patch.object(main, "demo_material") as material:
            main.main()
        material.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
