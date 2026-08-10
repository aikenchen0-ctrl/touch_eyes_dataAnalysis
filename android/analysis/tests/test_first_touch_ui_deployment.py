import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


class FirstTouchUiDeploymentTests(unittest.TestCase):
    def test_panel_shows_first_touch_default_separately_from_final_models(self):
        activity = (
            REPOSITORY_ROOT
            / "android"
            / "app"
            / "src"
            / "main"
            / "java"
            / "com"
            / "zhifaios"
            / "eyes"
            / "android"
            / "MainActivity.java"
        ).read_text(encoding="utf-8")
        strings = (
            REPOSITORY_ROOT
            / "android"
            / "app"
            / "src"
            / "main"
            / "res"
            / "values"
            / "strings.xml"
        ).read_text(encoding="utf-8")

        self.assertIn("controller.firstTouchModelId()", activity)
        self.assertIn("first_touch_model_default_format", activity)
        self.assertIn('name="first_touch_model_title">按下期模型状态<', strings)
        self.assertIn('name="impact_model_title">旧模型对照（不用于新标准）<', strings)


if __name__ == "__main__":
    unittest.main()
