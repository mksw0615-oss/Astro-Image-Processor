import importlib.util
import unittest
from pathlib import Path

from PIL import Image, ImageDraw


spec = importlib.util.spec_from_file_location("moon", Path(__file__).resolve().parents[1] / "moon.py")
moon = importlib.util.module_from_spec(spec)
spec.loader.exec_module(moon)


class MoonSuggestionTests(unittest.TestCase):
    def test_overbright_image_suggests_brightness_below_one(self):
        image = Image.new("L", (100, 100), 0)
        draw = ImageDraw.Draw(image)
        draw.ellipse((20, 20, 80, 80), fill=250)
        suggestions = moon.suggest_moon_settings(image)
        self.assertLess(suggestions["brightness"], 1.0)


if __name__ == "__main__":
    unittest.main()
