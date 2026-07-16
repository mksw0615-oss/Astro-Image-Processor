from pathlib import Path
from PIL import Image
import importlib.util


def load_main_module():
    spec = importlib.util.spec_from_file_location('main_mod', Path(__file__).resolve().parents[1] / 'main.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def create_saturn_like_image(path: Path):
    image = Image.new('RGB', (400, 400), (0, 0, 0))
    image = image.crop((0, 0, 400, 400))
    for y in range(150, 250):
        for x in range(150, 250):
            image.putpixel((x, y), (255, 255, 255))
    image.save(path)
    return path


def test_saturn_like_image_is_detected_as_planet(tmp_path):
    module = load_main_module()
    image_path = create_saturn_like_image(tmp_path / 'saturn_like.png')
    assert module.detect_image_type(str(image_path)) == 'planet'
