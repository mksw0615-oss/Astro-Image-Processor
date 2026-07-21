from PIL import Image

import planet


def test_crop_around_object_returns_tighter_region():
    image = Image.new("RGB", (120, 120), (0, 0, 0))

    for x in range(30, 80):
        for y in range(30, 80):
            image.putpixel((x, y), (250, 250, 250))

    cropped = planet.crop_around_object(image, padding=8)

    assert cropped.size[0] < image.size[0]
    assert cropped.size[1] < image.size[1]


def test_enhance_planet_returns_original_size_after_zoom_crop():
    image = Image.new("RGB", (100, 100), (20, 20, 20))

    for x in range(30, 70):
        for y in range(30, 70):
            image.putpixel((x, y), (240, 120, 80))

    processed = planet.enhance_planet(image, 12, 1.0, 0.4, 0.6)

    assert processed.size == image.size
