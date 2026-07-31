import numpy as np
from PIL import Image

import main
import calibrate
import galaxy
import nebula
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


def test_extended_halo_score_separates_diffuse_core_from_isolated_point():
    galaxy_like = Image.new("L", (100, 100), 20)
    planet_like = Image.new("L", (100, 100), 20)

    for x in range(100):
        for y in range(100):
            distance_squared = (x - 50) ** 2 + (y - 50) ** 2
            if distance_squared <= 12 ** 2:
                galaxy_like.putpixel((x, y), max(20, 240 - int(distance_squared ** 0.5) * 12))
            if distance_squared <= 3 ** 2:
                planet_like.putpixel((x, y), 240)

    assert main.get_extended_halo_score(galaxy_like) >= 0.18
    assert main.get_extended_halo_score(planet_like) < 0.18


def test_galaxy_structure_score_prefers_elliptical_smooth_halo():
    galaxy_like = Image.new("L", (120, 120), 20)
    nebula_like = Image.new("L", (120, 120), 20)

    for x in range(120):
        for y in range(120):
            ellipse_distance = ((x - 60) / 24) ** 2 + ((y - 60) / 10) ** 2
            if ellipse_distance <= 1:
                galaxy_like.putpixel((x, y), int(60 + 170 * (1 - ellipse_distance)))

            left_lobe = (x - 43) ** 2 + (y - 65) ** 2
            right_lobe = (x - 77) ** 2 + (y - 48) ** 2
            if left_lobe <= 13 ** 2 or right_lobe <= 10 ** 2:
                nebula_like.putpixel((x, y), 150)

    assert main.get_galaxy_structure_score(galaxy_like) >= 0.42
    assert main.get_galaxy_structure_score(nebula_like) < 0.42


def test_galaxy_enhancement_creates_dark_sky_without_clipping_core():
    image = Image.new("RGB", (40, 40), (80, 80, 80))
    for x in range(16, 24):
        for y in range(16, 24):
            image.putpixel((x, y), (250, 250, 250))

    processed = galaxy.enhance_galaxy(image, 1.3, 2.0, 1.0, 1.0, 1.0)

    assert processed.getpixel((20, 20))[0] > 180
    assert processed.getpixel((4, 4))[0] < 60


def test_galaxy_noise_reduction_keeps_bright_core_intact():
    image = Image.new("RGB", (40, 40), (80, 80, 80))
    for x in range(16, 24):
        for y in range(16, 24):
            image.putpixel((x, y), (250, 250, 250))

    processed = galaxy.enhance_galaxy(image, 1.2, 1.2, 1.0, 1.0, 1.0, 1.0)
    assert processed.getpixel((20, 20))[0] > 180


def test_nebula_enhancement_preserves_bright_core_and_lifts_faint_sky():
    image = Image.new("RGB", (40, 40), (60, 75, 85))
    for x in range(16, 24):
        for y in range(16, 24):
            image.putpixel((x, y), (255, 250, 245))

    processed = nebula.enhance_nebula(image, 1.12, 1.1, 1.1, 1.0, 1.0, 0.0)

    assert processed.getpixel((20, 20))[0] > 220
    assert processed.getpixel((4, 4))[0] < 60


def test_background_neutralization_removes_pink_sky_cast():
    image = Image.new("RGB", (30, 30), (120, 80, 100))
    neutralized = calibrate.neutralize_background(np.array(image, dtype=np.float32))

    background = calibrate.estimate_sky_background(neutralized)
    assert max(background) - min(background) < 1


def test_color_calibration_balances_a_neutral_reference():
    image = np.full((30, 30, 3), (90, 80, 70), dtype=np.float32)
    image[10:20, 10:20] = (180, 160, 140)

    calibrated = calibrate.color_calibrate(image)
    reference = calibrated[10:20, 10:20].mean(axis=(0, 1))
    assert max(reference) - min(reference) < 2
