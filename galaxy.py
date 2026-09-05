from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageStat

import quality


DEFAULT_SETTINGS = {
    "shadow_lift": 1.25,
    "local_contrast": 1.5,
    "color": 1.22,
    "sharpness": 1.10,
    "background": 0.90,
    "noise_reduction": 0.85,
}


def process(image_path):
    image_path = clean_path(image_path)
    path = Path(image_path)

    if not path.exists():
        print("Could not find that image. Check the path and try again.")
        return

    try:
        image = Image.open(path)
    except Exception:
        print("Could not open that file as an image.")
        return

    analysis = quality.analyze_image_quality(path, detected_type="galaxy")
    print()
    quality.print_quality_report(analysis)
    print()

    suggestions = suggest_galaxy_settings(image)

    print("What each value does:")
    print("Shadow / midtone lift -> reveals faint arms without brightening the core")
    print("Local contrast -> brings out dust lanes and faint detail")
    print("Color -> boosts or reduces color saturation")
    print("Sharpness -> makes edges crisper or softer")
    print("Background level -> sets the dark sky floor after background subtraction")
    print("Noise reduction -> smooths faint grain while protecting stars and the core")
    print()

    shadow_lift = ask_for_number("Shadow / midtone lift", suggestions["shadow_lift"])
    local_contrast = ask_for_number("Local contrast", suggestions["local_contrast"])
    color = ask_for_number("Color balance / saturation", suggestions["color"])
    sharpness = ask_for_number("Sharpness", suggestions["sharpness"])
    background = ask_for_number("Background level", suggestions["background"])
    noise_reduction = ask_for_number("Noise reduction", suggestions["noise_reduction"])

    processed = enhance_galaxy(
        image, shadow_lift, local_contrast, color, sharpness, background, noise_reduction
    )

    output_path = make_output_path(path)
    processed.save(output_path)

    print()
    print(f"Enhanced image saved: {output_path}")

    show_processed_image(processed)


def clean_path(image_path):
    return image_path.strip().strip('"').strip("'")


def ask_for_number(setting_name, default_value):
    user_input = input(f"{setting_name} amount [{default_value}]: ").strip()

    if user_input == "":
        return default_value

    try:
        return float(user_input)
    except ValueError:
        print(f"Invalid input. Using {default_value}.")
        return default_value


def suggest_galaxy_settings(image):
    rgb = image.convert("RGB")
    gray = image.convert("L")
    stat = ImageStat.Stat(gray)
    average_brightness = stat.mean[0]
    contrast_spread = stat.stddev[0]
    edge_strength = get_edge_strength(gray)
    color_strength = get_color_strength(rgb)

    shadow_lift = DEFAULT_SETTINGS["shadow_lift"]
    local_contrast = DEFAULT_SETTINGS["local_contrast"]
    color = DEFAULT_SETTINGS["color"]
    sharpness = DEFAULT_SETTINGS["sharpness"]
    background = DEFAULT_SETTINGS["background"]
    noise_reduction = DEFAULT_SETTINGS["noise_reduction"]

    if average_brightness < 70:
        shadow_lift = 1.4
        background = 0.85
    elif average_brightness < 100:
        shadow_lift = 1.3
    elif average_brightness > 200:
        shadow_lift = 1.05
        background = 0.95
    elif average_brightness > 170:
        shadow_lift = 1.15

    if contrast_spread < 30:
        local_contrast = 1.8
    elif contrast_spread < 50:
        local_contrast = 1.6
    elif contrast_spread > 80:
        local_contrast = 1.5

    if color_strength < 25:
        color = 1.25
    elif color_strength < 40:
        color = 1.22
    elif color_strength > 70:
        color = 1.15

    if edge_strength < 8:
        sharpness = 1.15
    elif edge_strength < 13:
        sharpness = 1.12
    elif edge_strength > 20:
        sharpness = 1.05
        noise_reduction = 0.90

    return {
        "shadow_lift": round(shadow_lift, 2),
        "local_contrast": round(local_contrast, 2),
        "color": round(color, 2),
        "sharpness": round(sharpness, 2),
        "background": round(background, 2),
        "noise_reduction": round(noise_reduction, 2),
    }


def get_edge_strength(gray):
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_stat = ImageStat.Stat(edges)

    return edge_stat.mean[0]


def get_color_strength(rgb):
    r, g, b = rgb.split()
    r_stat = ImageStat.Stat(r)
    g_stat = ImageStat.Stat(g)
    b_stat = ImageStat.Stat(b)

    return (r_stat.stddev[0] + g_stat.stddev[0] + b_stat.stddev[0]) / 3


def enhance_galaxy(image, shadow_lift, local_contrast, color, sharpness, background, noise_reduction=0.55):
    """Reveal galaxy structure without turning the sky background into grit.

    This is deliberately a conservative, data-preserving stretch.  It can make
    real colour and dust lanes easier to see, but it cannot manufacture the
    fine structure present only in a longer, stacked exposure.
    """
    original = np.asarray(image.convert("RGB"), dtype=np.float32)
    sky_floor = min(max(background, 0.0), 1.0) * 4.0
    signal = subtract_sky_background(original, sky_floor)
    luminance = (
        signal[:, :, 0] * 0.2126
        + signal[:, :, 1] * 0.7152
        + signal[:, :, 2] * 0.0722
    )

    # The raw frame is dominated by a pink sky.  After subtracting that sky,
    # an asinh stretch lifts the faint outer arms while compressing the bright
    # nucleus and stars instead of clipping them.
    white_point = max(12.0, float(np.percentile(luminance, 99.9)))
    shadow_lift = min(max(shadow_lift, 0.8), 1.5)
    stretch_strength = 3.5 + (shadow_lift - 1.0) * 6.0
    stretched = np.arcsinh(signal / white_point * stretch_strength)
    stretched *= 255.0 / np.arcsinh(stretch_strength)
    stretched_rgb = np.clip(stretched, 0, 255).astype(np.uint8)

    # The stretch makes faint read noise visible as dusty/grainy texture.
    # Denoise before CLAHE, then blend only into darker pixels: stars, the core,
    # and strong dust-lane edges remain from the unblurred image.
    noise_reduction = min(max(noise_reduction, 0.0), 1.0)
    denoised_rgb = cv2.fastNlMeansDenoisingColored(
        stretched_rgb, None, h=7, hColor=8, templateWindowSize=7, searchWindowSize=21
    )
    stretched_luminance = cv2.cvtColor(stretched_rgb, cv2.COLOR_RGB2GRAY)
    denoise_mask = np.clip((145.0 - stretched_luminance) / 105.0, 0.0, 1.0)
    denoise_mask *= noise_reduction
    stretched_rgb = np.clip(
        stretched_rgb * (1.0 - denoise_mask[:, :, None])
        + denoised_rgb * denoise_mask[:, :, None],
        0,
        255,
    ).astype(np.uint8)

    # Use CLAHE only as a very restrained luminance-only adjustment.  Applying
    # it strongly across the whole frame was amplifying sensor noise into the
    # mottled, sandy background seen in the previous output.
    lab = cv2.cvtColor(stretched_rgb, cv2.COLOR_RGB2LAB)
    local_contrast = min(max(local_contrast, 0.5), 1.5)
    clahe = cv2.createCLAHE(clipLimit=local_contrast, tileGridSize=(16, 16))
    clahe_luminance = clahe.apply(lab[:, :, 0])
    detail_mask = np.clip((185.0 - lab[:, :, 0]) / 145.0, 0.0, 1.0) * 0.18
    lab[:, :, 0] = np.clip(
        lab[:, :, 0] * (1.0 - detail_mask) + clahe_luminance * detail_mask,
        0,
        255,
    ).astype(np.uint8)

    enhanced_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32)
    enhanced_rgb = apply_galaxy_colour_balance(enhanced_rgb)
    enhanced = Image.fromarray(enhanced_rgb.astype(np.uint8))
    enhanced = ImageEnhance.Color(enhanced).enhance(min(max(color, 0.8), 1.45))
    enhanced = ImageEnhance.Sharpness(enhanced).enhance(min(max(sharpness, 0.8), 1.2))
    return enhanced


def subtract_sky_background(image_array, target_sky_level=3.0):
    """Remove sky cast and broad gradients using only likely-sky pixels.

    A single global percentile makes one side of a vignetted frame black and
    leaves a colour cast on the other.  A robust per-channel plane gives a
    smooth dark sky while leaving the extended galaxy signal intact.
    """
    height, width = image_array.shape[:2]
    luminance = cv2.cvtColor(image_array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    sky_limit = np.percentile(luminance, 45)
    mask = luminance <= sky_limit
    y, x = np.nonzero(mask)

    if len(x) < 100:
        sky_rgb = np.percentile(image_array.reshape(-1, 3), 20, axis=0)
        return np.clip(image_array - sky_rgb + target_sky_level, 0, 255)

    # Fit at most 20k points: enough for a stable background plane without
    # slowing down large camera frames.
    step = max(1, len(x) // 20000)
    sample_x = x[::step]
    sample_y = y[::step]
    x_normalized = sample_x.astype(np.float32) / max(width - 1, 1)
    y_normalized = sample_y.astype(np.float32) / max(height - 1, 1)
    design = np.column_stack((np.ones_like(x_normalized), x_normalized, y_normalized))
    background = np.empty_like(image_array)
    grid_y, grid_x = np.mgrid[0:height, 0:width]
    grid_design = np.column_stack((
        np.ones(height * width, dtype=np.float32),
        (grid_x.reshape(-1) / max(width - 1, 1)).astype(np.float32),
        (grid_y.reshape(-1) / max(height - 1, 1)).astype(np.float32),
    ))
    for channel in range(3):
        values = image_array[sample_y, sample_x, channel]
        coefficients, _, _, _ = np.linalg.lstsq(design, values, rcond=None)
        background[:, :, channel] = (grid_design @ coefficients).reshape(height, width)

    return np.clip(image_array - background + target_sky_level, 0, 255)


def apply_galaxy_colour_balance(rgb):
    """Neutralize residual sky cast with a small, cool deep-sky bias."""
    luminance = cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    sky = rgb[luminance < np.percentile(luminance, 45)]
    if len(sky) == 0:
        return np.clip(rgb, 0, 255)

    sky_median = np.maximum(np.median(sky, axis=0), 1.0)
    neutral = np.exp(np.mean(np.log(sky_median)))
    scale = np.clip(neutral / sky_median, 0.85, 1.18)
    # Keep calibration modest: real galaxy colour remains data-driven.
    scale *= np.array((0.98, 1.0, 1.05), dtype=np.float32)
    return np.clip(rgb * scale, 0, 255)


def darken_background(image, background):
    background = min(max(background, 0.0), 1.0)

    pixels = image.load()
    width, height = image.size

    for y in range(height):
        for x in range(width):
            red, green, blue = pixels[x, y]
            brightness = (red + green + blue) / 3

            if brightness < 70:
                pixels[x, y] = (
                    int(red * background),
                    int(green * background),
                    int(blue * background),
                )

    return image


def make_output_path(path):
    output_folder = path.parent / "outputs"
    output_folder.mkdir(exist_ok=True)

    return output_folder / f"{path.stem}_galaxy_enhanced{path.suffix}"


def show_processed_image(processed):
    processed.show(title="Enhanced Galaxy Image")
