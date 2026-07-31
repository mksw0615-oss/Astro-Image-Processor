from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageStat

import quality


DEFAULT_SETTINGS = {
    "brightness": 1.12,
    "contrast": 1.10,
    "color": 1.18,
    "sharpness": 1.10,
    "background": 0.96,
    "glow": 0.12,
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

    analysis = quality.analyze_image_quality(path, detected_type="nebula")
    print()
    quality.print_quality_report(analysis)
    print()

    suggestions = suggest_nebula_settings(image)

    print("What each value does:")
    print("Brightness -> controls the nonlinear nebula stretch")
    print("Contrast -> adjusts faint nebula detail after the stretch")
    print("Color -> boosts or reduces color saturation")
    print("Sharpness -> makes edges crisper or softer")
    print("Background darkening -> darkens the darker parts more or less")
    print("Glow/detail boost -> adds or reduces soft glow and detail")
    print()

    brightness = ask_for_number("Brightness", suggestions["brightness"])
    contrast = ask_for_number("Contrast", suggestions["contrast"])
    color = ask_for_number("Color balance / saturation", suggestions["color"])
    sharpness = ask_for_number("Sharpness", suggestions["sharpness"])
    background = ask_for_number("Background darkening", suggestions["background"])
    glow = ask_for_number("Glow/detail boost", suggestions["glow"])

    processed = enhance_nebula(image, brightness, contrast, color, sharpness, background, glow)

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


def suggest_nebula_settings(image):
    rgb = image.convert("RGB")
    gray = image.convert("L")
    stat = ImageStat.Stat(gray)
    average_brightness = stat.mean[0]
    contrast_spread = stat.stddev[0]
    edge_strength = get_edge_strength(gray)
    color_strength = get_color_strength(rgb)

    brightness = DEFAULT_SETTINGS["brightness"]
    contrast = DEFAULT_SETTINGS["contrast"]
    color = DEFAULT_SETTINGS["color"]
    sharpness = DEFAULT_SETTINGS["sharpness"]
    background = DEFAULT_SETTINGS["background"]
    glow = DEFAULT_SETTINGS["glow"]

    if average_brightness < 70:
        brightness = 1.18
        background = 0.93
    elif average_brightness < 100:
        brightness = 1.15
    elif average_brightness > 190:
        brightness = 1.08
        background = 0.98
    elif average_brightness > 160:
        brightness = 1.10

    if contrast_spread < 30:
        contrast = 1.25
    elif contrast_spread < 50:
        contrast = 1.20
    elif contrast_spread > 80:
        contrast = 1.08

    if color_strength < 20:
        color = 1.28
    elif color_strength < 35:
        color = 1.22
    elif color_strength > 60:
        color = 1.08

    if edge_strength < 8:
        sharpness = 1.20
        glow = 0.18
    elif edge_strength < 13:
        sharpness = 1.15
        glow = 0.14
    elif edge_strength > 20:
        sharpness = 1.0
        glow = 0.08

    return {
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "color": round(color, 2),
        "sharpness": round(sharpness, 2),
        "background": round(background, 2),
        "glow": round(glow, 2),
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


def enhance_nebula(image, brightness, contrast, color, sharpness, background, glow):
    """Extract faint nebula signal with sky subtraction and an asinh stretch."""
    original = np.asarray(image.convert("RGB"), dtype=np.float32)
    signal = subtract_sky_gradient(original)
    luminance = (
        signal[:, :, 0] * 0.2126
        + signal[:, :, 1] * 0.7152
        + signal[:, :, 2] * 0.0722
    )

    # Asinh is the standard astronomy-style stretch: faint material grows
    # rapidly while stars and the nebula core are compressed instead of clipped.
    white_point = max(12.0, float(np.percentile(luminance, 99.95)))
    brightness = min(max(brightness, 0.8), 1.35)
    stretch_strength = 3.5 + (brightness - 1.0) * 6.0
    processed = np.arcsinh(signal / white_point * stretch_strength)
    processed *= 255.0 / np.arcsinh(stretch_strength)

    # Modest contrast after stretching makes dust lanes clearer without the
    # per-channel autocontrast that previously turned the background green.
    contrast = min(max(contrast, 0.8), 1.25)
    processed = (processed - 16.0) * contrast + 16.0
    processed_image = Image.fromarray(np.clip(processed, 0, 255).astype(np.uint8))

    processed_image = ImageEnhance.Color(processed_image).enhance(min(max(color, 0.8), 1.25))
    # Background subtraction already establishes the black point.  A second
    # threshold-based darkening pass creates visible patches, so do not apply it.
    processed_image = add_soft_glow(processed_image, glow)
    processed_image = ImageEnhance.Sharpness(processed_image).enhance(min(max(sharpness, 0.8), 1.25))

    return processed_image


def subtract_sky_gradient(image_array, target_sky_level=3.0):
    """Subtract a robust RGB sky estimate without altering image gradients.

    A per-channel plane can overfit uneven sensor/sky colour and create false
    green or magenta bands after stretching.  The lower percentile is stable
    in star fields and removes the dominant sky cast while retaining smooth,
    natural gradients in the original data.
    """
    background = np.percentile(image_array.reshape(-1, 3), 20, axis=0)
    return np.clip(image_array - background + target_sky_level, 0, 255)


def darken_background(image, background):
    background = min(max(background, 0.0), 1.0)

    pixels = image.load()
    width, height = image.size

    for y in range(height):
        for x in range(width):
            red, green, blue = pixels[x, y]
            brightness = (red + green + blue) / 3

            if brightness < 75:
                pixels[x, y] = (
                    int(red * background),
                    int(green * background),
                    int(blue * background),
                )

    return image


def add_soft_glow(image, glow):
    glow = min(max(glow, 0.0), 1.0)
    blurred = image.filter(ImageFilter.GaussianBlur(radius=2))

    return Image.blend(image, blurred, glow)


def make_output_path(path):
    output_folder = path.parent / "outputs"
    output_folder.mkdir(exist_ok=True)

    return output_folder / f"{path.stem}_nebula_enhanced{path.suffix}"


def show_processed_image(processed):
    processed.show(title="Enhanced Nebula Image")
