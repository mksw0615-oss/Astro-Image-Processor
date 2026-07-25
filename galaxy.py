from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageStat

import quality


DEFAULT_SETTINGS = {
    "shadow_lift": 1.25,
    "local_contrast": 2.0,
    "color": 1.35,
    "sharpness": 1.25,
    "background": 0.90,
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
    print("Background darkening -> darkens the darker parts more or less")
    print()

    shadow_lift = ask_for_number("Shadow / midtone lift", suggestions["shadow_lift"])
    local_contrast = ask_for_number("Local contrast", suggestions["local_contrast"])
    color = ask_for_number("Color balance / saturation", suggestions["color"])
    sharpness = ask_for_number("Sharpness", suggestions["sharpness"])
    background = ask_for_number("Background darkening", suggestions["background"])

    processed = enhance_galaxy(image, shadow_lift, local_contrast, color, sharpness, background)

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
        local_contrast = 2.6
    elif contrast_spread < 50:
        local_contrast = 2.3
    elif contrast_spread > 80:
        local_contrast = 1.5

    if color_strength < 25:
        color = 1.6
    elif color_strength < 40:
        color = 1.45
    elif color_strength > 70:
        color = 1.15

    if edge_strength < 8:
        sharpness = 1.45
    elif edge_strength < 13:
        sharpness = 1.35
    elif edge_strength > 20:
        sharpness = 1.05

    return {
        "shadow_lift": round(shadow_lift, 2),
        "local_contrast": round(local_contrast, 2),
        "color": round(color, 2),
        "sharpness": round(sharpness, 2),
        "background": round(background, 2),
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


def enhance_galaxy(image, shadow_lift, local_contrast, color, sharpness, background):
    """Enhance faint galaxy detail while preserving the bright nucleus."""
    original = np.array(image.convert("RGB"), dtype=np.uint8)
    lab = cv2.cvtColor(original, cv2.COLOR_RGB2LAB)
    luminance = lab[:, :, 0]

    # A gamma curve lifts the shadows more gently than a linear brightness
    # multiplier.  Blend it only into the 20--180 range, then fade it out.
    shadow_lift = min(max(shadow_lift, 1.0), 2.0)
    gamma_luminance = np.power(luminance / 255.0, 1.0 / shadow_lift) * 255.0
    midtone_mask = np.clip((180.0 - luminance) / 160.0, 0.0, 1.0)
    midtone_mask *= (luminance >= 20) & (luminance <= 180)
    toned_luminance = luminance * (1.0 - midtone_mask) + gamma_luminance * midtone_mask

    # CLAHE gives local contrast to dust lanes and spiral arms.  Its result is
    # also restricted to shadows and midtones, so it cannot overexpose a core.
    local_contrast = min(max(local_contrast, 0.5), 4.0)
    clahe = cv2.createCLAHE(clipLimit=local_contrast, tileGridSize=(8, 8))
    clahe_luminance = clahe.apply(luminance)
    enhanced_luminance = toned_luminance * (1.0 - midtone_mask) + clahe_luminance * midtone_mask
    lab[:, :, 0] = np.clip(enhanced_luminance, 0, 255).astype(np.uint8)

    enhanced = Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))
    enhanced = ImageEnhance.Color(enhanced).enhance(color)
    enhanced = darken_background(enhanced, background)
    enhanced = ImageEnhance.Sharpness(enhanced).enhance(sharpness)

    # Pixels at and above 240 are the galaxy nucleus/highlights.  Preserve
    # them completely, with a soft transition from 220 to avoid a hard edge.
    highlight_mask = np.clip((luminance.astype(np.float32) - 220.0) / 20.0, 0.0, 1.0)
    processed = np.array(enhanced, dtype=np.float32)
    result = processed * (1.0 - highlight_mask[:, :, None]) + original * highlight_mask[:, :, None]
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


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
