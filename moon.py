from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageStat

import quality


DEFAULT_SETTINGS = {
    "brightness": 1.05,
    "contrast": 1.35,
    "sharpness": 1.8,
    "detail": 1.3,
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

    analysis = quality.analyze_image_quality(path, detected_type="moon")
    print()
    quality.print_quality_report(analysis)
    print()

    suggestions = suggest_moon_settings(image)

    print("What each value does:")
    print("Brightness -> makes the whole image lighter or darker")
    print("Contrast -> increases or reduces the difference between bright and dark areas")
    print("Sharpness -> makes edges crisper or softer")
    print("Detail -> adds or reduces extra fine texture")
    print()

    brightness = ask_for_number("Brightness", suggestions["brightness"])
    contrast = ask_for_number("Contrast", suggestions["contrast"])
    sharpness = ask_for_number("Sharpness", suggestions["sharpness"])
    detail = ask_for_number("Detail boost", suggestions["detail"])

    processed = enhance_moon(image, brightness, contrast, sharpness, detail)

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


def suggest_moon_settings(image):
    gray = image.convert("L")
    stat = ImageStat.Stat(gray)
    average_brightness = stat.mean[0]
    contrast_spread = stat.stddev[0]
    edge_strength = get_edge_strength(gray)
    brightest_pixel = get_brightest_pixel(gray)

    brightness = DEFAULT_SETTINGS["brightness"]
    contrast = DEFAULT_SETTINGS["contrast"]
    sharpness = DEFAULT_SETTINGS["sharpness"]
    detail = DEFAULT_SETTINGS["detail"]

    if brightest_pixel >= 240:
        brightness = 0.8
    elif brightest_pixel >= 220:
        brightness = 0.9
    elif brightest_pixel >= 180:
        brightness = 0.98
    elif average_brightness < 75:
        brightness = 1.25
    elif average_brightness < 105:
        brightness = 1.15
    elif average_brightness > 175:
        brightness = 0.9
    elif average_brightness > 145:
        brightness = 0.98

    if contrast_spread < 35:
        contrast = 1.55
    elif contrast_spread < 55:
        contrast = 1.4
    elif contrast_spread > 85:
        contrast = 1.15
    elif brightest_pixel >= 220:
        contrast = 1.2

    if edge_strength < 7:
        sharpness = 2.1
        detail = 1.45
    elif edge_strength < 12:
        sharpness = 1.85
        detail = 1.35
    elif edge_strength > 22:
        sharpness = 1.35
        detail = 1.15

    if brightest_pixel >= 220:
        sharpness = 0.9
        detail = 0.9

    return {
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "sharpness": round(sharpness, 2),
        "detail": round(detail, 2),
    }


def get_edge_strength(gray):
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_stat = ImageStat.Stat(edges)

    return edge_stat.mean[0]


def get_percentile(gray, percentile):
    histogram = gray.histogram()
    target_count = gray.width * gray.height * (percentile / 100)
    running_count = 0

    for pixel_value, count in enumerate(histogram):
        running_count += count

        if running_count >= target_count:
            return pixel_value

    return 255


def get_brightest_pixel(gray):
    return max(gray.getextrema())


def enhance_moon(image, brightness, contrast, sharpness, detail):
    image = image.convert("L")

    image = ImageEnhance.Brightness(image).enhance(brightness)
    image = ImageOps.autocontrast(image)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    image = ImageEnhance.Sharpness(image).enhance(sharpness)

    detail_layer = image.filter(ImageFilter.DETAIL)
    image = Image.blend(image, detail_layer, min(max(detail - 1.0, 0), 1))

    return image.convert("RGB")


def make_output_path(path):
    output_folder = path.parent / "outputs"
    output_folder.mkdir(exist_ok=True)

    return output_folder / f"{path.stem}_moon_enhanced{path.suffix}"


def show_processed_image(processed):
    processed.show(title="Enhanced Moon Image")
