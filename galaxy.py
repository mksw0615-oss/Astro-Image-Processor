from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageStat


DEFAULT_SETTINGS = {
    "brightness": 1.25,
    "contrast": 1.45,
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

    print()
    suggestions = suggest_galaxy_settings(image)

    print("What each value does:")
    print("Brightness -> makes the whole image lighter or darker")
    print("Contrast -> increases or reduces the difference between bright and dark areas")
    print("Color -> boosts or reduces color saturation")
    print("Sharpness -> makes edges crisper or softer")
    print("Background darkening -> darkens the darker parts more or less")
    print()

    brightness = ask_for_number("Brightness", suggestions["brightness"])
    contrast = ask_for_number("Contrast", suggestions["contrast"])
    color = ask_for_number("Color balance / saturation", suggestions["color"])
    sharpness = ask_for_number("Sharpness", suggestions["sharpness"])
    background = ask_for_number("Background darkening", suggestions["background"])

    processed = enhance_galaxy(image, brightness, contrast, color, sharpness, background)

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

    brightness = DEFAULT_SETTINGS["brightness"]
    contrast = DEFAULT_SETTINGS["contrast"]
    color = DEFAULT_SETTINGS["color"]
    sharpness = DEFAULT_SETTINGS["sharpness"]
    background = DEFAULT_SETTINGS["background"]

    if average_brightness < 70:
        brightness = 1.4
        background = 0.85
    elif average_brightness < 100:
        brightness = 1.3
    elif average_brightness > 200:
        brightness = 1.05
        background = 0.95
    elif average_brightness > 170:
        brightness = 1.15

    if contrast_spread < 30:
        contrast = 1.65
    elif contrast_spread < 50:
        contrast = 1.55
    elif contrast_spread > 80:
        contrast = 1.25

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
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
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


def enhance_galaxy(image, brightness, contrast, color, sharpness, background):
    image = image.convert("RGB")

    image = ImageEnhance.Brightness(image).enhance(brightness)
    image = ImageOps.autocontrast(image, cutoff=1)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    image = ImageEnhance.Color(image).enhance(color)
    image = darken_background(image, background)
    image = ImageEnhance.Sharpness(image).enhance(sharpness)

    return image.filter(ImageFilter.SMOOTH_MORE)


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
