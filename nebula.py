from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageStat


DEFAULT_SETTINGS = {
    "brightness": 1.30,
    "contrast": 1.40,
    "color": 1.60,
    "sharpness": 1.20,
    "background": 0.88,
    "glow": 0.25,
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
    suggestions = suggest_nebula_settings(image)

    print("Suggested starting values based on this image:")
    print(f"Brightness: {suggestions['brightness']}  -> makes the whole image lighter or darker")
    print(f"Contrast:   {suggestions['contrast']}  -> increases or reduces the difference between bright and dark areas")
    print(f"Color:      {suggestions['color']}  -> boosts or reduces color saturation")
    print(f"Sharpness:  {suggestions['sharpness']}  -> makes edges crisper or softer")
    print(f"Background darkening: {suggestions['background']}  -> darkens the darker parts more or less")
    print(f"Glow/detail boost: {suggestions['glow']}  -> adds or reduces soft glow and detail")
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
        brightness = 1.5
        background = 0.82
    elif average_brightness < 100:
        brightness = 1.4
    elif average_brightness > 190:
        brightness = 1.1
        background = 0.95
    elif average_brightness > 160:
        brightness = 1.2

    if contrast_spread < 30:
        contrast = 1.6
    elif contrast_spread < 50:
        contrast = 1.5
    elif contrast_spread > 80:
        contrast = 1.2

    if color_strength < 20:
        color = 1.85
    elif color_strength < 35:
        color = 1.7
    elif color_strength > 60:
        color = 1.3

    if edge_strength < 8:
        sharpness = 1.4
        glow = 0.35
    elif edge_strength < 13:
        sharpness = 1.25
        glow = 0.3
    elif edge_strength > 20:
        sharpness = 1.0
        glow = 0.15

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
    image = image.convert("RGB")

    image = ImageEnhance.Brightness(image).enhance(brightness)
    image = ImageOps.autocontrast(image, cutoff=1)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    image = ImageEnhance.Color(image).enhance(color)
    image = darken_background(image, background)
    image = add_soft_glow(image, glow)
    image = ImageEnhance.Sharpness(image).enhance(sharpness)

    return image.filter(ImageFilter.SMOOTH)


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
