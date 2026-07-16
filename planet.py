from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageStat


DEFAULT_SETTINGS = {
    "brightness": 1.1,
    "contrast": 1.3,
    "color": 1.4,
    "sharpness": 1.8,
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
    suggestions = suggest_planet_settings(image)

    print("Suggested starting values based on this image:")
    print(f"Brightness: {suggestions['brightness']}  -> makes the whole image lighter or darker")
    print(f"Contrast:   {suggestions['contrast']}  -> increases or reduces the difference between bright and dark areas")
    print(f"Color:      {suggestions['color']}  -> boosts or reduces color saturation")
    print(f"Sharpness:  {suggestions['sharpness']}  -> makes edges crisper or softer")
    print()

    brightness = ask_for_number("Brightness", suggestions["brightness"])
    contrast = ask_for_number("Contrast", suggestions["contrast"])
    color = ask_for_number("Color balance / saturation", suggestions["color"])
    sharpness = ask_for_number("Sharpness", suggestions["sharpness"])

    processed = enhance_planet(image, brightness, contrast, color, sharpness)

    output_path = make_output_path(path)
    processed.save(output_path)

    print()
    print(f"Enhanced image saved: {output_path}")

    show_images(image, processed)


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


def suggest_planet_settings(image):
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

    if average_brightness < 60:
        brightness = 1.35
    elif average_brightness < 90:
        brightness = 1.2
    elif average_brightness > 200:
        brightness = 0.95
    elif average_brightness > 170:
        brightness = 1.05

    if contrast_spread < 30:
        contrast = 1.3
    elif contrast_spread < 50:
        contrast = 1.2
    elif contrast_spread > 80:
        contrast = 1.05

    if color_strength < 25:
        color = 1.35
    elif color_strength < 40:
        color = 1.2
    elif color_strength > 70:
        color = 1.05

    if edge_strength < 8:
        sharpness = 1.2
    elif edge_strength < 13:
        sharpness = 1.1
    elif edge_strength > 20:
        sharpness = 1.0

    return {
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "color": round(color, 2),
        "sharpness": round(sharpness, 2),
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
    
    color_variance = (r_stat.stddev[0] + g_stat.stddev[0] + b_stat.stddev[0]) / 3

    return color_variance


def enhance_planet(image, brightness, contrast, color, sharpness):
    image = image.convert("RGB")

    image = ImageEnhance.Brightness(image).enhance(brightness)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    image = ImageEnhance.Color(image).enhance(color)

    if sharpness > 1.2:
        image = ImageEnhance.Sharpness(image).enhance(sharpness)
    else:
        image = image.filter(ImageFilter.SMOOTH)

    return image.filter(ImageFilter.SMOOTH)


def make_output_path(path):
    output_folder = path.parent / "outputs"
    output_folder.mkdir(exist_ok=True)

    return output_folder / f"{path.stem}_planet_enhanced{path.suffix}"


def show_images(original, processed):
    processed.show(title="Enhanced Image") 