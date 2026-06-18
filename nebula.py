from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


def process(image_path):
    print("Nebula enhancement selected.")

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

    print(f"Processing image: {path}")
    print()
    print("Recommended starting values:")
    print("Brightness: 1.30")
    print("Contrast:   1.40")
    print("Color:      1.60")
    print("Sharpness:  1.20")
    print("Background darkening: 0.88")
    print("Glow/detail boost: 0.25")
    print()

    brightness = ask_for_number("Brightness", 1.30)
    contrast = ask_for_number("Contrast", 1.40)
    color = ask_for_number("Color balance / saturation", 1.60)
    sharpness = ask_for_number("Sharpness", 1.20)
    background = ask_for_number("Background darkening", 0.88)
    glow = ask_for_number("Glow/detail boost", 0.25)

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
