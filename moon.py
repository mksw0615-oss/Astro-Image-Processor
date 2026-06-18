from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


def process(image_path):
    print("Moon enhancement selected.")

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
    print("Brightness: 1.05")
    print("Contrast:   1.35")
    print("Sharpness:  1.8")
    print("Detail:     1.3")
    print()

    brightness = ask_for_number("Brightness", 1.05)
    contrast = ask_for_number("Contrast", 1.35)
    sharpness = ask_for_number("Sharpness", 1.8)
    detail = ask_for_number("Detail boost", 1.3)

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
