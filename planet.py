from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageStat

import quality


DEFAULT_SETTINGS = {
    "local_contrast": 0.75,
    "chromatic_aberration": 0.35,
    "deconvolution": 0.55,
    "crop_padding": 8,
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

    analysis = quality.analyze_image_quality(path, detected_type="planet")
    print()
    quality.print_quality_report(analysis)
    print()

    suggestions = suggest_planet_settings(image)

    print("What each value does:")
    print("Crop padding -> crops around the main object")
    print("Local contrast -> increases local detail without changing the whole image globally")
    print("Chromatic aberration -> removes color fringing around edges")
    print("Deconvolution/sharpening -> sharpens the object and restores detail")
    print()

    crop_padding = ask_for_int("Crop padding", suggestions["crop_padding"])
    local_contrast = ask_for_number("Local contrast", suggestions["local_contrast"])
    chromatic_aberration = ask_for_number("Chromatic aberration", suggestions["chromatic_aberration"])
    deconvolution = ask_for_number("Deconvolution/sharpening", suggestions["deconvolution"])

    processed = enhance_planet(image, crop_padding, local_contrast, chromatic_aberration, deconvolution)

    output_path = make_output_path(path)
    processed.save(output_path)

    print()
    print(f"Enhanced image saved: {output_path}")

    show_processed_image(processed)


def clean_path(image_path):
    return image_path.strip().strip('"').strip("'")


def ask_for_int(setting_name, default_value):
    user_input = input(f"{setting_name} [{default_value}]: ").strip()

    if user_input == "":
        return default_value

    try:
        value = int(user_input)
    except ValueError:
        print(f"Invalid input. Using {default_value}.")
        return default_value

    return value


def ask_for_number(setting_name, default_value):
    user_input = input(f"{setting_name} [{default_value}]: ").strip()

    if user_input == "":
        return default_value

    try:
        return float(user_input)
    except ValueError:
        print(f"Invalid input. Using {default_value}.")
        return default_value


def suggest_planet_settings(image):
    gray = image.convert("L")
    stat = ImageStat.Stat(gray)
    average_brightness = stat.mean[0]

    local_contrast = DEFAULT_SETTINGS["local_contrast"]
    chromatic_aberration = DEFAULT_SETTINGS["chromatic_aberration"]
    deconvolution = DEFAULT_SETTINGS["deconvolution"]
    crop_padding = DEFAULT_SETTINGS["crop_padding"]

    if average_brightness < 80:
        local_contrast = 0.9
        deconvolution = 0.7
        crop_padding = 10
    elif average_brightness > 160:
        local_contrast = 0.6
        crop_padding = 6

    return {
        "local_contrast": round(local_contrast, 2),
        "chromatic_aberration": round(chromatic_aberration, 2),
        "deconvolution": round(deconvolution, 2),
        "crop_padding": crop_padding,
    }


def enhance_planet(image, crop_padding, local_contrast, chromatic_aberration, deconvolution):
    image = image.convert("RGB")
    cropped = crop_around_object(image, padding=max(0, int(crop_padding)))

    if cropped.size != image.size:
        return cropped.resize(image.size, Image.Resampling.LANCZOS)

    local_contrast_image = apply_local_contrast(cropped, strength=max(0.0, float(local_contrast)))
    chroma_corrected = correct_chromatic_aberration(local_contrast_image, amount=max(0.0, float(chromatic_aberration)))
    return apply_deconvolution(chroma_corrected, strength=max(0.0, float(deconvolution)))


def apply_local_contrast(image, strength):
    rgb = np.array(image, dtype=np.float32)
    luminance = np.clip(
        rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114,
        0,
        255,
    )

    blurred = np.array(Image.fromarray(luminance.astype("uint8")).filter(ImageFilter.GaussianBlur(radius=2)), dtype=np.float32)
    enhanced_luminance = np.clip(luminance + (luminance - blurred) * strength, 0, 255)
    scale = np.clip(enhanced_luminance / (luminance + 1e-6), 0.0, 3.0)

    adjusted = rgb * np.stack([scale, scale, scale], axis=-1)
    adjusted = np.clip(adjusted, 0, 255).astype("uint8")

    return Image.fromarray(adjusted, mode="RGB")


def correct_chromatic_aberration(image, amount):
    if amount <= 0:
        return image

    rgb = np.array(image, dtype=np.float32)
    red = rgb[..., 0]
    green = rgb[..., 1]
    blue = rgb[..., 2]
    shift = max(1, int(round(amount * 3)))
    shifted_red = np.roll(red, shift, axis=1)
    shifted_blue = np.roll(blue, -shift, axis=1)

    corrected = np.stack(
        [
            np.clip(0.85 * red + 0.15 * shifted_red, 0, 255),
            green,
            np.clip(0.85 * blue + 0.15 * shifted_blue, 0, 255),
        ],
        axis=-1,
    )

    return Image.fromarray(corrected.astype("uint8"), mode="RGB")


def apply_deconvolution(image, strength):
    if strength <= 0:
        return image

    percent = int(120 + strength * 180)
    return image.filter(ImageFilter.UnsharpMask(radius=1, percent=percent, threshold=3))


def crop_around_object(image, padding=8):
    gray = image.convert("L")
    width, height = gray.size

    bright_pixels = []
    for y in range(height):
        for x in range(width):
            value = gray.getpixel((x, y))
            if value >= 120:
                bright_pixels.append((x, y))

    if not bright_pixels:
        return image.copy()

    min_x = min(x for x, _ in bright_pixels)
    max_x = max(x for x, _ in bright_pixels)
    min_y = min(y for _, y in bright_pixels)
    max_y = max(y for _, y in bright_pixels)

    left = max(0, min_x - padding)
    upper = max(0, min_y - padding)
    right = min(image.width, max_x + padding)
    lower = min(image.height, max_y + padding)

    if right - left < 10 or lower - upper < 10:
        return image.copy()

    return image.crop((left, upper, right, lower))

def make_output_path(path):
    output_folder = path.parent / "outputs"
    output_folder.mkdir(exist_ok=True)

    return output_folder / f"{path.stem}_planet_enhanced{path.suffix}"


def show_processed_image(processed):
    processed.show(title="Enhanced Planet Image") 