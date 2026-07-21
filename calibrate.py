from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def process():
    print("Calibration mode selected.")
    print()
    print("Calibration cleans your image before stacking.")
    print("Enter the path to one image file to calibrate it.")
    print()

    light_path = ask_for_image_file("Enter the image file path")

    if light_path is None:
        return

    print()
    print("Calibrating image...")

    light = open_image_as_array(light_path)

    if light is None:
        print("Could not read that image file.")
        return

    calibrated = calibrate_light(light, None, None, None)
    output_folder = light_path.parent / "outputs" / "calibrated"
    output_folder.mkdir(parents=True, exist_ok=True)
    output_path = output_folder / f"{light_path.stem}_calibrated.png"
    save_array_as_image(calibrated, output_path)

    print()
    print(f"Saved calibrated image to: {output_path}")


def ask_for_image_file(prompt):
    file_path = clean_path(input(f"{prompt}: "))
    path = Path(file_path)

    if not path.exists():
        print("Could not find that file. Check the path and try again.")
        return None

    if not path.is_file():
        print("That path is not a file.")
        return None

    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        print("That file does not look like a supported image file.")
        return None

    return path


def clean_path(path):
    return path.strip().strip('"').strip("'")


def open_image_as_array(path):
    try:
        image = Image.open(path).convert("RGB")
    except Exception:
        return None

    return np.array(image, dtype=np.float32)


def calibrate_light(light, master_dark, master_flat, master_bias):
    calibrated = light.copy()

    if master_bias is not None and master_bias.shape == calibrated.shape:
        calibrated = calibrated - master_bias

    if master_dark is not None and master_dark.shape == calibrated.shape:
        calibrated = calibrated - master_dark

    if master_flat is not None and master_flat.shape == calibrated.shape:
        flat = master_flat.copy()

        if master_bias is not None and master_bias.shape == flat.shape:
            flat = flat - master_bias

        flat_average = np.mean(flat)

        if flat_average > 0:
            normalized_flat = flat / flat_average
            normalized_flat = np.maximum(normalized_flat, 0.1)
            calibrated = calibrated / normalized_flat

    return np.clip(calibrated, 0, 255)


def save_array_as_image(image_array, output_path):
    image_array = image_array.astype(np.uint8)
    image = Image.fromarray(image_array)
    image.save(output_path)