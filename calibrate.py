from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def process():
    print("Calibration mode selected.")
    print()
    print("Calibration cleans your real space images before stacking.")
    print("You need light frames. Dark, flat, and bias folders are optional.")
    print()

    light_folder = ask_for_folder("Enter the light frames folder path")

    if light_folder is None:
        return

    dark_folder = ask_for_optional_folder("Enter the dark frames folder path, or press Enter to skip")
    flat_folder = ask_for_optional_folder("Enter the flat frames folder path, or press Enter to skip")
    bias_folder = ask_for_optional_folder("Enter the bias frames folder path, or press Enter to skip")

    light_paths = find_image_files(light_folder)

    if len(light_paths) == 0:
        print("No light frame images found.")
        return

    print()
    print(f"Found {len(light_paths)} light frame(s).")
    print("Building calibration frames...")

    master_dark = build_master_frame(dark_folder, "dark") if dark_folder else None
    master_flat = build_master_frame(flat_folder, "flat") if flat_folder else None
    master_bias = build_master_frame(bias_folder, "bias") if bias_folder else None

    output_folder = light_folder / "outputs" / "calibrated"
    output_folder.mkdir(parents=True, exist_ok=True)

    print("Calibrating light frames...")

    saved_count = 0

    for light_path in light_paths:
        light = open_image_as_array(light_path)

        if light is None:
            print(f"Skipping unreadable image: {light_path.name}")
            continue

        calibrated = calibrate_light(light, master_dark, master_flat, master_bias)
        output_path = output_folder / f"{light_path.stem}_calibrated.png"
        save_array_as_image(calibrated, output_path)
        saved_count += 1

    print()
    print(f"Saved {saved_count} calibrated image(s) to: {output_folder}")


def ask_for_folder(prompt):
    folder_path = clean_path(input(f"{prompt}: "))
    folder = Path(folder_path)

    if not folder.exists():
        print("Could not find that folder. Check the path and try again.")
        return None

    if not folder.is_dir():
        print("That path is not a folder.")
        return None

    return folder


def ask_for_optional_folder(prompt):
    folder_path = clean_path(input(f"{prompt}: "))

    if folder_path == "":
        return None

    folder = Path(folder_path)

    if not folder.exists() or not folder.is_dir():
        print("Could not use that folder. Skipping it.")
        return None

    return folder


def clean_path(path):
    return path.strip().strip('"').strip("'")


def find_image_files(folder):
    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def build_master_frame(folder, frame_type):
    image_paths = find_image_files(folder)

    if len(image_paths) == 0:
        print(f"No {frame_type} frames found. Skipping {frame_type} calibration.")
        return None

    arrays = []

    for path in image_paths:
        image_array = open_image_as_array(path)

        if image_array is not None:
            arrays.append(image_array)

    if len(arrays) == 0:
        return None

    target_shape = arrays[0].shape
    matching_arrays = [array for array in arrays if array.shape == target_shape]

    print(f"Created master {frame_type} from {len(matching_arrays)} frame(s).")
    return np.median(matching_arrays, axis=0)


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