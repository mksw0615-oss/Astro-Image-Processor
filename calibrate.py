from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def process():
    print("Calibration mode selected.")
    print()
    print("Calibration removes camera artifacts, neutralizes the sky background,")
    print("and balances RGB colour before stacking or enhancement.")
    print("Enter the path to one image file to calibrate it.")
    print()

    light_path = ask_for_image_file("Enter the image file path")

    if light_path is None:
        return

    print()
    print("Calibrating image, neutralizing background, and balancing colour...")

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

    calibrated = np.clip(calibrated, 0, 255)

    # These operations happen before any galaxy/nebula enhancement.  They keep
    # a colour cast in the sky from being amplified later by saturation or
    # local-contrast tools.
    calibrated = neutralize_background(calibrated)
    calibrated = color_calibrate(calibrated)
    return neutralize_background(calibrated)


def estimate_sky_background(image_array, percentile=30):
    """Estimate the RGB sky level from the faintest pixels in the frame.

    Stars and the galaxy core are bright outliers, so a median of the darkest
    portion of the luminance distribution gives a more useful sky estimate
    than averaging the entire image.
    """
    pixels = image_array.reshape(-1, 3)
    luminance = (
        pixels[:, 0] * 0.2126
        + pixels[:, 1] * 0.7152
        + pixels[:, 2] * 0.0722
    )
    cutoff = np.percentile(luminance, percentile)
    sky_pixels = pixels[luminance <= cutoff]
    if sky_pixels.size == 0:
        sky_pixels = pixels
    return np.median(sky_pixels, axis=0)


def neutralize_background(image_array, target_sky_level=12.0):
    """Subtract the estimated sky and leave a dark, neutral baseline."""
    background = estimate_sky_background(image_array)
    return np.clip(image_array - background + target_sky_level, 0, 255)


def color_calibrate(image_array):
    """Balance RGB gains using bright, near-neutral reference pixels.

    Low-saturation bright pixels are normally stars or a galaxy's white core.
    Equalising their RGB medians corrects a camera/sky colour cast without
    treating the desired blue arms or brown dust lanes as something to remove.
    """
    pixels = image_array.reshape(-1, 3)
    luminance = (
        pixels[:, 0] * 0.2126
        + pixels[:, 1] * 0.7152
        + pixels[:, 2] * 0.0722
    )
    saturation = pixels.max(axis=1) - pixels.min(axis=1)
    bright_limit = np.percentile(luminance, 70)
    highlight_limit = np.percentile(luminance, 99.5)
    neutral_limit = np.percentile(saturation, 45)
    reference_mask = (
        (luminance >= bright_limit)
        & (luminance <= highlight_limit)
        & (saturation <= neutral_limit)
    )

    if np.count_nonzero(reference_mask) < 50:
        reference_mask = (luminance >= bright_limit) & (luminance <= highlight_limit)

    reference = np.median(pixels[reference_mask], axis=0)
    target = float(np.mean(reference))
    gains = target / np.maximum(reference, 1.0)
    gains = np.clip(gains, 0.6, 1.6)
    return np.clip(image_array * gains, 0, 255)


def save_array_as_image(image_array, output_path):
    image_array = image_array.astype(np.uint8)
    image = Image.fromarray(image_array)
    image.save(output_path)
