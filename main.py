from pathlib import Path

from PIL import Image, ImageFilter, ImageStat

import galaxy
import moon
import nebula
import planet


def detect_image_type(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception:
        print("Could not open that file as an image.")
        return None

    gray = image.convert("L")
    stat = ImageStat.Stat(gray)
    average_brightness = stat.mean[0]
    contrast_spread = stat.stddev[0]
    edge_strength = get_edge_strength(gray)

    hsv = image.convert("HSV")
    saturation_stat = ImageStat.Stat(hsv.split()[1])
    saturation = saturation_stat.mean[0]

    brightest_pixel = max(gray.getextrema())
    diffuse_region_score = get_diffuse_region_score(image)
    bright_object_score = get_bright_object_score(gray, threshold=180)
    point_like_score = get_point_like_score(gray)

    if (
        brightest_pixel >= 220
        and bright_object_score > 0.5
        and contrast_spread > 20
        and saturation < 80
        and point_like_score < 0.85
    ):
        return "moon"

    if diffuse_region_score > 0.6 and saturation > 45 and average_brightness < 220:
        return "nebula"

    if (
        brightest_pixel >= 170
        and bright_object_score > 0.12
        and point_like_score > 0.55
        and contrast_spread < 120
        and average_brightness < 220
    ):
        return "planet"

    if (
        saturation > 35
        and average_brightness < 200
        and point_like_score < 0.6
        and diffuse_region_score > 0.1
    ):
        return "galaxy"

    return "planet"


def get_edge_strength(gray):
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_stat = ImageStat.Stat(edges)
    return edge_stat.mean[0]


def get_bright_object_score(gray, threshold):
    width, height = gray.size
    count = 0
    min_x = width
    min_y = height
    max_x = 0
    max_y = 0

    for y in range(height):
        for x in range(width):
            value = gray.getpixel((x, y))
            if value >= threshold:
                count += 1
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)

    if count == 0:
        return 0.0

    area_fraction = count / (width * height)
    bbox_width = max_x - min_x + 1
    bbox_height = max_y - min_y + 1
    bbox_area = bbox_width * bbox_height
    fill_ratio = count / bbox_area if bbox_area else 0.0
    aspect_ratio = min(bbox_width, bbox_height) / max(bbox_width, bbox_height) if max(bbox_width, bbox_height) else 0.0

    return min(1.0, (area_fraction / 0.03) * 0.5 + fill_ratio * 0.3 + aspect_ratio * 0.2)


def get_diffuse_region_score(image):
    gray = image.convert("L")
    width, height = gray.size
    sample = gray.resize((max(20, width // 10), max(20, height // 10)))
    pixels = list(sample.get_flattened_data())

    if not pixels:
        return 0.0

    mean_brightness = sum(pixels) / len(pixels)
    variance = sum((p - mean_brightness) ** 2 for p in pixels) / len(pixels)

    return min(1.0, variance / 5000.0)


def get_point_like_score(gray):
    width, height = gray.size
    if width <= 0 or height <= 0:
        return 0.0

    bbox = gray.getbbox()
    if not bbox:
        return 0.0

    left, upper, right, lower = bbox
    bbox_width = right - left + 1
    bbox_height = lower - upper + 1
    bbox_area = bbox_width * bbox_height
    image_area = width * height

    if bbox_area <= 0:
        return 0.0

    compactness = bbox_area / image_area
    aspect_ratio = min(bbox_width, bbox_height) / max(bbox_width, bbox_height) if max(bbox_width, bbox_height) else 0.0

    return min(1.0, compactness * 0.7 + aspect_ratio * 0.3)


def main():
    print("Welcome to Astro Image Processor!")
    print()

    while True:
        image_path = input("Enter the image path: ").strip()
        if not image_path:
            print("No image path provided. Exiting.")
            break

        print("Processing image...")

        image_path = image_path.strip().strip('"').strip("'")
        detected_type = detect_image_type(image_path)

        if detected_type is None:
            continue

        if detected_type == "moon":
            print("Selected mode: moon")
            moon.process(image_path)
        elif detected_type == "planet":
            print("Selected mode: planet")
            planet.process(image_path)
        elif detected_type == "galaxy":
            print("Selected mode: galaxy")
            galaxy.process(image_path)
        elif detected_type == "nebula":
            print("Selected mode: nebula")
            nebula.process(image_path)
        else:
            print("Could not determine the image type automatically.")


if __name__ == "__main__":
    main()
