from pathlib import Path

from PIL import Image, ImageFilter, ImageStat

import calibrate
import galaxy
import moon
import nebula
import planet
import quality
import stack


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
    halo_score = get_extended_halo_score(gray)
    galaxy_structure_score = get_galaxy_structure_score(gray)
    lunar_disc_score = get_lunar_disc_score(gray)

    # A low or partial Moon can have a dark face while still filling a large,
    # round area of the frame.  Its overall bright-pixel fraction is therefore
    # much lower than a full Moon's and can otherwise resemble a smooth galaxy.
    # Check its resolved limb before evaluating diffuse deep-sky structures.
    if lunar_disc_score >= 0.55:
        return "moon"

    # A galaxy needs more than a bright centre: its extended light should be
    # smooth and elliptical.  This keeps an irregular, diffuse nebula (such as
    # Orion) out of Galaxy Mode even when its core is bright.
    if galaxy_structure_score >= 0.42 and average_brightness < 220:
        return "galaxy"

    if diffuse_region_score > 0.35 and average_brightness < 220:
        return "nebula"

    # A compact galaxy can be nearly round, so retain a conservative radial
    # fallback only when at least some elliptical/smooth structure is present.
    if (
        halo_score >= 0.24
        and galaxy_structure_score >= 0.28
        and average_brightness < 220
    ):
        return "galaxy"

    if (
        (diffuse_region_score > 0.1 or halo_score >= 0.08)
        and average_brightness < 220
    ):
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


def get_lunar_disc_score(gray):
    """Score a large, resolved, nearly circular bright foreground body.

    This deliberately measures the connected shape of the lunar surface, not
    just its illuminated fraction.  A crescent or eclipsed Moon can be mostly
    dark, but its visible face still forms a broad round region; galaxies are
    normally much smaller in the frame and have a more elongated halo.
    """
    max_dimension = 500
    width, height = gray.size
    if max(width, height) > max_dimension:
        scale = max_dimension / max(width, height)
        gray = gray.resize(
            (max(1, int(width * scale)), max(1, int(height * scale))),
            Image.Resampling.LANCZOS,
        )

    width, height = gray.size
    if width < 30 or height < 30:
        return 0.0

    pixels = list(gray.getdata())
    sky_level = sorted(pixels)[len(pixels) // 2]
    peak = max(pixels)
    if peak - sky_level < 55:
        return 0.0

    # Include the dim lunar face while excluding the black sky.  Using a
    # fraction of the scene's usable range makes this work for both bright
    # crescents and deliberately underexposed lunar images.
    threshold = sky_level + max(12, int((peak - sky_level) * 0.13))
    active = [value >= threshold for value in pixels]
    visited = bytearray(width * height)
    largest = None

    for start, is_active in enumerate(active):
        if not is_active or visited[start]:
            continue

        stack = [start]
        visited[start] = 1
        count = 0
        min_x = max_x = start % width
        min_y = max_y = start // width

        while stack:
            index = stack.pop()
            x = index % width
            y = index // width
            count += 1
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)

            for neighbor_y in range(max(0, y - 1), min(height, y + 2)):
                row_start = neighbor_y * width
                for neighbor_x in range(max(0, x - 1), min(width, x + 2)):
                    neighbor = row_start + neighbor_x
                    if active[neighbor] and not visited[neighbor]:
                        visited[neighbor] = 1
                        stack.append(neighbor)

        if largest is None or count > largest[0]:
            largest = (count, min_x, min_y, max_x, max_y)

    if largest is None:
        return 0.0

    count, min_x, min_y, max_x, max_y = largest
    bbox_width = max_x - min_x + 1
    bbox_height = max_y - min_y + 1
    bbox_area = bbox_width * bbox_height
    image_area = width * height
    area_fraction = count / image_area
    fill_ratio = count / bbox_area
    aspect_ratio = min(bbox_width, bbox_height) / max(bbox_width, bbox_height)

    # The object needs to be a genuinely resolved body, rather than a compact
    # galaxy or a point source.  A partially lit lunar disc need not fill its
    # bounding box completely, hence the deliberately tolerant fill score.
    area_score = min(1.0, area_fraction / 0.10)
    fill_score = min(1.0, max(0.0, (fill_ratio - 0.32) / 0.38))
    round_score = min(1.0, max(0.0, (aspect_ratio - 0.62) / 0.30))
    return area_score * 0.40 + fill_score * 0.30 + round_score * 0.30


def get_extended_halo_score(gray):
    """Return how strongly the brightest sources are surrounded by a halo.

    A star or unresolved planet drops to the sky level a few pixels from its
    centre.  A galaxy core has a gentler brightness falloff, leaving measurable
    light in both an inner and an outer ring.  The image is reduced first so
    this test remains quick on large camera files.
    """
    max_dimension = 500
    width, height = gray.size
    if max(width, height) > max_dimension:
        scale = max_dimension / max(width, height)
        gray = gray.resize(
            (max(1, int(width * scale)), max(1, int(height * scale))),
            Image.Resampling.LANCZOS,
        )

    width, height = gray.size
    if width < 25 or height < 25:
        return 0.0

    pixels = list(gray.getdata())
    sky_level = sorted(pixels)[len(pixels) // 2]
    peak_threshold = max(sky_level + 25, 100)

    # Test a handful of separated bright candidates.  This prevents one bright
    # foreground star from hiding a fainter galaxy nucleus elsewhere in frame.
    bright_pixels = []
    for y in range(12, height - 12, 2):
        for x in range(12, width - 12, 2):
            value = gray.getpixel((x, y))
            if value >= peak_threshold:
                bright_pixels.append((value, x, y))

    # Keep the brightest candidates first, then require them to be separated
    # from one another.  Limiting this list also keeps the scan fast on a
    # star-rich image.
    candidates = []
    for value, x, y in sorted(bright_pixels, reverse=True)[:500]:
        if all((x - old_x) ** 2 + (y - old_y) ** 2 >= 20 ** 2 for _, old_x, old_y in candidates):
            candidates.append((value, x, y))
        if len(candidates) == 12:
            break

    best_score = 0.0
    for peak, center_x, center_y in candidates:
        core_total = core_count = inner_total = inner_count = halo_total = halo_count = 0
        for y in range(center_y - 12, center_y + 13):
            for x in range(center_x - 12, center_x + 13):
                distance_squared = (x - center_x) ** 2 + (y - center_y) ** 2
                value = gray.getpixel((x, y)) - sky_level
                if distance_squared <= 3 ** 2:
                    core_total += value
                    core_count += 1
                elif 4 ** 2 <= distance_squared <= 7 ** 2:
                    inner_total += value
                    inner_count += 1
                elif 8 ** 2 <= distance_squared <= 12 ** 2:
                    halo_total += value
                    halo_count += 1

        if not (core_count and inner_count and halo_count):
            continue

        core = max(0.0, core_total / core_count)
        inner = max(0.0, inner_total / inner_count)
        halo = max(0.0, halo_total / halo_count)
        contrast = max(1.0, peak - sky_level)

        # Both rings must retain light.  The outer ring is weighted more
        # heavily because it is the useful separator from an isolated point.
        score = min(1.0, (inner / contrast) * 0.35 + (halo / contrast) * 0.65)
        best_score = max(best_score, score)

    return best_score


def get_galaxy_structure_score(gray):
    """Score smooth, elliptical diffuse light surrounding a bright core.

    Nebulae can also have bright centres and halos.  Their surrounding light is
    usually irregular, whereas a galaxy has a coherent elliptical distribution
    with a gradual decline from its nucleus.  The calculation is deliberately
    performed on a blurred, reduced copy so individual stars do not dominate.
    """
    max_dimension = 500
    width, height = gray.size
    if max(width, height) > max_dimension:
        scale = max_dimension / max(width, height)
        gray = gray.resize(
            (max(1, int(width * scale)), max(1, int(height * scale))),
            Image.Resampling.LANCZOS,
        )

    width, height = gray.size
    radius = 36
    if width < radius * 2 + 1 or height < radius * 2 + 1:
        return 0.0

    blurred = gray.filter(ImageFilter.GaussianBlur(radius=3))
    values = list(blurred.getdata())
    sky_level = sorted(values)[len(values) * 30 // 100]
    candidates = []

    for y in range(radius, height - radius, 3):
        for x in range(radius, width - radius, 3):
            value = blurred.getpixel((x, y))
            if value >= sky_level + 12:
                candidates.append((value, x, y))

    selected = []
    for value, x, y in sorted(candidates, reverse=True)[:400]:
        if all((x - old_x) ** 2 + (y - old_y) ** 2 >= 30 ** 2 for _, old_x, old_y in selected):
            selected.append((value, x, y))
        if len(selected) == 10:
            break

    best_score = 0.0
    for peak, center_x, center_y in selected:
        ring_values = []
        points = []
        core_total = core_count = inner_total = inner_count = outer_total = outer_count = 0

        for y in range(center_y - radius, center_y + radius + 1):
            for x in range(center_x - radius, center_x + radius + 1):
                dx = x - center_x
                dy = y - center_y
                distance_squared = dx * dx + dy * dy
                value = blurred.getpixel((x, y))
                if 30 ** 2 <= distance_squared <= radius ** 2:
                    ring_values.append(value)
                elif distance_squared <= 28 ** 2:
                    points.append((dx, dy, value))

        if not ring_values:
            continue
        local_sky = sorted(ring_values)[len(ring_values) // 2]
        if peak - local_sky < 12:
            continue

        total_weight = weighted_x = weighted_y = 0.0
        for dx, dy, value in points:
            weight = max(0.0, value - local_sky)
            total_weight += weight
            weighted_x += dx * weight
            weighted_y += dy * weight

            distance_squared = dx * dx + dy * dy
            if distance_squared <= 5 ** 2:
                core_total += weight
                core_count += 1
            elif 7 ** 2 <= distance_squared <= 15 ** 2:
                inner_total += weight
                inner_count += 1
            elif 17 ** 2 <= distance_squared <= 28 ** 2:
                outer_total += weight
                outer_count += 1

        if not (total_weight and core_count and inner_count and outer_count):
            continue

        mean_x = weighted_x / total_weight
        mean_y = weighted_y / total_weight
        cov_xx = cov_yy = cov_xy = 0.0
        for dx, dy, value in points:
            weight = max(0.0, value - local_sky)
            cov_xx += weight * (dx - mean_x) ** 2
            cov_yy += weight * (dy - mean_y) ** 2
            cov_xy += weight * (dx - mean_x) * (dy - mean_y)

        cov_xx /= total_weight
        cov_yy /= total_weight
        cov_xy /= total_weight
        trace = cov_xx + cov_yy
        determinant_term = max(0.0, (cov_xx - cov_yy) ** 2 + 4 * cov_xy ** 2)
        major_variance = (trace + determinant_term ** 0.5) / 2
        minor_variance = (trace - determinant_term ** 0.5) / 2
        axis_ratio = (major_variance / max(minor_variance, 0.01)) ** 0.5
        if axis_ratio < 1.32:
            continue

        # Galaxies are approximately symmetric around their nucleus; Orion's
        # emission is made of uneven lobes and dark lanes.  Compare each pixel
        # in the diffuse halo with its opposite point around the bright core.
        symmetry_difference = symmetry_total = 0.0
        for dy in range(-28, 29):
            for dx in range(-28, 29):
                distance_squared = dx * dx + dy * dy
                if distance_squared > 28 ** 2 or (dy < 0 or (dy == 0 and dx <= 0)):
                    continue
                signal = max(0.0, blurred.getpixel((center_x + dx, center_y + dy)) - local_sky)
                opposite_signal = max(0.0, blurred.getpixel((center_x - dx, center_y - dy)) - local_sky)
                symmetry_difference += abs(signal - opposite_signal)
                symmetry_total += signal + opposite_signal

        symmetry_score = 1.0 - symmetry_difference / max(symmetry_total, 1.0)
        if symmetry_score < 0.62:
            continue

        core = core_total / core_count
        inner = inner_total / inner_count
        outer = outer_total / outer_count
        profile_score = min(1.0, inner / max(core * 0.35, 1.0), outer / max(core * 0.16, 1.0))
        ellipse_score = min(1.0, max(0.0, (axis_ratio - 1.15) / 0.65))
        best_score = max(
            best_score,
            ellipse_score * 0.45 + profile_score * 0.30 + symmetry_score * 0.25,
        )

    return best_score


def main():
    print("Welcome to Astro Image Processor!")
    print()
    print("Type 'calibration' to enter calibration mode")
    print("Type 'stack' to enter stacking mode")
    print("Otherwise, enter the path to an image file")
    print()

    while True:
        user_input = input("Enter 'calibration', 'stack', or an image path: ").strip()
        if not user_input:
            print("No input provided. Exiting.")
            break

        cleaned_input = user_input.strip().strip('"').strip("'")
        lowered_input = cleaned_input.lower()

        if lowered_input == "calibration":
            calibrate.process()
            continue

        if lowered_input == "stack":
            stack.process()
            continue

        print("Processing image...")
        detected_type = detect_image_type(cleaned_input)

        if detected_type is None:
            continue

        analysis = quality.analyze_image_quality(cleaned_input, detected_type=detected_type)
        print(quality.format_detection_line(analysis))
        print()

        if detected_type == "moon":
            print("Selected mode: moon")
            moon.process(cleaned_input)
        elif detected_type == "planet":
            print("Selected mode: planet")
            planet.process(cleaned_input)
        elif detected_type == "galaxy":
            print("Selected mode: galaxy")
            galaxy.process(cleaned_input)
        elif detected_type == "nebula":
            print("Selected mode: nebula")
            nebula.process(cleaned_input)
        else:
            print("Could not determine the image type automatically.")


if __name__ == "__main__":
    main()
