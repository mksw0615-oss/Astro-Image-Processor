from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageFilter, ImageStat

MAX_ANALYSIS_DIMENSION = 700


def prepare_analysis_image(image_path, max_dimension=MAX_ANALYSIS_DIMENSION):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    if max(width, height) > max_dimension:
        scale = max_dimension / max(width, height)
        image = image.resize(
            (int(width * scale), int(height * scale)), Image.Resampling.LANCZOS
        )
    gray = np.array(image.convert("L"), dtype=np.uint8)
    return image, gray


def analyze_image_quality(image_path, detected_type=None):
    path = Path(image_path)

    if not path.exists():
        return None

    try:
        image, gray = prepare_analysis_image(path)
    except Exception:
        return None

    average_brightness = float(gray.mean())
    contrast_spread = float(gray.std())
    max_pixel = int(gray.max())
    bright_ratio = get_bright_pixel_ratio(gray, threshold=230)
    bright_object_score = get_bright_object_score(gray, threshold=180)
    point_like_score = get_point_like_score(gray)
    edge_strength = get_edge_strength(gray)

    object_name = classify_object_name(
        detected_type, gray, bright_object_score, point_like_score, average_brightness
    )
    confidence = estimate_confidence(
        detected_type, bright_object_score, contrast_spread, average_brightness, point_like_score
    )

    issues = {
        "Overexposed": severity_label(
            get_overexposure_score(gray, average_brightness, bright_ratio, max_pixel)
        ),
        "Atmospheric dispersion": severity_label(get_dispersion_score(image)),
        "Focus quality": severity_label(get_focus_score(gray)),
        "Tracking drift": severity_label(get_tracking_drift_score(gray, edge_strength)),
    }

    summary = {
        "average_brightness": average_brightness,
        "contrast_spread": contrast_spread,
        "edge_strength": edge_strength,
        "bright_ratio": bright_ratio,
        "bright_object_score": bright_object_score,
        "point_like_score": point_like_score,
    }
    recommendations = build_recommendations(issues, detected_type, summary)

    return {
        "object_name": object_name,
        "confidence": confidence,
        "issues": issues,
        "recommendations": recommendations,
        "summary": summary,
    }


def format_detection_line(analysis):
    if not analysis:
        return "Detected Object: Unknown"

    return f"Detected Object: {analysis['object_name']} ({analysis['confidence']}%)"


def print_quality_report(analysis):
    if not analysis:
        print("Could not analyze image quality.")
        return

    print("Issues:")
    for name, severity in analysis["issues"].items():
        print(f"• {name}: {severity}")

    if analysis["recommendations"]:
        print()
        print("Recommendation:")
        for recommendation in analysis["recommendations"]:
            print(f"✓ {recommendation}")


def severity_label(score):
    if score >= 0.8:
        return "Severe"
    if score >= 0.6:
        return "Moderate"
    if score >= 0.4:
        return "Fair"
    if score >= 0.2:
        return "Low"
    return "None"


def get_bright_pixel_ratio(gray, threshold):
    bright_pixels = int(np.count_nonzero(gray >= threshold))
    return float(bright_pixels) / max(1, gray.size)


def get_edge_strength(gray):
    edges = Image.fromarray(gray).filter(ImageFilter.FIND_EDGES)
    return float(ImageStat.Stat(edges).mean[0])


def get_bright_object_score(gray, threshold):
    mask = gray >= threshold
    if not mask.any():
        return 0.0

    y, x = np.nonzero(mask)
    count = y.size
    area_fraction = float(count) / gray.size
    bbox_width = x.max() - x.min() + 1
    bbox_height = y.max() - y.min() + 1
    bbox_area = bbox_width * bbox_height
    fill_ratio = float(count) / bbox_area if bbox_area else 0.0
    aspect_ratio = (
        min(bbox_width, bbox_height) / max(bbox_width, bbox_height)
        if bbox_width and bbox_height
        else 0.0
    )

    return min(1.0, (area_fraction / 0.03) * 0.5 + fill_ratio * 0.3 + aspect_ratio * 0.2)


def get_point_like_score(gray):
    mask = gray > 0
    if not mask.any():
        return 0.0

    y, x = np.nonzero(mask)
    bbox_width = x.max() - x.min() + 1
    bbox_height = y.max() - y.min() + 1
    area = bbox_width * bbox_height
    compactness = float(area) / gray.size if area else 0.0
    aspect_ratio = (
        min(bbox_width, bbox_height) / max(bbox_width, bbox_height)
        if bbox_width and bbox_height
        else 0.0
    )

    return min(1.0, compactness * 0.7 + aspect_ratio * 0.3)


def get_overexposure_score(gray, average_brightness, bright_ratio, max_pixel):
    score = max(0.0, bright_ratio * 4.0 - 0.05)
    score += 0.1 if max_pixel >= 245 else 0.0
    score += 0.1 if average_brightness > 180 else 0.0
    return min(1.0, score)


def get_dispersion_score(image):
    red, _, blue = image.split()
    red_edges = red.filter(ImageFilter.FIND_EDGES)
    blue_edges = blue.filter(ImageFilter.FIND_EDGES)
    diff = ImageChops.difference(red_edges, blue_edges)
    score = float(ImageStat.Stat(diff).mean[0]) / 120.0
    return min(1.0, score)


def get_focus_score(gray):
    lap = laplacian_variance(gray)
    score = 1.0 - min(1.0, lap / 1200.0)
    return max(0.0, score)


def laplacian_variance(gray):
    arr = np.array(gray, dtype=np.float32)
    padded = np.pad(arr, 1, mode="reflect")
    lap = (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        - 4.0 * padded[1:-1, 1:-1]
    )
    return float(np.var(lap))


def get_tracking_drift_score(gray, edge_strength):
    mask = gray > 0
    if not mask.any():
        return 0.0

    y, x = np.nonzero(mask)
    width = x.max() - x.min() + 1
    height = y.max() - y.min() + 1
    aspect_ratio = max(width / height, height / width)
    ratio_score = min(1.0, abs(aspect_ratio - 1.0) * 0.7)
    edge_score = 1.0 - min(1.0, edge_strength / 24.0)
    return min(1.0, ratio_score * 0.6 + edge_score * 0.4)


def classify_object_name(detected_type, gray, bright_object_score, point_like_score, average_brightness):
    if detected_type == "moon":
        return "Moon"
    if detected_type == "nebula":
        return "Nebula"
    if detected_type == "galaxy":
        return "Galaxy"

    if detected_type == "planet":
        mask = gray > 0
        if mask.any():
            y, x = np.nonzero(mask)
            width = x.max() - x.min() + 1
            height = y.max() - y.min() + 1
            aspect_ratio = max(width / height, height / width)
            if aspect_ratio > 1.35 and bright_object_score > 0.55:
                return "Saturn"

        if average_brightness > 170 and point_like_score > 0.75:
            return "Jupiter"

        return "Planet"

    return "Astronomical Object"


def estimate_confidence(detected_type, bright_object_score, contrast_spread, average_brightness, point_like_score):
    score = 0.5
    if detected_type == "moon":
        score += 0.18
    elif detected_type == "planet":
        score += 0.16
    elif detected_type == "galaxy":
        score += 0.14
    elif detected_type == "nebula":
        score += 0.14

    score += min(0.12, bright_object_score * 0.12)
    score += min(0.08, contrast_spread / 200.0)
    score += 0.05 if average_brightness > 60 else 0.0
    score += min(0.05, point_like_score * 0.05)

    return int(min(0.99, score) * 100)


def build_recommendations(issues, detected_type, summary):
    recommendations = []
    overexposed = issues.get("Overexposed")
    dispersion = issues.get("Atmospheric dispersion")
    focus = issues.get("Focus quality")
    drift = issues.get("Tracking drift")
    average_brightness = summary.get("average_brightness", 0.0)
    contrast_spread = summary.get("contrast_spread", 0.0)
    edge_strength = summary.get("edge_strength", 0.0)
    bright_object_score = summary.get("bright_object_score", 0.0)
    point_like_score = summary.get("point_like_score", 0.0)

    if detected_type == "moon":
        if overexposed in {"Moderate", "Severe"} or average_brightness > 170:
            recommendations.append("Lower exposure to avoid overexposed lunar surface.")
            recommendations.append("Use a shorter shutter speed because the Moon is very bright.")
        if focus in {"Fair", "Moderate", "Severe"}:
            recommendations.append("Increase sharpness moderately to reveal surface details.")
        if contrast_spread < 45:
            recommendations.append("Observe near first/last quarter for better crater shadows.")

    elif detected_type == "planet":
        if dispersion in {"Moderate", "Severe"} or edge_strength < 10:
            recommendations.append("Reduce chromatic aberration caused by atmospheric dispersion.")
        if focus in {"Fair", "Moderate", "Severe"} or bright_object_score < 0.6:
            recommendations.append("Capture a video instead of a single image.")
        if drift in {"Moderate", "Severe"}:
            recommendations.append("Observe when the planet is higher above the horizon.")
        if bright_object_score < 0.75 or point_like_score < 0.65:
            recommendations.append("Use higher magnification or crop around the planet.")

    elif detected_type in {"galaxy", "nebula"}:
        if average_brightness < 90 or contrast_spread < 55:
            recommendations.append("Use longer exposure or stack multiple images.")
        if average_brightness > 120 or bright_object_score < 0.2:
            recommendations.append("Observe under darker skies with less light pollution.")
        if drift in {"Moderate", "Severe"} or edge_strength < 8:
            recommendations.append("Improve tracking accuracy to avoid star trailing.")
        if contrast_spread < 70:
            recommendations.append("Increase contrast carefully to preserve faint structures.")

    if overexposed in {"Moderate", "Severe"} and detected_type != "moon":
        recommendations.append("Lower exposure")
    if dispersion in {"Moderate", "Severe"} and detected_type != "planet":
        recommendations.append("Capture video instead of single frame")
    if focus in {"Fair", "Moderate", "Severe"}:
        recommendations.append("Refocus and use a steady mount")
    if drift in {"Moderate", "Severe"} and detected_type not in {"galaxy", "nebula"}:
        recommendations.append("Observe when the object is higher in the sky")

    if not recommendations:
        recommendations.append("Your capture looks good. Keep the current setup and settings.")

    unique = []
    for recommendation in recommendations:
        if recommendation not in unique:
            unique.append(recommendation)
    return unique
