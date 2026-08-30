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
    saturation = get_mean_saturation(image)
    shadow_ratio = float(np.count_nonzero(gray <= 8)) / max(1, gray.size)
    noise_score = get_noise_score(gray)
    color_cast = get_color_cast_score(image)

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
        "saturation": saturation,
        "shadow_ratio": shadow_ratio,
        "noise_score": noise_score,
        "color_cast": color_cast,
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
        print("Top image-driven recommendations:")
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


def get_mean_saturation(image):
    """Return the average HSV saturation, scaled from 0.0 to 1.0."""
    saturation = np.array(image.convert("HSV"), dtype=np.uint8)[:, :, 1]
    return float(saturation.mean()) / 255.0


def get_noise_score(gray):
    """Estimate fine-grain noise without treating broad object detail as noise."""
    smooth = np.array(
        Image.fromarray(gray).filter(ImageFilter.GaussianBlur(radius=1.2)),
        dtype=np.float32,
    )
    residual = np.abs(gray.astype(np.float32) - smooth)
    return min(1.0, float(residual.std()) / 18.0)


def get_color_cast_score(image):
    """Estimate an overall channel imbalance, scaled from 0.0 to 1.0."""
    channel_means = np.array(image, dtype=np.float32).mean(axis=(0, 1))
    return min(1.0, float(channel_means.max() - channel_means.min()) / 90.0)


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


RECOMMENDATION_LIBRARY = (
    ("lower_exposure", "Lower exposure to protect clipped highlights."),
    ("shorter_shutter", "Use a shorter shutter speed to retain bright surface detail."),
    ("raise_exposure", "Increase exposure slightly to bring faint signal above the background."),
    ("stack_frames", "Stack more frames to improve faint detail and reduce random noise."),
    ("refocus", "Refocus carefully, then use only moderate sharpening."),
    ("steady_mount", "Use a steadier mount or shorter sub-exposures to limit motion blur."),
    ("improve_tracking", "Improve tracking alignment to avoid trailing during longer captures."),
    ("correct_dispersion", "Correct atmospheric color separation before increasing sharpness."),
    ("higher_altitude", "Capture when the object is higher in the sky for steadier air."),
    ("video_capture", "Capture a video sequence and keep the sharpest frames."),
    ("increase_contrast", "Increase local contrast gently to reveal faint structure."),
    ("preserve_highlights", "Reduce contrast or sharpening slightly around the brightest areas."),
    ("darker_sky", "Use darker skies or stronger background subtraction to improve contrast."),
    ("crop_object", "Crop more tightly around the object before enlarging or sharpening."),
    ("correct_color_cast", "Neutralize the overall color cast before boosting saturation."),
    ("boost_color", "Apply a modest color boost after contrast adjustment."),
    ("reduce_saturation", "Reduce saturation slightly to keep color noise under control."),
    ("denoise", "Apply light noise reduction before final sharpening."),
    ("calibrate_frames", "Use dark and flat calibration frames to reduce camera artifacts."),
    ("phase_timing", "Capture near lunar first or last quarter for stronger crater shadows."),
)


def clamp_score(value):
    return min(1.0, max(0.0, float(value)))


def issue_score(issues, name):
    return {"None": 0.0, "Low": 0.25, "Fair": 0.5, "Moderate": 0.75, "Severe": 1.0}.get(
        issues.get(name, "None"), 0.0
    )


def recommendation_score(key, issues, detected_type, summary):
    """Score one recommendation from measured image characteristics.

    Scores intentionally come from image analysis metrics, not a fixed list per
    object type.  Object type only prevents advice that would not fit a target.
    """
    overexposed = issue_score(issues, "Overexposed")
    dispersion = issue_score(issues, "Atmospheric dispersion")
    focus = issue_score(issues, "Focus quality")
    drift = issue_score(issues, "Tracking drift")
    brightness = summary.get("average_brightness", 0.0)
    contrast = summary.get("contrast_spread", 0.0)
    bright_object = summary.get("bright_object_score", 0.0)
    saturation = summary.get("saturation", 0.0)
    shadows = summary.get("shadow_ratio", 0.0)
    noise = summary.get("noise_score", 0.0)
    color_cast = summary.get("color_cast", 0.0)
    deep_sky = detected_type in {"galaxy", "nebula"}
    resolved_body = detected_type in {"moon", "planet"}
    faintness = clamp_score((90.0 - brightness) / 90.0)
    low_contrast = clamp_score((65.0 - contrast) / 65.0)

    scores = {
        "lower_exposure": overexposed,
        "shorter_shutter": overexposed * (1.0 if resolved_body else 0.25),
        "raise_exposure": max(faintness, clamp_score((shadows - 0.70) / 0.25)),
        "stack_frames": max(noise, faintness * 0.85) * (1.0 if deep_sky else 0.55),
        "refocus": focus,
        "steady_mount": max(drift, focus * 0.45),
        "improve_tracking": drift * (1.0 if deep_sky else 0.7),
        "correct_dispersion": dispersion,
        "higher_altitude": max(dispersion, drift) * 0.8,
        "video_capture": max(focus, dispersion) * (1.0 if resolved_body else 0.25),
        "increase_contrast": low_contrast * (1.0 if deep_sky else 0.6),
        "preserve_highlights": overexposed * 0.85,
        "darker_sky": clamp_score((brightness - 95.0) / 100.0) * (1.0 if deep_sky else 0.2),
        "crop_object": clamp_score((0.70 - bright_object) / 0.70) * (1.0 if resolved_body else 0.55),
        "correct_color_cast": color_cast,
        "boost_color": clamp_score((0.18 - saturation) / 0.18) * 0.7,
        "reduce_saturation": clamp_score((saturation - 0.65) / 0.25) * 0.7,
        "denoise": noise,
        "calibrate_frames": max(noise * 0.8, color_cast * 0.6) * (1.0 if deep_sky else 0.45),
        "phase_timing": low_contrast * (1.0 if detected_type == "moon" else 0.0),
    }
    return clamp_score(scores[key])


def build_recommendations(issues, detected_type, summary, limit=5):
    """Rank the image-driven recommendation library and return the best five."""
    ranked = []
    for position, (key, text) in enumerate(RECOMMENDATION_LIBRARY):
        score = recommendation_score(key, issues, detected_type, summary)
        ranked.append((score, position, text))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [text for _, _, text in ranked[:limit]]
