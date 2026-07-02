from __future__ import annotations

from PIL import Image

from api.model_adapter import get_onnx_classifier
from api.preprocessing import Region, image_quality


def classify_regions(image: Image.Image, regions: list[Region]) -> dict:
    quality = image_quality(image)
    model = get_onnx_classifier()
    if model is not None:
        return _classify_with_onnx(image, quality, model)
    return _classify_with_heuristics(image, regions, quality)


def _classify_with_onnx(image: Image.Image, quality: dict, model) -> dict:
    predictions = model.predict(image)
    findings = [
        _finding(item.code, item.label, item.score, item.rationale)
        for item in predictions
    ]
    review_flags = []
    if quality["warnings"]:
        review_flags.append("Retake photo in diffuse daylight before trusting the result.")
    if findings and findings[0]["score"] < 0.35:
        review_flags.append("The trained classifier did not find a strong signal.")
    if any(item["code"] == "clinician_review" and item["score"] > 0.5 for item in findings):
        review_flags.append("A clinician-review signal was elevated.")
    return {
        "model": {
            "source": "onnx",
            "name": "skin_classifier.onnx",
            "input_size": 224,
            "calibration": model.calibration,
        },
        "quality": quality,
        "findings": findings,
        "review_flags": review_flags,
    }


def _classify_with_heuristics(image: Image.Image, regions: list[Region], quality: dict) -> dict:
    redness = sum(region.redness_index for region in regions) / max(1, len(regions))
    cheek_redness = _mean_for(["left_cheek", "right_cheek"], regions, "redness_index")
    chin_contrast = _mean_for(["chin"], regions, "contrast")
    unevenness = _regional_spread(regions)

    findings = [
        _finding(
            "rosacea_like_redness",
            "Rosacea-like facial redness",
            min(0.86, cheek_redness * 5.4),
            "Cheek redness is elevated relative to the rest of the face.",
        ),
        _finding(
            "dermatitis_like_irritation",
            "Dermatitis-like irritation",
            min(0.78, (redness * 3.1) + (unevenness * 0.55)),
            "Patchy color variation and redness can be consistent with irritation.",
        ),
        _finding(
            "acne_like_texture",
            "Acne-like texture or inflammatory spots",
            min(0.72, chin_contrast * 1.25 + cheek_redness * 2.0),
            "Localized contrast and redness may indicate inflammatory bumps or spots.",
        ),
        _finding(
            "hyperpigmentation_like_uneven_tone",
            "Uneven tone / hyperpigmentation-like pattern",
            min(0.80, unevenness * 1.55),
            "Regional brightness differences suggest uneven pigmentation or lighting effects.",
        ),
    ]
    findings = sorted(findings, key=lambda item: item["score"], reverse=True)

    review_flags = []
    if quality["warnings"]:
        review_flags.append("Retake photo in diffuse daylight before trusting the result.")
    if max(item["score"] for item in findings) < 0.28:
        review_flags.append("No strong signal from the MVP classifier.")
    if any(item["score"] > 0.68 for item in findings):
        review_flags.append("Consider clinician review if symptoms are painful, spreading, bleeding, infected, or persistent.")

    return {
        "model": {
            "source": "heuristic",
            "name": "facial-region-color-texture-fallback",
            "input_size": None,
            "calibration": {"enabled": False, "profile": "", "alpha": 0.0},
        },
        "quality": quality,
        "findings": findings,
        "review_flags": review_flags,
    }


def _finding(code: str, label: str, score: float, rationale: str) -> dict:
    score = max(0.0, min(1.0, score))
    if score >= 0.66:
        level = "higher"
    elif score >= 0.38:
        level = "moderate"
    else:
        level = "low"
    return {
        "code": code,
        "label": label,
        "score": round(score, 3),
        "level": level,
        "rationale": rationale,
    }


def _mean_for(names: list[str], regions: list[Region], attr: str) -> float:
    selected = [getattr(region, attr) for region in regions if region.name in names]
    return sum(selected) / max(1, len(selected))


def _regional_spread(regions: list[Region]) -> float:
    if not regions:
        return 0.0
    values = [region.brightness for region in regions]
    return max(values) - min(values)
