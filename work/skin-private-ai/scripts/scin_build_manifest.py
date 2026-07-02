from __future__ import annotations

import argparse
import ast
import csv
import random
import urllib.request
from pathlib import Path


SCIN_BASE_URL = "https://storage.googleapis.com/dx-scin-public-data"
IMAGE_COLUMNS = ["image_1_path", "image_2_path", "image_3_path"]
TARGET_LABELS = {
    "acne_like_texture",
    "rosacea_like_redness",
    "dermatitis_like_irritation",
    "hyperpigmentation_like_uneven_tone",
    "folliculitis_like_bumps",
    "clinician_review",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a face-skin training manifest from the SCIN dataset.")
    parser.add_argument("--raw-dir", default="data/raw/scin", help="SCIN raw folder.")
    parser.add_argument("--output", default="data/raw/scin/face_skin_manifest.csv")
    parser.add_argument("--head-neck-only", action="store_true", default=True)
    parser.add_argument("--include-all-body-sites", action="store_false", dest="head_neck_only")
    parser.add_argument("--download-metadata", action="store_true")
    parser.add_argument("--download-images", action="store_true")
    parser.add_argument("--max-per-label", type=int, default=0, help="0 means unlimited.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    dataset_dir = raw_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if args.download_metadata:
        _download_if_missing("dataset/scin_cases.csv", dataset_dir / "scin_cases.csv")
        _download_if_missing("dataset/scin_labels.csv", dataset_dir / "scin_labels.csv")

    cases_path = dataset_dir / "scin_cases.csv"
    labels_path = dataset_dir / "scin_labels.csv"
    if not cases_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            "Missing SCIN CSV files. Run with --download-metadata or place scin_cases.csv and scin_labels.csv under data/raw/scin/dataset/."
        )

    cases = _read_by_case_id(cases_path)
    labels = _read_by_case_id(labels_path)
    rows = _build_rows(cases, labels, head_neck_only=args.head_neck_only)
    rows = _cap_per_label(rows, args.max_per_label, args.seed)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_manifest(rows, output)

    if args.download_images:
        for row in rows:
            source_path = row["image_path"]
            _download_if_missing(source_path, raw_dir / source_path)

    counts = {}
    for row in rows:
        counts[row["label"]] = counts.get(row["label"], 0) + 1
    print(f"wrote {output}")
    for label in sorted(counts):
        print(f"{label}: {counts[label]}")


def _read_by_case_id(path: Path) -> dict[str, dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return {row["case_id"]: row for row in reader}


def _build_rows(cases: dict[str, dict], labels: dict[str, dict], *, head_neck_only: bool) -> list[dict]:
    rows = []
    for case_id, case in cases.items():
        label_row = labels.get(case_id, {})
        target_label = _map_case_to_target(case, label_row)
        if not target_label:
            continue
        if head_neck_only and _truthy(case.get("body_parts_head_or_neck")) is False:
            continue
        for image_column in IMAGE_COLUMNS:
            image_path = case.get(image_column, "").strip()
            if not image_path:
                continue
            rows.append(
                {
                    "case_id": case_id,
                    "image_path": image_path,
                    "label": target_label,
                    "related_category": case.get("related_category", ""),
                    "fitzpatrick_skin_type": case.get("fitzpatrick_skin_type", ""),
                    "monk_skin_tone_us": label_row.get("monk_skin_tone_label_us", ""),
                    "monk_skin_tone_india": label_row.get("monk_skin_tone_label_india", ""),
                }
            )
    return rows


def _map_case_to_target(case: dict, labels: dict) -> str | None:
    related = case.get("related_category", "").upper()
    weighted = _weighted_label_dict(labels.get("weighted_skin_condition_label", ""))
    label_text = " ".join(weighted.keys()).lower()

    if related == "ACNE" or "acne" in label_text:
        return "acne_like_texture"
    if "rosacea" in label_text:
        return "rosacea_like_redness"
    if any(term in label_text for term in ["eczema", "dermatitis", "rash", "psoriasis"]):
        return "dermatitis_like_irritation"
    if related == "PIGMENTARY_PROBLEM" or any(
        term in label_text for term in ["hyperpigmentation", "melasma", "pigment", "vitiligo"]
    ):
        return "hyperpigmentation_like_uneven_tone"
    if "folliculitis" in label_text:
        return "folliculitis_like_bumps"
    if related == "GROWTH_OR_MOLE" or any(term in label_text for term in ["melanoma", "carcinoma", "nevus"]):
        return "clinician_review"
    return None


def _weighted_label_dict(value: str) -> dict:
    if not value or value.lower() == "nan":
        return {}
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    return parsed


def _cap_per_label(rows: list[dict], max_per_label: int, seed: int) -> list[dict]:
    if max_per_label <= 0:
        return rows
    rng = random.Random(seed)
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["label"], []).append(row)
    capped = []
    for label_rows in grouped.values():
        rng.shuffle(label_rows)
        capped.extend(label_rows[:max_per_label])
    rng.shuffle(capped)
    return capped


def _write_manifest(rows: list[dict], output: Path) -> None:
    fieldnames = [
        "case_id",
        "image_path",
        "label",
        "related_category",
        "fitzpatrick_skin_type",
        "monk_skin_tone_us",
        "monk_skin_tone_india",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _download_if_missing(source_path: str, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"{SCIN_BASE_URL}/{source_path}"
    print(f"download {url}")
    urllib.request.urlretrieve(url, destination)


def _truthy(value: str | None) -> bool:
    return str(value).strip().upper() in {"1", "TRUE", "YES", "Y"}


if __name__ == "__main__":
    main()
