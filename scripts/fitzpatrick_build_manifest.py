from __future__ import annotations

import argparse
import csv
import random
import urllib.request
from pathlib import Path

CSV_URL = "https://raw.githubusercontent.com/mattgroh/fitzpatrick17k/main/fitzpatrick17k.csv"
TARGET_LABELS = {
    "acne_like_texture",
    "dermatitis_like_irritation",
    "folliculitis_like_bumps",
    "hyperpigmentation_like_uneven_tone",
    "clinician_review",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a mapped Fitzpatrick17k manifest for our skin classes.")
    parser.add_argument("--raw-dir", default="data/raw/fitzpatrick17k")
    parser.add_argument("--output", default="data/raw/fitzpatrick17k/manifest.csv")
    parser.add_argument("--download-metadata", action="store_true")
    parser.add_argument("--download-images", action="store_true")
    parser.add_argument("--max-per-label", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_dir / "fitzpatrick17k.csv"
    if args.download_metadata and not csv_path.exists():
        _download(CSV_URL, csv_path)
    if not csv_path.exists():
        fallback = Path("data/raw/fitzpatrick17k.csv")
        if fallback.exists():
            csv_path = fallback
        else:
            raise FileNotFoundError("Missing Fitzpatrick17k CSV. Run with --download-metadata.")

    rows = _build_rows(csv_path)
    rows = _cap_per_label(rows, args.max_per_label, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_manifest(rows, output)

    if args.download_images:
        for row in rows:
            target = raw_dir / row["image_path"]
            if target.exists():
                continue
            try:
                _download(row["source_url"], target)
            except Exception as exc:
                print(f"failed: {row['source_url']} ({exc})")

    counts = {}
    for row in rows:
        counts[row["label"]] = counts.get(row["label"], 0) + 1
    print(f"wrote {output}")
    for label in sorted(counts):
        print(f"{label}: {counts[label]}")


def _build_rows(csv_path: Path) -> list[dict]:
    rows = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            target = _map_label(row["label"], row["nine_partition_label"])
            if not target:
                continue
            image_name = f"images/{row['md5hash']}.jpg"
            rows.append(
                {
                    "image_path": image_name,
                    "label": target,
                    "source_label": row["label"],
                    "nine_partition_label": row["nine_partition_label"],
                    "fitzpatrick_scale": row["fitzpatrick_scale"],
                    "source_url": row["url"],
                }
            )
    return rows


def _map_label(label: str, partition: str) -> str | None:
    text = label.lower()
    if "acne" in text:
        return "acne_like_texture"
    if "folliculitis" in text:
        return "folliculitis_like_bumps"
    if any(term in text for term in ["eczema", "dermatitis", "psoriasis", "drug eruption", "urticaria"]):
        return "dermatitis_like_irritation"
    if any(term in text for term in ["pigment", "melasma", "vitiligo"]):
        return "hyperpigmentation_like_uneven_tone"
    if partition.startswith("malignant") or any(
        term in text for term in ["melanoma", "carcinoma", "sarcoma", "lymphoma"]
    ):
        return "clinician_review"
    return None


def _cap_per_label(rows: list[dict], max_per_label: int, seed: int) -> list[dict]:
    if max_per_label <= 0:
        return rows
    rng = random.Random(seed)
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["label"], []).append(row)
    capped = []
    for items in grouped.values():
        rng.shuffle(items)
        capped.extend(items[:max_per_label])
    rng.shuffle(capped)
    return capped


def _write_manifest(rows: list[dict], output: Path) -> None:
    fieldnames = [
        "image_path",
        "label",
        "source_label",
        "nine_partition_label",
        "fitzpatrick_scale",
        "source_url",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 skin-private-ai portfolio research"},
    )
    print(f"download {url}")
    with urllib.request.urlopen(request, timeout=30) as response:
        destination.write_bytes(response.read())


if __name__ == "__main__":
    main()
