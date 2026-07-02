from __future__ import annotations

import argparse
import csv
from pathlib import Path


SEVERITY_LABELS = {
    0: "acne_mild",
    1: "acne_moderate",
    2: "acne_severe",
    3: "acne_very_severe",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build train/val CSV manifests from an unpacked ACNE04 dataset."
    )
    parser.add_argument(
        "--raw-dir",
        default="data/raw/acne04",
        help="Folder containing the unpacked official ACNE04 archive.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw/acne04",
        help="Where to write acne04_train_manifest.csv and acne04_val_manifest.csv.",
    )
    parser.add_argument(
        "--fold",
        default="0",
        help="ACNE04 cross-validation fold index, usually 0 through 4.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    voc_root = _find_voc_root(raw_dir)
    image_root = _find_image_root(voc_root)
    split_root = _find_split_root(voc_root)
    train_file = split_root / f"NNEW_trainval_{args.fold}.txt"
    val_file = split_root / f"NNEW_test_{args.fold}.txt"

    if not train_file.exists() or not val_file.exists():
        raise FileNotFoundError(
            "Could not find ACNE04 split files. Expected "
            f"{train_file} and {val_file}. Unpack Classification.tar first."
        )

    train_rows = _read_split(train_file, image_root, split="train")
    val_rows = _read_split(val_file, image_root, split="val")
    train_manifest = output_dir / "acne04_train_manifest.csv"
    val_manifest = output_dir / "acne04_val_manifest.csv"
    combined_manifest = output_dir / "acne04_manifest.csv"
    _write_manifest(train_manifest, train_rows)
    _write_manifest(val_manifest, val_rows)
    _write_manifest(combined_manifest, train_rows + val_rows)

    print(f"image_root={image_root}")
    print(f"wrote {train_manifest} rows={len(train_rows)}")
    print(f"wrote {val_manifest} rows={len(val_rows)}")
    print(f"wrote {combined_manifest} rows={len(train_rows) + len(val_rows)}")
    _print_counts("train", train_rows)
    _print_counts("val", val_rows)


def _find_voc_root(raw_dir: Path) -> Path:
    candidates = [
        raw_dir / "Classification",
        raw_dir / "VOCdevkit2007" / "VOC2007",
        raw_dir / "VOCdevkit" / "VOC2007",
        raw_dir / "VOC2007",
    ]
    for candidate in candidates:
        if (candidate / "ImageSets" / "Main").exists() or any(candidate.glob("NNEW_trainval_*.txt")):
            return candidate
    matches = [
        path
        for path in raw_dir.rglob("VOC2007")
        if (path / "ImageSets" / "Main").exists()
    ]
    if matches:
        return matches[0]
    matches = [
        path
        for path in raw_dir.rglob("Classification")
        if any(path.glob("NNEW_trainval_*.txt"))
    ]
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find an ACNE04 classification folder under {raw_dir}.")


def _find_image_root(voc_root: Path) -> Path:
    for name in ["JPEGImages_300", "JPEGImages"]:
        candidate = voc_root / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find JPEGImages_300 or JPEGImages under {voc_root}.")


def _find_split_root(voc_root: Path) -> Path:
    candidate = voc_root / "ImageSets" / "Main"
    if any(candidate.glob("NNEW_trainval_*.txt")):
        return candidate
    if any(voc_root.glob("NNEW_trainval_*.txt")):
        return voc_root
    raise FileNotFoundError(f"Could not find ACNE04 NNEW split files under {voc_root}.")


def _read_split(split_file: Path, image_root: Path, *, split: str) -> list[dict[str, str]]:
    rows = []
    with split_file.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.split()
            if len(parts) != 3:
                raise ValueError(f"{split_file}:{line_number} should have filename label lesion_count.")
            filename, severity, lesion_count = parts
            severity_id = int(severity)
            label = SEVERITY_LABELS.get(severity_id)
            if label is None:
                raise ValueError(f"{split_file}:{line_number} has unknown severity {severity}.")
            image_path = (image_root / filename).resolve()
            if not image_path.exists():
                print(f"missing: {image_path}")
                continue
            rows.append(
                {
                    "image_path": str(image_path),
                    "filename": filename,
                    "label": label,
                    "severity_id": str(severity_id),
                    "lesion_count": str(int(lesion_count)),
                    "split": split,
                }
            )
    if not rows:
        raise ValueError(f"No usable rows found in {split_file}.")
    return rows


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["image_path", "filename", "label", "severity_id", "lesion_count", "split"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_counts(name: str, rows: list[dict[str, str]]) -> None:
    counts = {label: 0 for label in SEVERITY_LABELS.values()}
    for row in rows:
        counts[row["label"]] += 1
    rendered = " ".join(f"{label}={count}" for label, count in counts.items())
    print(f"{name}: {rendered}")


if __name__ == "__main__":
    main()
