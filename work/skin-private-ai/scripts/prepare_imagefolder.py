from __future__ import annotations

import argparse
import csv
import hashlib
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a CSV manifest into train/val ImageFolder layout.")
    parser.add_argument("--manifest", required=True, help="CSV with image path and label columns.")
    parser.add_argument("--image-root", required=True, help="Root folder for relative image paths.")
    parser.add_argument("--output", required=True, help="Output folder containing train/ and val/.")
    parser.add_argument("--path-column", default="image_path")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--split-column", default=None, help="Optional column with train/val values.")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy", action="store_true", help="Copy files instead of hard-linking them.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = Path(args.manifest)
    image_root = Path(args.image_root)
    output = Path(args.output)
    rows = _read_rows(manifest, args.path_column, args.label_column, args.split_column)
    random.seed(args.seed)

    if args.split_column:
        seen_digests: set[str] = set()
        for split in ["train", "val"]:
            grouped = _group_by_label([row for row in rows if row.get("split") == split])
            for label, items in grouped.items():
                written = _write_split(items, image_root, output / split / label, copy=args.copy, seen_digests=seen_digests)
                print(f"{label}: {split}={written}")
        return

    grouped = _group_by_label(rows)
    for label, items in grouped.items():
        random.shuffle(items)
        split_at = max(1, round(len(items) * (1 - args.val_ratio)))
        train_items = items[:split_at]
        val_items = items[split_at:]
        train_written = _write_split(train_items, image_root, output / "train" / label, copy=args.copy)
        val_written = _write_split(val_items, image_root, output / "val" / label, copy=args.copy)
        print(f"{label}: train={train_written} val={val_written}")


def _read_rows(manifest: Path, path_column: str, label_column: str, split_column: str | None) -> list[dict]:
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if path_column not in reader.fieldnames or label_column not in reader.fieldnames:
            raise ValueError(f"Manifest must include {path_column!r} and {label_column!r}.")
        if split_column and split_column not in reader.fieldnames:
            raise ValueError(f"Manifest must include split column {split_column!r}.")
        rows = []
        for row in reader:
            image_path = row[path_column].strip()
            label = _normalize_label(row[label_column])
            if image_path and label:
                item = {"image_path": image_path, "label": label}
                if split_column:
                    split = row[split_column].strip().lower()
                    if split not in {"train", "val"}:
                        raise ValueError(f"Unsupported split {split!r}; expected train or val.")
                    item["split"] = split
                rows.append(item)
    if not rows:
        raise ValueError("No usable rows found in manifest.")
    return rows


def _group_by_label(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["label"], []).append(row)
    return grouped


def _write_split(
    items: list[dict],
    image_root: Path,
    destination: Path,
    *,
    copy: bool,
    seen_digests: set[str] | None = None,
) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    seen: set[str] = seen_digests if seen_digests is not None else set()
    written = 0
    for item in items:
        source = image_root / item["image_path"]
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if not source.exists():
            print(f"missing: {source}")
            continue
        digest = _file_digest(source)
        if digest in seen:
            continue
        seen.add(digest)
        target = destination / f"{digest[:16]}{source.suffix.lower()}"
        if target.exists():
            continue
        if copy:
            shutil.copy2(source, target)
        else:
            try:
                target.hardlink_to(source)
            except OSError:
                shutil.copy2(source, target)
        written += 1
    return written


def _file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalize_label(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")


if __name__ == "__main__":
    main()
