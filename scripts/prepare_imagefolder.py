from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a CSV manifest into train/val ImageFolder layout.")
    parser.add_argument("--manifest", required=True, help="CSV with image path and label columns.")
    parser.add_argument("--image-root", required=True, help="Root folder for relative image paths.")
    parser.add_argument("--output", required=True, help="Output folder containing train/ and val/.")
    parser.add_argument("--path-column", default="image_path")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--group-column", default="case_id", help="Case/patient column used to prevent leakage.")
    parser.add_argument("--split-column", default=None, help="Optional column with train/val values.")
    parser.add_argument(
        "--allow-image-level-split",
        action="store_true",
        help="Permit row-level fallback when no group column exists. Avoid this for medical image datasets.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy", action="store_true", help="Copy files instead of hard-linking them.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = Path(args.manifest)
    image_root = Path(args.image_root)
    output = Path(args.output)
    rows = _read_rows(manifest, args.path_column, args.label_column, args.split_column, args.group_column)
    rng = random.Random(args.seed)
    split_audit: dict[str, dict] = {}

    if args.split_column:
        seen_digests: set[str] = set()
        _assert_no_group_overlap(rows)
        for split in ["train", "val"]:
            grouped = _group_by_label([row for row in rows if row.get("split") == split])
            for label, items in grouped.items():
                written = _write_split(items, image_root, output / split / label, copy=args.copy, seen_digests=seen_digests)
                print(f"{label}: {split}={written}")
        split_audit = _split_audit(rows)
        _write_split_audit(output, split_audit, args)
        return

    grouped = _group_by_label(rows)
    for label, items in grouped.items():
        train_items, val_items = _grouped_train_val_split(
            items,
            val_ratio=args.val_ratio,
            rng=rng,
            allow_image_level_split=args.allow_image_level_split,
        )
        for row in train_items:
            row["split"] = "train"
        for row in val_items:
            row["split"] = "val"
        seen_digests: set[str] = set()
        train_written = _write_split(train_items, image_root, output / "train" / label, copy=args.copy, seen_digests=seen_digests)
        val_written = _write_split(val_items, image_root, output / "val" / label, copy=args.copy, seen_digests=seen_digests)
        print(f"{label}: train={train_written} val={val_written}")
    _assert_no_group_overlap(rows)
    split_audit = _split_audit(rows)
    _write_split_audit(output, split_audit, args)


def _read_rows(
    manifest: Path,
    path_column: str,
    label_column: str,
    split_column: str | None,
    group_column: str,
) -> list[dict]:
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if path_column not in reader.fieldnames or label_column not in reader.fieldnames:
            raise ValueError(f"Manifest must include {path_column!r} and {label_column!r}.")
        if split_column and split_column not in reader.fieldnames:
            raise ValueError(f"Manifest must include split column {split_column!r}.")
        has_group_column = group_column in (reader.fieldnames or [])
        rows = []
        for row in reader:
            image_path = row[path_column].strip()
            label = _normalize_label(row[label_column])
            if image_path and label:
                group_id = row.get(group_column, "").strip() if has_group_column else ""
                item = {"image_path": image_path, "label": label, "group_id": group_id}
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


def _grouped_train_val_split(
    items: list[dict],
    *,
    val_ratio: float,
    rng: random.Random,
    allow_image_level_split: bool,
) -> tuple[list[dict], list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    missing_group_count = 0
    for row in items:
        group_id = row.get("group_id", "").strip()
        if not group_id:
            missing_group_count += 1
            if not allow_image_level_split:
                raise ValueError(
                    "Manifest is missing group IDs for at least one row. "
                    "Pass --group-column with a case/patient ID column, or --allow-image-level-split for non-medical datasets."
                )
            group_id = f"image::{row['image_path']}"
            row["group_id"] = group_id
        groups[group_id].append(row)

    group_ids = list(groups)
    rng.shuffle(group_ids)
    if len(group_ids) < 2:
        raise ValueError(f"Need at least two unique groups for label {items[0]['label']!r}.")

    val_group_count = max(1, round(len(group_ids) * val_ratio))
    if val_group_count >= len(group_ids):
        val_group_count = len(group_ids) - 1
    val_group_ids = set(group_ids[:val_group_count])
    train_items = [row for group_id in group_ids if group_id not in val_group_ids for row in groups[group_id]]
    val_items = [row for group_id in group_ids if group_id in val_group_ids for row in groups[group_id]]
    if missing_group_count:
        print(f"warning: {missing_group_count} rows used image path as fallback group id")
    return train_items, val_items


def _assert_no_group_overlap(rows: list[dict]) -> None:
    train_groups = {row["group_id"] for row in rows if row.get("split") == "train" and row.get("group_id")}
    val_groups = {row["group_id"] for row in rows if row.get("split") == "val" and row.get("group_id")}
    overlap = train_groups & val_groups
    if overlap:
        sample = ", ".join(sorted(overlap)[:5])
        raise ValueError(f"Group leakage detected: {len(overlap)} case/patient IDs appear in train and val. Examples: {sample}")


def _split_audit(rows: list[dict]) -> dict:
    audit: dict[str, dict] = {
        "leakage_check": {
            "group_overlap_count": 0,
            "passed": True,
        },
        "splits": {},
    }
    for split in ["train", "val"]:
        split_rows = [row for row in rows if row.get("split") == split]
        by_label: dict[str, dict] = {}
        for row in split_rows:
            label = row["label"]
            item = by_label.setdefault(label, {"images": 0, "groups": set()})
            item["images"] += 1
            if row.get("group_id"):
                item["groups"].add(row["group_id"])
        audit["splits"][split] = {
            "images": len(split_rows),
            "groups": len({row["group_id"] for row in split_rows if row.get("group_id")}),
            "labels": {
                label: {"images": values["images"], "groups": len(values["groups"])}
                for label, values in sorted(by_label.items())
            },
        }
    return audit


def _write_split_audit(output: Path, audit: dict, args: argparse.Namespace) -> None:
    audit["config"] = {
        "manifest": args.manifest,
        "path_column": args.path_column,
        "label_column": args.label_column,
        "group_column": args.group_column,
        "split_column": args.split_column,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "allow_image_level_split": args.allow_image_level_split,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "split_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"wrote {output / 'split_audit.json'}")


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
