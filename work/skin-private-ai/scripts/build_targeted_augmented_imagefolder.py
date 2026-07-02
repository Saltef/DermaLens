from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


DEFAULT_TARGETS = {
    "clinician_review": 160,
    "folliculitis_like_bumps": 180,
    "hyperpigmentation_like_uneven_tone": 160,
    "rosacea_like_redness": 80,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a face-validation ImageFolder with targeted minority-class train augmentation."
    )
    parser.add_argument("--base", required=True, help="Primary ImageFolder with train/ and val/.")
    parser.add_argument("--augment", required=True, help="Auxiliary ImageFolder used only for train augmentation.")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--targets",
        default=json.dumps(DEFAULT_TARGETS),
        help="JSON object mapping label to desired train count after augmentation.",
    )
    parser.add_argument("--copy", action="store_true")
    parser.add_argument(
        "--preserve-val",
        action="store_true",
        help="Keep the full base validation split while still preventing train augmentation from duplicating base images.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = Path(args.base)
    augment = Path(args.augment)
    output = Path(args.output)
    targets = json.loads(args.targets)
    seen: set[str] = set()

    if not (base / "train").exists() or not (base / "val").exists():
        raise FileNotFoundError("Base dataset must include train/ and val/.")
    if not (augment / "train").exists():
        raise FileNotFoundError("Augment dataset must include train/.")

    counts: dict[str, dict[str, int]] = {"train": {}, "val": {}}
    if args.preserve_val:
        train_seen: set[str] = set()
        val_seen: set[str] = set()
        _copy_split(base / "train", output / "train", counts["train"], train_seen, copy=args.copy)
        _copy_split(base / "val", output / "val", counts["val"], val_seen, copy=args.copy)
        seen = train_seen | val_seen
    else:
        _copy_split(base / "train", output / "train", counts["train"], seen, copy=args.copy)
        _copy_split(base / "val", output / "val", counts["val"], seen, copy=args.copy)

    for label, target_count in targets.items():
        current = counts["train"].get(label, 0)
        needed = max(0, int(target_count) - current)
        if needed <= 0:
            continue
        source_dir = augment / "train" / label
        if not source_dir.exists():
            print(f"missing augmentation label: {label}")
            continue
        added = _copy_limited(
            source_dir,
            output / "train" / label,
            needed,
            seen,
            copy=args.copy,
        )
        counts["train"][label] = counts["train"].get(label, 0) + added
        print(f"augment {label}: requested={needed} added={added}")

    manifest = {
        "base": str(base),
        "augment": str(augment),
        "targets": targets,
        "counts": counts,
        "note": "Validation is copied only from the base dataset; augmentation is train-only.",
        "preserve_val": args.preserve_val,
    }
    (output / "augmentation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for split in ["train", "val"]:
        for label, count in sorted(counts[split].items()):
            print(f"{split}/{label}: {count}")
    print(f"wrote {output / 'augmentation_manifest.json'}")


def _copy_split(source_split: Path, output_split: Path, counts: dict[str, int], seen: set[str], *, copy: bool) -> None:
    for label_dir in sorted(source_split.iterdir()):
        if not label_dir.is_dir():
            continue
        added = _copy_limited(label_dir, output_split / label_dir.name, limit=None, seen=seen, copy=copy)
        counts[label_dir.name] = counts.get(label_dir.name, 0) + added


def _copy_limited(source_dir: Path, target_dir: Path, limit: int | None, seen: set[str], *, copy: bool) -> int:
    target_dir.mkdir(parents=True, exist_ok=True)
    added = 0
    for source in sorted(source_dir.iterdir()):
        if not source.is_file():
            continue
        digest = _digest(source)
        if digest in seen:
            continue
        seen.add(digest)
        target = target_dir / f"{digest[:16]}{source.suffix.lower()}"
        if not target.exists():
            _link_or_copy(source, target, copy=copy)
            added += 1
        if limit is not None and added >= limit:
            break
    return added


def _link_or_copy(source: Path, target: Path, *, copy: bool) -> None:
    if copy:
        shutil.copy2(source, target)
        return
    try:
        target.hardlink_to(source)
    except OSError:
        shutil.copy2(source, target)


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    main()
