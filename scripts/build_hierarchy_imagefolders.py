from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

DEFAULT_GROUPS = {
    "texture_bumps": ["acne_like_texture", "folliculitis_like_bumps"],
    "redness_irritation": ["dermatitis_like_irritation", "rosacea_like_redness"],
    "review_pigment": ["clinician_review", "hyperpigmentation_like_uneven_tone"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build hierarchical ImageFolder datasets from a flat ImageFolder.")
    parser.add_argument("--source", required=True, help="Flat ImageFolder with train/ and val/ splits.")
    parser.add_argument("--output", required=True, help="Output folder for group and branch datasets.")
    parser.add_argument("--copy", action="store_true", help="Copy instead of hard-linking.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    output = Path(args.output)
    label_to_group = {
        label: group
        for group, labels in DEFAULT_GROUPS.items()
        for label in labels
    }

    if not (source / "train").exists() or not (source / "val").exists():
        raise FileNotFoundError("Source must contain train/ and val/ folders.")

    manifest = {
        "groups": DEFAULT_GROUPS,
        "source": str(source),
        "datasets": {
            "group": "group",
            "branches": {group: f"branches/{group}" for group in DEFAULT_GROUPS},
        },
    }
    (output / "hierarchy.json").parent.mkdir(parents=True, exist_ok=True)
    (output / "hierarchy.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    counts: dict[str, dict[str, int]] = {}
    for split in ["train", "val"]:
        for label_dir in sorted((source / split).iterdir()):
            if not label_dir.is_dir():
                continue
            label = label_dir.name
            group = label_to_group.get(label)
            if group is None:
                print(f"skip unmapped label: {label}")
                continue
            for image_path in sorted(label_dir.iterdir()):
                if not image_path.is_file():
                    continue
                digest = _file_digest(image_path)
                _link_or_copy(
                    image_path,
                    output / "group" / split / group / f"{digest[:16]}{image_path.suffix.lower()}",
                    copy=args.copy,
                )
                _link_or_copy(
                    image_path,
                    output / "branches" / group / split / label / f"{digest[:16]}{image_path.suffix.lower()}",
                    copy=args.copy,
                )
                counts.setdefault(split, {}).setdefault(group, 0)
                counts[split][group] += 1

    for split, split_counts in counts.items():
        rendered = " ".join(f"{name}={count}" for name, count in sorted(split_counts.items()))
        print(f"{split}: {rendered}")
    print(f"wrote {output / 'hierarchy.json'}")


def _link_or_copy(source: Path, target: Path, *, copy: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    if copy:
        shutil.copy2(source, target)
        return
    try:
        target.hardlink_to(source)
    except OSError:
        shutil.copy2(source, target)


def _file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    main()
