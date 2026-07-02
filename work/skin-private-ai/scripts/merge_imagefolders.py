from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge multiple ImageFolder datasets into one train/val tree.")
    parser.add_argument("--input", action="append", required=True, help="Input dataset root with train/ and val/.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--copy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    counts: dict[str, int] = {}
    for input_root in [Path(item) for item in args.input]:
        for split in ["train", "val"]:
            split_root = input_root / split
            if not split_root.exists():
                continue
            for class_dir in split_root.iterdir():
                if not class_dir.is_dir():
                    continue
                target_dir = output / split / class_dir.name
                target_dir.mkdir(parents=True, exist_ok=True)
                for source in class_dir.iterdir():
                    if not source.is_file():
                        continue
                    digest = _digest(source)
                    target = target_dir / f"{input_root.name}-{digest[:16]}{source.suffix.lower()}"
                    if target.exists():
                        continue
                    if args.copy:
                        shutil.copy2(source, target)
                    else:
                        try:
                            target.hardlink_to(source)
                        except OSError:
                            shutil.copy2(source, target)
                    key = f"{split}/{class_dir.name}"
                    counts[key] = counts.get(key, 0) + 1
    for key in sorted(counts):
        print(f"{key}: {counts[key]}")


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    main()
