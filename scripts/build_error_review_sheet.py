from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an HTML contact sheet and editable CSV for model error review.")
    parser.add_argument("--errors-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--title", default="Skin Model Error Review")
    parser.add_argument("--thumb-size", type=int, default=220)
    parser.add_argument("--copy-originals", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    assets_dir = output_dir / "thumbs"
    originals_dir = output_dir / "originals"
    assets_dir.mkdir(parents=True, exist_ok=True)
    if args.copy_originals:
        originals_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_rows(Path(args.errors_csv))
    enriched = []
    for idx, row in enumerate(rows, start=1):
        source = Path(row["path"])
        thumb_path = assets_dir / f"{idx:04d}_{source.stem}.jpg"
        _make_thumb(source, thumb_path, args.thumb_size)
        original_rel = ""
        if args.copy_originals:
            original_path = originals_dir / f"{idx:04d}_{source.name}"
            if not original_path.exists():
                shutil.copy2(source, original_path)
            original_rel = original_path.relative_to(output_dir).as_posix()
        enriched.append(
            {
                **row,
                "id": idx,
                "thumb": thumb_path.relative_to(output_dir).as_posix(),
                "original": original_rel,
                "review_action": "",
                "new_label": "",
                "review_notes": "",
            }
        )

    _write_review_csv(output_dir / "review_queue.csv", enriched)
    _write_summary(output_dir / "summary.json", enriched)
    _write_html(output_dir / "index.html", enriched, args.title)
    print(f"wrote {output_dir / 'index.html'}")
    print(f"wrote {output_dir / 'review_queue.csv'}")
    print(f"wrote {output_dir / 'summary.json'}")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _make_thumb(source: Path, target: Path, size: int) -> None:
    if target.exists():
        return
    with Image.open(source) as image:
        image = image.convert("RGB")
        image.thumbnail((size, size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (size, size), (245, 246, 248))
        x = (size - image.width) // 2
        y = (size - image.height) // 2
        canvas.paste(image, (x, y))
        canvas.save(target, quality=88)


def _write_review_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "id",
        "path",
        "actual",
        "predicted",
        "confidence",
        "second_label",
        "second_confidence",
        "review_action",
        "new_label",
        "review_notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    pair_counts = Counter((row["actual"], row["predicted"]) for row in rows)
    actual_counts = Counter(row["actual"] for row in rows)
    predicted_counts = Counter(row["predicted"] for row in rows)
    payload = {
        "total_errors": len(rows),
        "by_actual": dict(sorted(actual_counts.items())),
        "by_predicted": dict(sorted(predicted_counts.items())),
        "actual_predicted_pairs": [
            {"actual": actual, "predicted": predicted, "count": count}
            for (actual, predicted), count in pair_counts.most_common()
        ],
        "review_actions": [
            "keep_label",
            "relabel",
            "exclude_ambiguous",
            "exclude_low_quality",
            "exclude_non_face_or_ood",
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_html(path: Path, rows: list[dict[str, str]], title: str) -> None:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["actual"], row["predicted"])].append(row)
    pair_counts = Counter((row["actual"], row["predicted"]) for row in rows)
    actual_counts = Counter(row["actual"] for row in rows)

    cards = []
    for (actual, predicted), group_rows in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0][0], item[0][1])):
        cards.append(
            f"<section class='group'><h2>{_esc(actual)} -> {_esc(predicted)} "
            f"<span>{len(group_rows)} cases</span></h2><div class='grid'>"
        )
        for row in group_rows:
            cards.append(_card(row))
        cards.append("</div></section>")

    summary_rows = "\n".join(
        f"<tr><td>{_esc(label)}</td><td>{count}</td></tr>"
        for label, count in sorted(actual_counts.items())
    )
    pair_rows = "\n".join(
        f"<tr><td>{_esc(actual)}</td><td>{_esc(predicted)}</td><td>{count}</td></tr>"
        for (actual, predicted), count in pair_counts.most_common()
    )

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{_esc(title)}</title>
  <style>
    :root {{ color-scheme: light; font-family: Inter, Segoe UI, Arial, sans-serif; }}
    body {{ margin: 0; background: #f7f8fa; color: #1c2430; }}
    header {{ padding: 28px 32px 18px; background: #ffffff; border-bottom: 1px solid #dde2ea; }}
    h1 {{ margin: 0 0 10px; font-size: 24px; letter-spacing: 0; }}
    p {{ margin: 0; color: #526071; line-height: 1.45; }}
    main {{ padding: 22px 32px 40px; }}
    .summary {{ display: grid; grid-template-columns: minmax(220px, 360px) minmax(320px, 1fr); gap: 18px; margin-bottom: 22px; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #dde2ea; }}
    th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid #edf0f4; font-size: 13px; }}
    th {{ background: #eef2f6; color: #2d3848; }}
    .group {{ margin: 0 0 28px; }}
    .group h2 {{ margin: 0 0 12px; font-size: 18px; letter-spacing: 0; }}
    .group h2 span {{ color: #647287; font-size: 13px; font-weight: 500; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 14px; }}
    .card {{ background: #fff; border: 1px solid #dbe1e8; border-radius: 8px; overflow: hidden; }}
    .card img {{ width: 100%; aspect-ratio: 1 / 1; object-fit: contain; background: #f1f3f6; display: block; }}
    .meta {{ padding: 11px 12px 12px; }}
    .id {{ font-size: 12px; color: #6b7788; margin-bottom: 7px; }}
    .line {{ font-size: 13px; margin: 4px 0; overflow-wrap: anywhere; }}
    .actual {{ color: #2f5f9f; font-weight: 650; }}
    .pred {{ color: #a63d40; font-weight: 650; }}
    .choices {{ margin-top: 10px; padding-top: 9px; border-top: 1px solid #edf0f4; color: #4d5b6d; }}
    code {{ font-size: 12px; color: #344256; }}
    @media (max-width: 760px) {{ header, main {{ padding-left: 16px; padding-right: 16px; }} .summary {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>{_esc(title)}</h1>
    <p>{len(rows)} misclassified images. Use <code>review_queue.csv</code> to mark keep, relabel, ambiguous, low-quality, or non-face/OOD decisions.</p>
  </header>
  <main>
    <div class="summary">
      <table><thead><tr><th>Actual Label</th><th>Error Count</th></tr></thead><tbody>{summary_rows}</tbody></table>
      <table><thead><tr><th>Actual</th><th>Predicted</th><th>Count</th></tr></thead><tbody>{pair_rows}</tbody></table>
    </div>
    {''.join(cards)}
  </main>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def _card(row: dict[str, str]) -> str:
    original_link = ""
    if row.get("original"):
        original_link = f"<div class='line'><a href='{_esc(row['original'])}'>open original</a></div>"
    return f"""
<article class="card">
  <img src="{_esc(row['thumb'])}" alt="review image {row['id']}" />
  <div class="meta">
    <div class="id">#{row['id']}</div>
    <div class="line">Actual: <span class="actual">{_esc(row['actual'])}</span></div>
    <div class="line">Predicted: <span class="pred">{_esc(row['predicted'])}</span> ({_esc(row['confidence'])})</div>
    <div class="line">Second: {_esc(row['second_label'])} ({_esc(row['second_confidence'])})</div>
    <div class="choices">
      <div class="line">Review: keep_label | relabel | exclude_ambiguous | exclude_low_quality | exclude_non_face_or_ood</div>
      <div class="line"><code>{_esc(row['path'])}</code></div>
      {original_link}
    </div>
  </div>
</article>
"""


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


if __name__ == "__main__":
    main()
