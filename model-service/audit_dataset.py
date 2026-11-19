"""Audit the processed apple-leaf dataset for unreadable files and leakage risks."""

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

SERVICE_DIR = Path(__file__).resolve().parent
CLASSES_PATH = SERVICE_DIR / "classes.json"
DEFAULT_EXCLUSIONS = SERVICE_DIR / "dataset_exclusions.json"
SPLITS = ("train", "val", "test")


def source_id(path: Path) -> str:
    stem = re.sub(r"_[0-2]$", "", path.stem, flags=re.IGNORECASE)
    return stem.casefold()


def load_exclusions(path: Path | None) -> set[str]:
    if path is None:
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(value).casefold() for value in payload.get("source_ids", [])}


def inspect_image(path: Path) -> tuple[tuple[int, int], str]:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        digest = hashlib.sha256(
            image.width.to_bytes(4, "big")
            + image.height.to_bytes(4, "big")
            + image.tobytes()
        ).hexdigest()
        return image.size, digest


def audit_dataset(data_root: Path, classes: list[str], excluded: set[str]) -> dict:
    records: list[dict] = []
    unreadable: list[dict] = []
    sizes: dict[str, int] = defaultdict(int)

    for split in SPLITS:
        split_root = data_root / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"Dataset split not found: {split_root}")
        found_classes = sorted(path.name for path in split_root.iterdir() if path.is_dir())
        if found_classes != sorted(classes):
            raise ValueError(
                f"Class folders for {split!r} do not match {classes}: {found_classes}"
            )
        for class_name in classes:
            for path in sorted((split_root / class_name).iterdir()):
                if not path.is_file():
                    continue
                record = {
                    "split": split,
                    "class": class_name,
                    "path": str(path),
                    "source_id": source_id(path),
                    "excluded": source_id(path) in excluded,
                }
                try:
                    size, digest = inspect_image(path)
                except (UnidentifiedImageError, OSError, ValueError) as exc:
                    unreadable.append({"path": str(path), "error": str(exc)})
                    continue
                record["size"] = list(size)
                record["pixel_sha256"] = digest
                records.append(record)
                sizes[f"{size[0]}x{size[1]}"] += 1

    by_source: dict[str, list[dict]] = defaultdict(list)
    by_pixels: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_source[record["source_id"]].append(record)
        by_pixels[record["pixel_sha256"]].append(record)

    source_conflicts = [
        {
            "source_id": source,
            "splits": sorted({row["split"] for row in rows}),
            "classes": sorted({row["class"] for row in rows}),
            "files": [
                {"split": row["split"], "class": row["class"], "path": row["path"]}
                for row in rows
            ],
        }
        for source, rows in sorted(by_source.items())
        if len({row["split"] for row in rows}) > 1
        or len({row["class"] for row in rows}) > 1
    ]
    exact_duplicates = [
        {
            "pixel_sha256": digest,
            "splits": sorted({row["split"] for row in rows}),
            "classes": sorted({row["class"] for row in rows}),
            "files": [
                {"split": row["split"], "class": row["class"], "path": row["path"]}
                for row in rows
            ],
        }
        for digest, rows in by_pixels.items()
        if len(rows) > 1
    ]

    clean_counts: dict[str, dict[str, int]] = {}
    clean_sources: dict[str, set[str]] = {split: set() for split in SPLITS}
    for split in SPLITS:
        clean_counts[split] = {}
        for class_name in classes:
            eligible = [
                row for row in records
                if row["split"] == split
                and row["class"] == class_name
                and not row["excluded"]
                and (split != "train" or row["path"].casefold().endswith("_0.jpg"))
            ]
            clean_counts[split][class_name] = len(eligible)
            clean_sources[split].update(row["source_id"] for row in eligible)

    source_overlap_after_exclusions = {
        f"{left}/{right}": sorted(clean_sources[left] & clean_sources[right])
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1:]
    }

    return {
        "data_root": str(data_root),
        "classes": classes,
        "file_count": len(records) + len(unreadable),
        "image_sizes": dict(sorted(sizes.items())),
        "unreadable_images": unreadable,
        "class_counts_including_augmented_copies": {
            split: {
                class_name: sum(
                    row["split"] == split and row["class"] == class_name
                    for row in records
                )
                for class_name in classes
            }
            for split in SPLITS
        },
        "clean_source_image_counts_after_exclusions": clean_counts,
        "source_id_conflicts": source_conflicts,
        "exact_pixel_duplicate_groups": exact_duplicates,
        "excluded_source_ids": sorted(excluded),
        "source_overlap_after_exclusions": source_overlap_after_exclusions,
        "label_review_limitation": (
            "Conflicting duplicate labels are reported, but remaining labels have not "
            "been independently verified by an agricultural expert."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=SERVICE_DIR / "evaluation" / "dataset_audit.json")
    parser.add_argument("--exclusions-file", type=Path, default=DEFAULT_EXCLUSIONS)
    args = parser.parse_args()

    classes = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    excluded = load_exclusions(args.exclusions_file)
    report = audit_dataset(args.data_root.expanduser().resolve(), classes, excluded)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Audited {report['file_count']} images; unreadable: {len(report['unreadable_images'])}")
    print("Clean source-image counts:", report["clean_source_image_counts_after_exclusions"])
    print("Source overlaps after exclusions:", report["source_overlap_after_exclusions"])
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
