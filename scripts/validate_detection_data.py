"""Validate the project's image-only YOLO detection dataset."""

import argparse
import hashlib
import math
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")


def validate_dataset(root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    seen: dict[str, tuple[str, Path]] = {}

    for split in SPLITS:
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise ValueError(f"{split}: missing images or labels directory under {root}")
        
        images = sorted(
            path for path in image_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        
        if not images:
            raise ValueError(f"{image_dir}: no supported images found")
        expected_labels: set[Path] = set()
        
        for image_path in images:
            relative = image_path.relative_to(image_dir)
            label_path = label_dir / relative.with_suffix(".txt")
            if label_path in expected_labels:
                raise ValueError(f"{image_path}: multiple images map to {label_path}")
            expected_labels.add(label_path)
            if not label_path.is_file():
                raise ValueError(f"{image_path}: missing label {label_path}")
            with Image.open(image_path) as image:
                image.load()
            digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
            if digest in seen and seen[digest][0] != split:
                raise ValueError(
                    f"Split leakage: {image_path} duplicates {seen[digest][1]}"
                )
            
            seen[digest] = (split, image_path)
            
            for line_number, line in enumerate(
                label_path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                fields = line.split()
                location = f"{label_path}:{line_number}"
                if len(fields) != 5 or fields[0] != "0":
                    raise ValueError(
                        f"{location}: expected '0 x_center y_center width height'"
                    )
                try:
                    x, y, width, height = map(float, fields[1:])
                except ValueError as error:
                    raise ValueError(f"{location}: coordinates must be numeric") from error
                if not all(math.isfinite(value) for value in (x, y, width, height)):
                    raise ValueError(f"{location}: coordinates must be finite")
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
                    raise ValueError(f"{location}: invalid normalized coordinates")
                tolerance = 1e-6
                
                if (
                    x - width / 2 < -tolerance
                    or y - height / 2 < -tolerance
                    or x + width / 2 > 1 + tolerance
                    or y + height / 2 > 1 + tolerance
                ):
                    raise ValueError(f"{location}: box extends beyond the image")
        
        orphan_labels = set(label_dir.rglob("*.txt")) - expected_labels
        
        if orphan_labels:
            raise ValueError(f"{split}: labels without images: {sorted(orphan_labels)}")
        counts[split] = len(images)
    
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/detection"))
    args = parser.parse_args()
    try:
        counts = validate_dataset(args.root)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Dataset validation failed: {error}\n")
    for split, count in counts.items():
        print(f"{split}: {count} images")
    print("Dataset validation passed.")


if __name__ == "__main__":
    main()
