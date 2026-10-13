from pathlib import Path

import pytest
import yaml
from PIL import Image

from scripts.validate_detection_data import validate_dataset


def make_dataset(root: Path) -> None:
    for index, split in enumerate(("train", "val", "test")):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        Image.new("L", (16, 16), color=index * 50).save(image_dir / "sample.png")
        (label_dir / "sample.txt").write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")


def test_valid_dataset(tmp_path: Path) -> None:
    make_dataset(tmp_path)
    assert validate_dataset(tmp_path) == {"train": 1, "val": 1, "test": 1}


def test_background_image(tmp_path: Path) -> None:
    make_dataset(tmp_path)
    (tmp_path / "labels" / "train" / "sample.txt").write_text("", encoding="utf-8")
    assert validate_dataset(tmp_path)["train"] == 1


@pytest.mark.parametrize(
    "label",
    [
        "1 0.5 0.5 0.2 0.2",
        "0 0.5 0.5 0 0.2",
        "0 nan 0.5 0.2 0.2",
        "0 0.05 0.5 0.2 0.2",
        "0 x 0.5 0.2 0.2",
        "0 0.5 0.5 0.2",
        "0 1.1 0.5 0.2 0.2",
        "0 0.5 0.5 inf 0.2",
    ],
)
def test_invalid_boxes(tmp_path: Path, label: str) -> None:
    make_dataset(tmp_path)
    (tmp_path / "labels" / "train" / "sample.txt").write_text(label, encoding="utf-8")
    with pytest.raises(ValueError):
        validate_dataset(tmp_path)


def test_missing_label(tmp_path: Path) -> None:
    make_dataset(tmp_path)
    (tmp_path / "labels" / "train" / "sample.txt").unlink()
    with pytest.raises(ValueError, match="missing label"):
        validate_dataset(tmp_path)


def test_duplicate_across_splits(tmp_path: Path) -> None:
    make_dataset(tmp_path)
    (tmp_path / "images" / "val" / "sample.png").write_bytes(
        (tmp_path / "images" / "train" / "sample.png").read_bytes()
    )
    with pytest.raises(ValueError, match="Split leakage"):
        validate_dataset(tmp_path)


def test_empty_split(tmp_path: Path) -> None:
    make_dataset(tmp_path)
    (tmp_path / "images" / "test" / "sample.png").unlink()
    with pytest.raises(ValueError, match="no supported images"):
        validate_dataset(tmp_path)


def test_orphan_label(tmp_path: Path) -> None:
    make_dataset(tmp_path)
    (tmp_path / "labels" / "train" / "orphan.txt").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="labels without images"):
        validate_dataset(tmp_path)


def test_corrupt_image(tmp_path: Path) -> None:
    make_dataset(tmp_path)
    (tmp_path / "images" / "train" / "sample.png").write_bytes(b"not an image")
    with pytest.raises(OSError):
        validate_dataset(tmp_path)


def test_shared_label_stem(tmp_path: Path) -> None:
    make_dataset(tmp_path)
    Image.new("L", (16, 16)).save(tmp_path / "images" / "train" / "sample.jpg")
    with pytest.raises(ValueError, match="multiple images"):
        validate_dataset(tmp_path)


def test_detection_configs() -> None:
    root = Path(__file__).resolve().parents[1]
    data = yaml.safe_load((root / "configs" / "detection-data.yaml").read_text())
    smoke = yaml.safe_load((root / "configs" / "detector-smoke.yaml").read_text())
    assert data["names"] == {0: "vessel"}
    for split in ("train", "val", "test"):
        assert (root / data["path"] / data[split]).is_dir()
        assert (root / data["path"] / "labels" / split).is_dir()
    assert (root / smoke["data"]).is_file()
    assert smoke["device"] == "cpu"
    assert smoke["epochs"] == 1
