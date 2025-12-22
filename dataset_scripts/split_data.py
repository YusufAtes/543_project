"""
Split images into train/val/test sets and write ID lists.

This script scans the `images/` folder under the dataset directory,
randomly splits image IDs into train/val/test (80/10/10 by default),
and writes three files under `dataset/splits/`:

- train_ids.txt
- val_ids.txt
- test_ids.txt

Each line in these files is an image ID (filename without extension),
which corresponds to an image path like `images/<id>.jpg`.
This format matches what `ArtCaptionDataset` in `im_text_dataset.py` expects.
"""

from pathlib import Path
import random
from typing import List, Tuple


def split_image_ids(
    dataset_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Create train/val/test splits from images under `images/` in dataset_dir.

    Returns:
        (train_ids, val_ids, test_ids) where each is a list of image IDs
        (filename stem without extension).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    images_dir = dataset_dir / "images"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    # Collect image files
    exts = {".jpg", ".jpeg", ".png"}
    image_files = sorted(
        [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    )
    if not image_files:
        raise RuntimeError(f"No image files found in {images_dir}")

    all_ids = [p.stem for p in image_files]
    print(f"Found {len(all_ids)} images in {images_dir}")

    # Random split
    random.seed(seed)
    random.shuffle(all_ids)

    n = len(all_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    # Remaining go to test
    n_test = n - n_train - n_val

    train_ids = all_ids[:n_train]
    val_ids = all_ids[n_train:n_train + n_val]
    test_ids = all_ids[n_train + n_val:]

    assert len(train_ids) + len(val_ids) + len(test_ids) == n

    print(
        f"Split sizes -> train: {len(train_ids)}, "
        f"val: {len(val_ids)}, test: {len(test_ids)}"
    )

    return train_ids, val_ids, test_ids


def main() -> None:
    # Resolve dataset directory as the parent of this script's directory
    scripts_dir = Path(__file__).resolve().parent
    dataset_dir = scripts_dir.parent

    print(f"Dataset directory: {dataset_dir}")

    train_ids, val_ids, test_ids = split_image_ids(dataset_dir)

    splits_dir = dataset_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    train_file = splits_dir / "train_ids.txt"
    val_file = splits_dir / "val_ids.txt"
    test_file = splits_dir / "test_ids.txt"

    def _write_ids(path: Path, ids: List[str]) -> None:
        with path.open("w", encoding="utf-8") as f:
            f.write("\n".join(ids))

    _write_ids(train_file, train_ids)
    _write_ids(val_file, val_ids)
    _write_ids(test_file, test_ids)

    total = len(train_ids) + len(val_ids) + len(test_ids)
    print("\nWrote split files:")
    print(f"  {train_file}  ({len(train_ids)} ids)")
    print(f"  {val_file}    ({len(val_ids)} ids)")
    print(f"  {test_file}   ({len(test_ids)} ids)")
    print(f"\nTotal images: {total}")


if __name__ == "__main__":
    main()


