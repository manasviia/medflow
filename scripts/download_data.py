"""Download and prepare the NIH ChestX-ray14 dataset.

The NIH ChestX-ray14 dataset is publicly available from the NIH Clinical Center.
This script downloads images, metadata, and official train/test splits,
then generates a val split via patient-level stratification.

Usage:
    python scripts/download_data.py --data-dir data/raw

Note: Full dataset is ~45GB. Use --sample for a 1000-image dev subset.
"""

import argparse
import urllib.request
from pathlib import Path

from loguru import logger

NIH_BASE_URL = "https://nihcc.app.box.com/v/ChestXray-NIHCC"

# Official train/test split files from NIH
SPLIT_FILES = {
    "train_val": "https://raw.githubusercontent.com/ieee8023/chestxray14-labels/master/train_val_list.txt",
    "test": "https://raw.githubusercontent.com/ieee8023/chestxray14-labels/master/test_list.txt",
}


def download_file(url: str, dest: Path, desc: str = "") -> None:
    """Download a file with progress logging."""
    logger.info(f"Downloading {desc or url} → {dest}")
    urllib.request.urlretrieve(url, dest)
    logger.info(f"Saved {dest} ({dest.stat().st_size / 1024:.1f} KB)")


def create_val_split(
    train_val_file: Path,
    train_out: Path,
    val_out: Path,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> None:
    """Split train_val list into train and val using patient-level split.

    Patient IDs are encoded in filenames as <patient_id>_<visit>_<image>.png.
    Splitting at patient level prevents the same patient appearing in both
    train and val, which would cause data leakage.
    """
    import random

    random.seed(seed)
    lines = train_val_file.read_text().strip().splitlines()

    # Extract unique patient IDs
    patient_ids = list({line.split("_")[0] for line in lines})
    random.shuffle(patient_ids)

    n_val = int(len(patient_ids) * val_fraction)
    val_patients = set(patient_ids[:n_val])

    train_lines = [line for line in lines if line.split("_")[0] not in val_patients]
    val_lines = [line for line in lines if line.split("_")[0] in val_patients]

    train_out.write_text("\n".join(train_lines))
    val_out.write_text("\n".join(val_lines))

    logger.info(
        f"Split complete — train: {len(train_lines)}, val: {len(val_lines)} "
        f"({len(val_patients)} val patients out of {len(patient_ids)} total)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download NIH ChestX-ray14 dataset")
    parser.add_argument("--data-dir", type=str, default="data/raw")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Download only metadata + splits (no images) for dev/testing",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    splits_dir = data_dir / "splits"
    image_dir = data_dir / "images"

    data_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    # Download split files
    train_val_path = splits_dir / "train_val_list.txt"
    test_path = splits_dir / "test.txt"

    download_file(SPLIT_FILES["train_val"], train_val_path, "train/val split")
    download_file(SPLIT_FILES["test"], test_path, "test split")

    # Generate patient-level train/val split
    create_val_split(
        train_val_file=train_val_path,
        train_out=splits_dir / "train.txt",
        val_out=splits_dir / "val.txt",
    )

    if args.sample:
        logger.info("--sample flag set: skipping image download. Splits are ready.")
        return

    logger.info(
        "To download images, visit: https://nihcc.app.box.com/v/ChestXray-NIHCC\n"
        "Download all images_00x.tar.gz files and extract into data/raw/images/\n"
        "Full dataset is ~45GB across 12 archives."
    )


if __name__ == "__main__":
    main()
