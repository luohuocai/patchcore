import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def is_zero_mask(label_path):
    with Image.open(label_path) as label_image:
        return label_image.getextrema()[1] == 0


def make_binary_mask(label_path, output_path):
    with Image.open(label_path) as label_image:
        label_array = np.array(label_image)
    mask_array = (label_array != 0).astype(np.uint8) * 255
    Image.fromarray(mask_array, mode="L").save(output_path)


def link_or_copy_image(image_path, output_image_path):
    try:
        output_image_path.hardlink_to(image_path)
    except OSError:
        shutil.copy2(image_path, output_image_path)


def copy_pair(image_path, label_path, output_image_path, output_mask_path=None):
    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    link_or_copy_image(image_path, output_image_path)
    if output_mask_path is not None:
        output_mask_path.parent.mkdir(parents=True, exist_ok=True)
        make_binary_mask(label_path, output_mask_path)


def convert(source, output):
    source = Path(source)
    output = Path(output)
    dataset_root = output / "srs"

    if not source.exists():
        raise FileNotFoundError(f"Source does not exist: {source}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output already exists and is not empty: {output}")

    samples = []
    missing_labels = []
    for split in ("train", "val", "test"):
        image_dir = source / "images" / split
        label_dir = source / "labels" / split
        for image_path in sorted(image_dir.glob("*.bmp")):
            label_path = label_dir / f"{image_path.stem}.png"
            if not label_path.exists():
                missing_labels.append(label_path)
                continue
            samples.append(
                {
                    "split": split,
                    "image_path": image_path,
                    "label_path": label_path,
                    "is_good": is_zero_mask(label_path),
                }
            )

    if missing_labels:
        missing_preview = "\n".join(str(path) for path in missing_labels[:10])
        raise FileNotFoundError(
            f"Missing {len(missing_labels)} label files. First missing labels:\n"
            f"{missing_preview}"
        )

    rows = []
    counts = {
        "train_good": 0,
        "test_good": 0,
        "test_defect": 0,
    }

    (dataset_root / "train" / "good").mkdir(parents=True, exist_ok=True)
    (dataset_root / "test" / "good").mkdir(parents=True, exist_ok=True)
    (dataset_root / "test" / "defect").mkdir(parents=True, exist_ok=True)
    (dataset_root / "ground_truth" / "defect").mkdir(parents=True, exist_ok=True)

    for sample in samples:
        split = sample["split"]
        image_path = sample["image_path"]
        label_path = sample["label_path"]
        output_name = f"{split}__{image_path.name}"
        if split == "train" and sample["is_good"]:
            target_image = dataset_root / "train" / "good" / output_name
            copy_pair(image_path, label_path, target_image)
            target_split = "train"
            target_class = "good"
            counts["train_good"] += 1
        elif sample["is_good"]:
            target_image = dataset_root / "test" / "good" / output_name
            copy_pair(image_path, label_path, target_image)
            target_split = "test"
            target_class = "good"
            counts["test_good"] += 1
        else:
            target_image = dataset_root / "test" / "defect" / output_name
            target_mask = (
                dataset_root
                / "ground_truth"
                / "defect"
                / f"{Path(output_name).stem}_mask.png"
            )
            copy_pair(image_path, label_path, target_image, target_mask)
            target_split = "test"
            target_class = "defect"
            counts["test_defect"] += 1

        rows.append(
            {
                "source_split": split,
                "source_image": str(image_path.relative_to(source)),
                "source_label": str(label_path.relative_to(source)),
                "source_mask_kind": "zero_mask_good"
                if sample["is_good"]
                else "nonzero_mask_defect",
                "target_split": target_split,
                "target_class": target_class,
                "target_image": str(target_image.relative_to(output)),
            }
        )

    metadata_path = output / "conversion_metadata.csv"
    output.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", newline="", encoding="utf-8") as metadata_file:
        writer = csv.DictWriter(metadata_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    readme_path = output / "README_SRS_patchcore.txt"
    readme_path.write_text(
        "\n".join(
            [
                "SRS PatchCore/MVTec-style conversion",
                "",
                "Source format: images/{train,val,test} + labels/{train,val,test}.",
                "Target format: srs/train/good, srs/test/good, srs/test/defect, srs/ground_truth/defect.",
                "",
                "Important:",
                "- Normal/good samples are detected only by all-zero label masks.",
                "- Filenames are not used to infer normal/defect status.",
                "- If no normal samples are found, empty good folders are still created.",
                "- PatchCore training still requires normal images in train/good.",
                "- Defect labels are converted to binary masks: 0 stays 0, non-zero becomes 255.",
                "",
                f"train/good: {counts['train_good']}",
                f"test/good: {counts['test_good']}",
                f"test/defect: {counts['test_defect']}",
            ]
        ),
        encoding="utf-8",
    )

    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default=r"C:\Users\Administrator\Desktop\dataset\SRS",
        help="Original SRS dataset directory.",
    )
    parser.add_argument(
        "--output",
        default=r"C:\Users\Administrator\Desktop\dataset\SRS_patchcore",
        help="Output directory for the converted MVTec-style dataset.",
    )
    args = parser.parse_args()

    counts = convert(args.source, args.output)
    for key, value in counts.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
