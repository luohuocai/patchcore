from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import cv2


IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}

REGION_SCRIPTS = {
    "chip": ("crop.py", "srs_cropped"),
    "line": ("srs_line/line.py", "srs_line"),
    "round": ("srs_round/round.py", "srs_round"),
    "squa": ("srs_squa/squa.py", "srs_squa"),
}

FIELDNAMES = [
    "region",
    "split",
    "anomaly",
    "source_image",
    "source_rel_path",
    "crop_image",
    "crop_rel_path",
    "crop_exists",
    "mask_image",
    "mask_exists",
    "full_width",
    "full_height",
    "x1",
    "y1",
    "x2",
    "y2",
    "crop_width",
    "crop_height",
    "output_width",
    "output_height",
    "mode",
]


def load_module(module_path: Path):
    module_name = "srs_region_{}_{}".format(module_path.stem, abs(hash(module_path)))
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load module: {}".format(module_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def module_default_args(module, script_path: Path, source_root: Path, output_root: Path):
    old_argv = sys.argv[:]
    try:
        sys.argv = [
            str(script_path),
            "--input-root",
            str(source_root),
            "--output-root",
            str(output_root),
        ]
        return module.parse_args()
    finally:
        sys.argv = old_argv


def iter_input_images(source_root: Path):
    image_dirs = [
        ("train", "good", source_root / "train" / "good"),
        ("test", "good", source_root / "test" / "good"),
        ("test", "defect", source_root / "test" / "defect"),
    ]
    for split, anomaly, image_dir in image_dirs:
        if not image_dir.exists():
            continue
        for path in sorted(image_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                yield split, anomaly, path


def find_existing_crop(path: Path, source_root: Path, region_root: Path) -> Path:
    rel_path = path.relative_to(source_root)
    expected = region_root / rel_path
    if expected.exists():
        return expected

    candidates = []
    for image_dir in (
        region_root / "train" / "good",
        region_root / "test" / "good",
        region_root / "test" / "defect",
    ):
        candidate = image_dir / path.name
        if candidate.exists():
            candidates.append(candidate)
    if candidates:
        return candidates[0]
    return expected


def read_image_size(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return 0, 0
    height, width = image.shape[:2]
    return width, height


def relative_to_or_self(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def chip_bbox(module, image, args):
    bbox = module.find_chip_bbox(
        image=image,
        max_side=args.max_side,
        saturation_threshold=args.saturation_threshold,
        value_threshold=args.value_threshold,
        column_min_fraction=args.column_min_fraction,
        row_min_fraction=args.row_min_fraction,
        margin=args.margin,
    )
    return bbox, "chip"


def line_bbox(module, image, args):
    chip = module.find_chip_bbox(
        image=image,
        max_side=args.max_side,
        saturation_threshold=args.saturation_threshold,
        value_threshold=args.value_threshold,
        column_min_fraction=args.column_min_fraction,
        row_min_fraction=args.row_min_fraction,
        margin=args.chip_margin,
    )
    return module.line_band_bbox(image, chip, args)


def round_bbox(module, image, args):
    chip = module.find_chip_bbox(
        image=image,
        max_side=args.max_side,
        saturation_threshold=args.saturation_threshold,
        value_threshold=args.value_threshold,
        column_min_fraction=args.column_min_fraction,
        row_min_fraction=args.row_min_fraction,
        margin=args.chip_margin,
    )
    candidates = module.hough_circle_candidates(image, chip, args)
    top_circle, bottom_circle, mode = module.select_circle_pair(candidates, chip, args)
    bbox = module.bbox_from_circles(
        (top_circle, bottom_circle), image.shape, margin=args.round_margin
    )
    return bbox, mode


def squa_bbox(module, image, args):
    chip = module.find_chip_bbox(
        image=image,
        max_side=args.max_side,
        saturation_threshold=args.saturation_threshold,
        value_threshold=args.value_threshold,
        column_min_fraction=args.column_min_fraction,
        row_min_fraction=args.row_min_fraction,
        margin=args.chip_margin,
    )
    candidates = module.rectangle_candidates(image, chip, args)
    top_rect, bottom_rect, mode = module.select_rect_pair(candidates, chip, args)
    bbox = module.bbox_from_rects((top_rect, bottom_rect), image.shape, args.rect_margin)
    bbox = module.trim_left_edge(bbox, chip, args)
    return bbox, mode


def compute_bbox(region: str, module, image, args):
    if region == "chip":
        return chip_bbox(module, image, args)
    if region == "line":
        return line_bbox(module, image, args)
    if region == "round":
        return round_bbox(module, image, args)
    if region == "squa":
        return squa_bbox(module, image, args)
    raise ValueError("Unsupported region: {}".format(region))


def mask_path_for_image(path: Path, source_root: Path, region_root: Path) -> Path | None:
    if path.parent.name != "defect":
        return None
    crop_mask = region_root / "ground_truth" / "defect" / "{}_mask.png".format(
        path.stem
    )
    if crop_mask.exists():
        return crop_mask
    return source_root / "ground_truth" / "defect" / "{}_mask.png".format(path.stem)


def build_region_manifest(
    region: str,
    source_root: Path,
    scripts_root: Path,
    region_root: Path,
    manifest_name: str,
) -> list[dict[str, object]]:
    script_rel_path, _ = REGION_SCRIPTS[region]
    script_path = scripts_root / script_rel_path
    if not script_path.exists():
        raise FileNotFoundError("Missing region script: {}".format(script_path))

    module = load_module(script_path)
    region_args = module_default_args(module, script_path, source_root, region_root)
    rows = []

    for split, anomaly, source_image in iter_input_images(source_root):
        image = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
        if image is None:
            print("[WARN] failed to read image: {}".format(source_image))
            continue

        bbox, mode = compute_bbox(region, module, image, region_args)
        x1, y1, x2, y2 = bbox
        crop_image = find_existing_crop(source_image, source_root, region_root)
        mask_image = mask_path_for_image(source_image, source_root, region_root)
        output_width, output_height = (
            read_image_size(crop_image) if crop_image.exists() else (0, 0)
        )
        full_height, full_width = image.shape[:2]

        rows.append(
            {
                "region": region,
                "split": split,
                "anomaly": anomaly,
                "source_image": str(source_image),
                "source_rel_path": str(source_image.relative_to(source_root)),
                "crop_image": str(crop_image),
                "crop_rel_path": relative_to_or_self(crop_image, region_root),
                "crop_exists": int(crop_image.exists()),
                "mask_image": str(mask_image) if mask_image is not None else "",
                "mask_exists": int(mask_image.exists())
                if mask_image is not None
                else 0,
                "full_width": full_width,
                "full_height": full_height,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "crop_width": x2 - x1,
                "crop_height": y2 - y1,
                "output_width": output_width,
                "output_height": output_height,
                "mode": mode,
            }
        )

    manifest_path = region_root / manifest_name
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print("Wrote {} rows: {}".format(len(rows), manifest_path))
    return rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build SRS region crop manifests with full-image ROI coordinates."
    )
    parser.add_argument(
        "--srs-root",
        type=Path,
        default=Path(r"C:\Users\Administrator\Desktop\dataset\SRS"),
        help="Root containing srs, srs_line, srs_round and srs_squa.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="Full SRS MVTec-style dataset root. Default: <srs-root>/srs.",
    )
    parser.add_argument(
        "--scripts-root",
        type=Path,
        default=None,
        help="Root containing the existing region crop scripts. Default: <srs-root>.",
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        default=["line", "round", "squa"],
        choices=sorted(REGION_SCRIPTS.keys()),
        help="Regions to describe.",
    )
    parser.add_argument(
        "--manifest-name",
        default="manifest.csv",
        help="Per-region manifest filename written under each region root.",
    )
    parser.add_argument(
        "--combined-manifest",
        type=Path,
        default=None,
        help="Combined manifest path. Default: <srs-root>/srs_region_manifest.csv.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    srs_root = args.srs_root.resolve()
    source_root = (args.source_root or (srs_root / "srs")).resolve()
    scripts_root = (args.scripts_root or srs_root).resolve()
    combined_manifest = (
        args.combined_manifest or (srs_root / "srs_region_manifest.csv")
    ).resolve()

    if not source_root.exists():
        raise SystemExit("source root does not exist: {}".format(source_root))

    all_rows = []
    for region in args.regions:
        _, default_region_dir = REGION_SCRIPTS[region]
        region_root = (srs_root / default_region_dir).resolve()
        all_rows.extend(
            build_region_manifest(
                region,
                source_root,
                scripts_root,
                region_root,
                args.manifest_name,
            )
        )

    combined_manifest.parent.mkdir(parents=True, exist_ok=True)
    with combined_manifest.open("w", newline="") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)
    print("Wrote {} combined rows: {}".format(len(all_rows), combined_manifest))


if __name__ == "__main__":
    main()
