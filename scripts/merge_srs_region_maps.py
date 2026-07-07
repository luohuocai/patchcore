from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


SUMMARY_COLUMNS = [
    "source_image",
    "source_rel_path",
    "image_score",
    "region_scores",
    "regions",
    "score_map_path",
    "heatmap_path",
    "overlay_path",
]


def normalize_path(path_value: str, base: Path | None = None) -> str:
    path = Path(path_value)
    if not path.is_absolute() and base is not None:
        path = base / path
    return str(path.resolve()).lower()


def resolve_path(path_value: str, base: Path | None = None) -> Path:
    path = Path(path_value)
    if not path.is_absolute() and base is not None:
        path = base / path
    return path.resolve()


def safe_output_stem(source_rel_path: str, source_image: Path) -> str:
    rel = source_rel_path or source_image.name
    stem = str(rel).replace("\\", "__").replace("/", "__")
    suffix = Path(stem).suffix
    if suffix:
        stem = stem[: -len(suffix)]
    return stem


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def parse_region_spec(spec: str):
    parts = spec.split("|")
    if len(parts) not in (3, 4):
        raise ValueError(
            "Region spec must be name|manifest.csv|segmentation_maps.csv[|weight]"
        )
    name = parts[0].strip()
    manifest_path = Path(parts[1]).resolve()
    map_index_path = Path(parts[2]).resolve()
    weight = float(parts[3]) if len(parts) == 4 else 1.0
    return name, manifest_path, map_index_path, weight


def manifest_lookup(rows: list[dict[str, str]], region: str, manifest_path: Path):
    path_lookup = {}
    name_lookup = defaultdict(list)
    for row in rows:
        if row.get("region") and row["region"] != region:
            continue
        crop_image = row.get("crop_image", "")
        if not crop_image:
            continue
        crop_key = normalize_path(crop_image, manifest_path.parent)
        path_lookup[crop_key] = row
        name_lookup[Path(crop_image).name].append(row)
    return path_lookup, name_lookup


def match_manifest_row(
    map_row: dict[str, str],
    path_lookup: dict[str, dict[str, str]],
    name_lookup: dict[str, list[dict[str, str]]],
    map_index_path: Path,
):
    image_path = map_row.get("image_path", "")
    image_key = normalize_path(image_path, map_index_path.parent)
    if image_key in path_lookup:
        return path_lookup[image_key]

    candidates = name_lookup.get(Path(image_path).name, [])
    if len(candidates) == 1:
        return candidates[0]
    return None


def collect_region_items(region_specs, source_root: Path | None):
    items_by_source = defaultdict(list)
    for spec in region_specs:
        region, manifest_path, map_index_path, weight = parse_region_spec(spec)
        manifest_rows = read_csv(manifest_path)
        map_rows = read_csv(map_index_path)
        path_lookup, name_lookup = manifest_lookup(manifest_rows, region, manifest_path)

        matched = 0
        for map_row in map_rows:
            manifest_row = match_manifest_row(
                map_row, path_lookup, name_lookup, map_index_path
            )
            if manifest_row is None:
                print(
                    "[WARN] no manifest row for region={} image={}".format(
                        region, map_row.get("image_path", "")
                    )
                )
                continue

            source_image = resolve_path(
                manifest_row["source_image"],
                source_root,
            )
            map_path = resolve_path(map_row["map_path"], map_index_path.parent)
            source_key = normalize_path(str(source_image))
            items_by_source[source_key].append(
                {
                    "region": region,
                    "source_image": source_image,
                    "manifest": manifest_row,
                    "map_path": map_path,
                    "weight": weight,
                    "anomaly_score": float(map_row.get("anomaly_score", 0.0)),
                }
            )
            matched += 1
        print(
            "Matched {} maps for region={} using {}".format(
                matched, region, map_index_path
            )
        )
    return items_by_source


def clipped_bbox(row: dict[str, str], width: int, height: int):
    x1 = max(0, min(width, int(round(float(row["x1"])))))
    y1 = max(0, min(height, int(round(float(row["y1"])))))
    x2 = max(0, min(width, int(round(float(row["x2"])))))
    y2 = max(0, min(height, int(round(float(row["y2"])))))
    return x1, y1, x2, y2


def write_heatmap_outputs(
    output_dir: Path,
    source_image: Path,
    source_rel_path: str,
    image: np.ndarray,
    full_score: np.ndarray,
    contribution_mask: np.ndarray,
    threshold: float,
    alpha: float,
    draw_rois: bool,
    roi_rows: list[dict[str, str]],
):
    stem = safe_output_stem(source_rel_path, source_image)
    score_map_path = output_dir / "score_maps" / "{}.npy".format(stem)
    heatmap_path = output_dir / "heatmaps" / "{}.png".format(stem)
    overlay_path = output_dir / "overlays" / "{}.png".format(stem)
    score_map_path.parent.mkdir(parents=True, exist_ok=True)
    heatmap_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)

    full_score = np.clip(full_score, 0.0, 1.0).astype(np.float32)
    np.save(score_map_path, full_score)

    heatmap_u8 = np.clip(full_score * 255.0, 0, 255).astype(np.uint8)
    color_heatmap = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    color_heatmap[contribution_mask == 0] = 0
    cv2.imwrite(str(heatmap_path), color_heatmap)

    overlay = cv2.addWeighted(image, 1.0 - alpha, color_heatmap, alpha, 0)
    overlay[contribution_mask == 0] = image[contribution_mask == 0]

    binary = (full_score >= threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)

    if draw_rois:
        height, width = image.shape[:2]
        for row in roi_rows:
            x1, y1, x2, y2 = clipped_bbox(row, width, height)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 1)
            cv2.putText(
                overlay,
                row["region"],
                (x1, max(12, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    cv2.imwrite(str(overlay_path), overlay)
    return score_map_path, heatmap_path, overlay_path


def merge_one_source(
    source_key: str,
    items: list[dict[str, object]],
    output_dir: Path,
    threshold: float,
    alpha: float,
    draw_rois: bool,
):
    source_image = Path(items[0]["source_image"])
    image = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
    if image is None:
        print("[WARN] failed to read source image: {}".format(source_image))
        return None

    height, width = image.shape[:2]
    full_score = np.zeros((height, width), dtype=np.float32)
    contribution_mask = np.zeros((height, width), dtype=np.uint8)
    region_scores = []
    roi_rows = []

    for item in items:
        row = dict(item["manifest"])
        row["region"] = str(item["region"])
        x1, y1, x2, y2 = clipped_bbox(row, width, height)
        if x2 <= x1 or y2 <= y1:
            print("[WARN] invalid bbox for {}: {}".format(source_image, row))
            continue

        map_path = Path(item["map_path"])
        if not map_path.exists():
            print("[WARN] missing score map: {}".format(map_path))
            continue

        region_score = np.load(str(map_path)).astype(np.float32)
        region_score = cv2.resize(
            region_score,
            (x2 - x1, y2 - y1),
            interpolation=cv2.INTER_LINEAR,
        )
        region_score = np.clip(region_score * float(item["weight"]), 0.0, None)

        target = full_score[y1:y2, x1:x2]
        np.maximum(target, region_score, out=target)
        contribution_mask[y1:y2, x1:x2] = 1
        region_scores.append("{}:{:.6f}".format(item["region"], region_score.max()))
        roi_rows.append(row)

    source_rel_path = str(items[0]["manifest"].get("source_rel_path", ""))
    score_map_path, heatmap_path, overlay_path = write_heatmap_outputs(
        output_dir,
        source_image,
        source_rel_path,
        image,
        full_score,
        contribution_mask,
        threshold,
        alpha,
        draw_rois,
        roi_rows,
    )

    return {
        "source_image": str(source_image),
        "source_rel_path": source_rel_path,
        "image_score": float(full_score.max()) if full_score.size else 0.0,
        "region_scores": ";".join(region_scores),
        "regions": ";".join(sorted({str(item["region"]) for item in items})),
        "score_map_path": str(score_map_path),
        "heatmap_path": str(heatmap_path),
        "overlay_path": str(overlay_path),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge SRS region PatchCore score maps back onto full images."
    )
    parser.add_argument(
        "--region",
        action="append",
        required=True,
        help=(
            "Repeated region spec: "
            "name|manifest.csv|segmentation_maps.csv[|weight]"
        ),
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="Optional base path for relative source_image fields.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"D:\lhc\patchcore\srs_region_merged"),
        help="Output folder for merged score maps and visualizations.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Normalized threshold used to draw final anomaly contours.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Heatmap overlay opacity.",
    )
    parser.add_argument(
        "--draw-rois",
        action="store_true",
        help="Draw region boxes and labels on overlay images.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    source_root = args.source_root.resolve() if args.source_root else None
    output_dir.mkdir(parents=True, exist_ok=True)

    items_by_source = collect_region_items(args.region, source_root)
    rows = []
    for source_key, items in sorted(items_by_source.items()):
        row = merge_one_source(
            source_key,
            items,
            output_dir,
            args.threshold,
            args.alpha,
            args.draw_rois,
        )
        if row is not None:
            rows.append(row)

    summary_path = output_dir / "merged_summary.csv"
    with summary_path.open("w", newline="") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print("Merged {} images.".format(len(rows)))
    print("Summary CSV: {}".format(summary_path))


if __name__ == "__main__":
    main()
