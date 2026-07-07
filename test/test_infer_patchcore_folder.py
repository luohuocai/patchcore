import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image

from patchcore import metrics


def _load_infer_module():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "infer_patchcore_folder.py"
    )
    spec = importlib.util.spec_from_file_location(
        "infer_patchcore_folder_for_test",
        script_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compute_pixel_optimal_threshold_matches_patchcore_metrics():
    module = _load_infer_module()
    segmentations = np.array(
        [
            [[0.0, 0.2], [0.8, 0.9]],
            [[0.1, 0.7], [0.3, 0.4]],
        ],
        dtype=np.float32,
    )
    masks_gt = np.array(
        [
            [[0, 0], [1, 1]],
            [[0, 1], [0, 0]],
        ],
        dtype=np.uint8,
    )

    threshold = module.compute_pixel_optimal_threshold(segmentations, masks_gt)
    expected = metrics.compute_pixelwise_retrieval_metrics(
        segmentations,
        masks_gt,
    )["optimal_threshold"]

    assert threshold == expected


def test_resolve_mask_paths_matches_mask_suffix_in_relative_folder(tmp_path):
    module = _load_infer_module()
    input_root = tmp_path / "images"
    mask_root = tmp_path / "masks"
    image_dir = input_root / "defect"
    mask_dir = mask_root / "defect"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    image_path = image_dir / "sample.png"
    mask_path = mask_dir / "sample_mask.png"
    Image.new("RGB", (4, 4), color=(0, 0, 0)).save(image_path)
    Image.new("L", (4, 4), color=255).save(mask_path)

    resolved = module.resolve_mask_paths(
        str(mask_root),
        str(input_root),
        [str(image_path)],
    )

    assert resolved == [str(mask_path.resolve())]


def test_resolve_mask_paths_uses_zero_mask_for_good_images_without_file(tmp_path):
    module = _load_infer_module()
    input_root = tmp_path / "images"
    mask_root = tmp_path / "masks"
    image_dir = input_root / "good"
    image_dir.mkdir(parents=True)
    mask_root.mkdir()
    image_path = image_dir / "sample.png"
    Image.new("RGB", (4, 4), color=(0, 0, 0)).save(image_path)

    resolved = module.resolve_mask_paths(
        str(mask_root),
        str(input_root),
        [str(image_path)],
    )

    assert resolved == [None]


def test_normalize_segmentations_uses_dataset_wide_range():
    module = _load_infer_module()
    segmentations = np.array(
        [
            [[2.0, 3.0], [4.0, 5.0]],
            [[6.0, 7.0], [8.0, 10.0]],
        ],
        dtype=np.float32,
    )

    normalized = module.normalize_segmentations(segmentations)

    assert normalized[0, 0, 0] == 0.0
    assert normalized[1, 1, 1] == 1.0
    assert np.isclose(normalized[0, 1, 1], 3.0 / 8.0)
