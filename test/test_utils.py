import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")

from patchcore import utils


def test_compute_anomaly_contours_from_thresholded_score_map():
    score_map = np.zeros((8, 8), dtype=np.float32)
    score_map[2:5, 3:6] = 0.8

    contours = utils._compute_anomaly_contours(score_map, threshold=0.5)

    assert contours
    assert all(contour.shape[1] == 2 for contour in contours)


def test_compute_anomaly_contours_handles_full_image_region():
    score_map = np.ones((8, 8), dtype=np.float32)

    contours = utils._compute_anomaly_contours(score_map, threshold=0.5)

    assert contours


def test_plot_segmentation_images_saves_contour_overlay_without_masks(tmp_path):
    image_path = tmp_path / "input.png"
    Image.new("RGB", (16, 12), color=(20, 40, 60)).save(image_path)
    score_map = np.zeros((12, 16), dtype=np.float32)
    score_map[3:8, 4:12] = 0.9
    output_dir = tmp_path / "visualizations"

    utils.plot_segmentation_images(
        str(output_dir),
        [str(image_path)],
        [score_map],
        anomaly_score_threshold=0.5,
        save_depth=1,
    )

    saved_image = output_dir / "input.png"
    assert saved_image.is_file()
    assert saved_image.stat().st_size > 0
