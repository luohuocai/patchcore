import numpy as np

from patchcore import metrics


def test_objectwise_detection_metrics_counts_hit_miss_and_over():
    segmentations = np.zeros((1, 12, 12), dtype=np.float32)
    ground_truth_masks = np.zeros((1, 12, 12), dtype=np.uint8)

    ground_truth_masks[0, 1:5, 1:5] = 1
    ground_truth_masks[0, 8:11, 8:11] = 1
    segmentations[0, 1:5, 1:5] = 0.9
    segmentations[0, 6:8, 1:3] = 0.9

    result = metrics.compute_objectwise_detection_metrics(
        segmentations,
        ground_truth_masks,
        threshold=0.5,
    )

    assert result["gt_total"] == 2
    assert result["pred_total"] == 2
    assert result["hit"] == 1
    assert result["miss"] == 1
    assert result["over"] == 1
    assert result["hit_rate"] == 0.5
    assert result["miss_rate"] == 0.5
    assert result["over_rate"] == 0.5


def test_objectwise_detection_metrics_counts_predictions_without_gt_as_over():
    segmentations = np.zeros((1, 8, 8), dtype=np.float32)
    ground_truth_masks = np.zeros((1, 8, 8), dtype=np.uint8)

    segmentations[0, 2:5, 2:5] = 0.9

    result = metrics.compute_objectwise_detection_metrics(
        segmentations,
        ground_truth_masks,
        threshold=0.5,
    )

    assert result["gt_total"] == 0
    assert result["pred_total"] == 1
    assert result["hit"] == 0
    assert result["miss"] == 0
    assert result["over"] == 1
    assert result["hit_rate"] == 0.0
    assert result["miss_rate"] == 0.0
    assert result["over_rate"] == 1.0
