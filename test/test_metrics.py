import numpy as np

from patchcore import metrics


def test_objectwise_detection_metrics_counts_sample_hit_when_any_gt_is_hit():
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

    assert result["sample_total"] == 1
    assert result["gt_total"] == 2
    assert result["pred_total"] == 2
    assert result["hit"] == 1
    assert result["miss"] == 0
    assert result["over"] == 0
    assert result["hit_rate"] == 1.0
    assert result["miss_rate"] == 0.0
    assert result["overall_miss_rate"] == 0.0
    assert result["over_rate"] == 0.0
    assert result["hit_object_count"] == 1
    assert result["miss_object_count"] == 1
    assert result["over_object_count"] == 1
    assert result["object_miss_rate"] == 0.5
    assert result["object_over_rate"] == 0.5
    assert result["hit_rule"] == "component_iou_at_least_threshold"
    assert result["hit_iou_threshold"] == 0.10


def test_objectwise_detection_metrics_counts_ok_sample_with_prediction_as_over():
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
    assert result["overall_miss_rate"] == 0.0
    assert result["over_rate"] == 1.0
    assert result["over_object_count"] == 1


def test_objectwise_detection_metrics_requires_iou_threshold_for_hit():
    segmentations = np.zeros((1, 20, 20), dtype=np.float32)
    ground_truth_masks = np.zeros((1, 20, 20), dtype=np.uint8)

    ground_truth_masks[0, 2:18, 2:18] = 1
    segmentations[0, 17:20, 17:20] = 0.9

    result = metrics.compute_objectwise_detection_metrics(
        segmentations,
        ground_truth_masks,
        threshold=0.5,
    )

    assert result["gt_total"] == 1
    assert result["pred_total"] == 1
    assert result["hit"] == 0
    assert result["miss"] == 1
    assert result["over"] == 1
    assert result["overall_miss_rate"] == 1.0
    assert result["over_rate"] == 1.0
    assert result["hit_object_count"] == 0
    assert result["miss_object_count"] == 1
    assert result["over_object_count"] == 1
