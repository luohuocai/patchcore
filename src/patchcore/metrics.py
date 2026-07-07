"""Anomaly metrics."""
import numpy as np
from scipy import ndimage
from sklearn import metrics


DEFAULT_MIN_COMPONENT_AREA = 1
DEFAULT_HIT_IOU_THRESHOLD = 0.10


def compute_imagewise_retrieval_metrics(
    anomaly_prediction_weights, anomaly_ground_truth_labels
):
    """
    Computes retrieval statistics (AUROC, FPR, TPR).

    Args:
        anomaly_prediction_weights: [np.array or list] [N] Assignment weights
                                    per image. Higher indicates higher
                                    probability of being an anomaly.
        anomaly_ground_truth_labels: [np.array or list] [N] Binary labels - 1
                                    if image is an anomaly, 0 if not.
    """
    fpr, tpr, thresholds = metrics.roc_curve(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    auroc = metrics.roc_auc_score(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    return {"auroc": auroc, "fpr": fpr, "tpr": tpr, "threshold": thresholds}


def compute_pixelwise_retrieval_metrics(anomaly_segmentations, ground_truth_masks):
    """
    Computes pixel-wise statistics (AUROC, FPR, TPR) for anomaly segmentations
    and ground truth segmentation masks.

    Args:
        anomaly_segmentations: [list of np.arrays or np.array] [NxHxW] Contains
                                generated segmentation masks.
        ground_truth_masks: [list of np.arrays or np.array] [NxHxW] Contains
                            predefined ground truth segmentation masks
    """
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    flat_anomaly_segmentations = anomaly_segmentations.ravel()
    flat_ground_truth_masks = ground_truth_masks.ravel()

    fpr, tpr, thresholds = metrics.roc_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    auroc = metrics.roc_auc_score(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )

    precision, recall, thresholds = metrics.precision_recall_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    F1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )

    optimal_threshold = thresholds[np.argmax(F1_scores)]
    predictions = (flat_anomaly_segmentations >= optimal_threshold).astype(int)
    fpr_optim = np.mean(predictions > flat_ground_truth_masks)
    fnr_optim = np.mean(predictions < flat_ground_truth_masks)

    return {
        "auroc": auroc,
        "fpr": fpr,
        "tpr": tpr,
        "optimal_threshold": optimal_threshold,
        "optimal_fpr": fpr_optim,
        "optimal_fnr": fnr_optim,
    }


def _as_numpy_stack(masks):
    if isinstance(masks, list):
        return np.stack(masks)
    return np.asarray(masks)


def _as_2d_mask(mask):
    mask = np.asarray(mask)
    if mask.ndim == 2:
        return mask

    mask = np.squeeze(mask)
    if mask.ndim == 2:
        return mask

    if mask.ndim == 3 and mask.shape[0] in (1, 3):
        return np.max(mask, axis=0)
    if mask.ndim == 3 and mask.shape[-1] in (1, 3):
        return np.max(mask, axis=-1)

    raise ValueError(
        "Expected a 2D mask or a mask with a singleton/channel dimension."
    )


def _connected_components(mask, min_component_area):
    mask = _as_2d_mask(mask).astype(bool)
    labels, _ = ndimage.label(mask)
    objects = ndimage.find_objects(labels)

    components = []
    for label_id, slices in enumerate(objects, start=1):
        if slices is None:
            continue

        component_mask = labels[slices] == label_id
        area = int(np.sum(component_mask))
        if area < min_component_area:
            continue

        components.append((slices, component_mask))
    return components


def _component_area(component):
    return int(np.sum(component[1]))


def _component_iou(component_a, component_b):
    slices_a, mask_a = component_a
    slices_b, mask_b = component_b
    y_slice_a, x_slice_a = slices_a
    y_slice_b, x_slice_b = slices_b

    y_start = max(y_slice_a.start, y_slice_b.start)
    y_stop = min(y_slice_a.stop, y_slice_b.stop)
    x_start = max(x_slice_a.start, x_slice_b.start)
    x_stop = min(x_slice_a.stop, x_slice_b.stop)
    if y_start >= y_stop or x_start >= x_stop:
        return 0.0

    mask_a_overlap = mask_a[
        y_start - y_slice_a.start : y_stop - y_slice_a.start,
        x_start - x_slice_a.start : x_stop - x_slice_a.start,
    ]
    mask_b_overlap = mask_b[
        y_start - y_slice_b.start : y_stop - y_slice_b.start,
        x_start - x_slice_b.start : x_stop - x_slice_b.start,
    ]
    intersection = int(np.sum(mask_a_overlap & mask_b_overlap))
    if intersection <= 0:
        return 0.0

    union = _component_area(component_a) + _component_area(component_b) - intersection
    return intersection / union if union > 0 else 0.0


def _iou_is_hit(iou, hit_iou_threshold):
    if hit_iou_threshold <= 0:
        return iou > 0
    return iou >= hit_iou_threshold


def _evaluate_contour_overlaps(
    gt_components,
    pred_components,
    hit_iou_threshold,
):
    gt_hit = [
        any(
            _iou_is_hit(_component_iou(gt_component, pred_component), hit_iou_threshold)
            for pred_component in pred_components
        )
        for gt_component in gt_components
    ]
    pred_hit = [
        any(
            _iou_is_hit(_component_iou(pred_component, gt_component), hit_iou_threshold)
            for gt_component in gt_components
        )
        for pred_component in pred_components
    ]
    return gt_hit, pred_hit


def compute_objectwise_detection_metrics(
    anomaly_segmentations,
    ground_truth_masks,
    threshold,
    hit_iou_threshold=None,
    min_component_area=DEFAULT_MIN_COMPONENT_AREA,
):
    """
    Computes sample-level hit, miss and over-detection metrics.

    Each predicted contour is represented by one thresholded connected component.
    A GT object is hit if its IoU with any predicted contour is at least
    hit_iou_threshold. A positive sample is counted as missed only when none of
    its GT objects is hit. A positive sample with any hit GT is not counted as
    over-detected. An OK sample is over-detected if it has any prediction.
    """
    if hit_iou_threshold is None:
        hit_iou_threshold = DEFAULT_HIT_IOU_THRESHOLD
    if hit_iou_threshold < 0 or hit_iou_threshold > 1:
        raise ValueError("hit_iou_threshold must be in [0, 1].")

    anomaly_segmentations = _as_numpy_stack(anomaly_segmentations)
    ground_truth_masks = _as_numpy_stack(ground_truth_masks)

    if len(anomaly_segmentations) != len(ground_truth_masks):
        raise ValueError("#Segmentations and #GT masks must match.")

    total_gt = 0
    total_pred = 0
    total_hit_objects = 0
    total_miss_objects = 0
    total_over_objects = 0
    total_samples = 0
    sample_hit = 0
    sample_miss = 0
    sample_over = 0
    per_image = []

    for segmentation, gt_mask in zip(anomaly_segmentations, ground_truth_masks):
        segmentation = _as_2d_mask(segmentation)
        gt_mask = _as_2d_mask(gt_mask) > 0
        pred_mask = segmentation >= threshold
        pred_components = _connected_components(pred_mask, min_component_area)
        gt_components = _connected_components(gt_mask, min_component_area)

        gt_hit, pred_hit = _evaluate_contour_overlaps(
            gt_components,
            pred_components,
            hit_iou_threshold,
        )

        hit_objects = int(np.sum(gt_hit))
        miss_objects = len(gt_components) - hit_objects
        over_objects = int(len(pred_hit) - np.sum(pred_hit))
        has_gt = len(gt_components) > 0
        has_hit_gt = hit_objects > 0
        miss_sample = int(has_gt and not has_hit_gt)
        over_sample = int(
            (not has_gt and len(pred_components) > 0)
            or (has_gt and not has_hit_gt and over_objects > 0)
        )
        hit_sample = int(has_hit_gt or (not has_gt and len(pred_components) == 0))

        total_gt += len(gt_components)
        total_pred += len(pred_components)
        total_hit_objects += hit_objects
        total_miss_objects += miss_objects
        total_over_objects += over_objects
        total_samples += 1
        sample_hit += hit_sample
        sample_miss += miss_sample
        sample_over += over_sample

        per_image.append(
            {
                "gt_count": len(gt_components),
                "pred_count": len(pred_components),
                "hit": hit_sample,
                "miss": miss_sample,
                "over": over_sample,
                "hit_object_count": hit_objects,
                "miss_object_count": miss_objects,
                "over_object_count": over_objects,
            }
        )

    sample_hit_rate = sample_hit / total_samples if total_samples > 0 else 0.0
    sample_miss_rate = sample_miss / total_samples if total_samples > 0 else 0.0
    sample_over_rate = sample_over / total_samples if total_samples > 0 else 0.0
    return {
        "sample_total": total_samples,
        "sample_hit_count": sample_hit,
        "sample_miss_count": sample_miss,
        "sample_over_count": sample_over,
        "sample_hit_rate": sample_hit_rate,
        "sample_miss_rate": sample_miss_rate,
        "sample_over_rate": sample_over_rate,
        "gt_total": total_gt,
        "pred_total": total_pred,
        "hit": sample_hit,
        "miss": sample_miss,
        "over": sample_over,
        "hit_rate": sample_hit_rate,
        "miss_rate": sample_miss_rate,
        "overall_miss_rate": sample_miss_rate,
        "over_rate": sample_over_rate,
        "hit_object_count": total_hit_objects,
        "miss_object_count": total_miss_objects,
        "over_object_count": total_over_objects,
        "object_hit_rate": total_hit_objects / total_gt if total_gt > 0 else 0.0,
        "object_miss_rate": total_miss_objects / total_gt if total_gt > 0 else 0.0,
        "object_over_rate": total_over_objects / total_pred if total_pred > 0 else 0.0,
        "threshold": float(threshold),
        "hit_rule": "component_iou_at_least_threshold",
        "sample_miss_rule": "positive_sample_missed_only_when_no_gt_object_is_hit",
        "sample_over_rule": "ok_sample_with_prediction_or_positive_sample_with_no_hit_and_false_prediction",
        "hit_iou_threshold": float(hit_iou_threshold),
        "min_component_area": min_component_area,
        "per_image": per_image,
    }
