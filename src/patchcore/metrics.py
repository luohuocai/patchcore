"""Anomaly metrics."""
import numpy as np
from scipy import ndimage
from sklearn import metrics


DEFAULT_MIN_COMPONENT_AREA = 1


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


def _component_overlaps_mask(component, mask):
    slices, component_mask = component
    return bool(np.any(mask[slices][component_mask]))


def _evaluate_contour_overlaps(
    gt_components,
    pred_components,
    gt_mask,
    pred_mask,
):
    gt_hit = [
        _component_overlaps_mask(gt_component, pred_mask)
        for gt_component in gt_components
    ]
    pred_hit = [
        _component_overlaps_mask(pred_component, gt_mask)
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
    Computes object-level hit, miss and over-detection metrics.

    Each predicted contour is represented by one thresholded connected component.
    A GT object is hit if any predicted contour overlaps that GT mask component.
    A predicted contour is over-detection if it does not overlap any GT anomaly
    pixel. IoU is not used.
    """
    del hit_iou_threshold

    anomaly_segmentations = _as_numpy_stack(anomaly_segmentations)
    ground_truth_masks = _as_numpy_stack(ground_truth_masks)

    if len(anomaly_segmentations) != len(ground_truth_masks):
        raise ValueError("#Segmentations and #GT masks must match.")

    total_gt = 0
    total_pred = 0
    total_hit = 0
    total_miss = 0
    total_over = 0
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
            gt_mask,
            pred_mask,
        )

        hit = int(np.sum(gt_hit))
        miss = len(gt_components) - hit
        over = int(len(pred_hit) - np.sum(pred_hit))

        total_gt += len(gt_components)
        total_pred += len(pred_components)
        total_hit += hit
        total_miss += miss
        total_over += over

        per_image.append(
            {
                "gt_count": len(gt_components),
                "pred_count": len(pred_components),
                "hit": hit,
                "miss": miss,
                "over": over,
                "hit_rate": hit / len(gt_components) if gt_components else 0.0,
                "miss_rate": miss / len(gt_components) if gt_components else 0.0,
                "over_rate": over / len(pred_components) if pred_components else 0.0,
            }
        )

    overall_miss_rate = total_miss / total_gt if total_gt > 0 else 0.0
    return {
        "gt_total": total_gt,
        "pred_total": total_pred,
        "hit": total_hit,
        "miss": total_miss,
        "over": total_over,
        "hit_rate": total_hit / total_gt if total_gt > 0 else 0.0,
        "miss_rate": overall_miss_rate,
        "overall_miss_rate": overall_miss_rate,
        "over_rate": total_over / total_pred if total_pred > 0 else 0.0,
        "threshold": float(threshold),
        "hit_rule": "predicted_contour_overlaps_gt_mask",
        "min_component_area": min_component_area,
        "per_image": per_image,
    }
