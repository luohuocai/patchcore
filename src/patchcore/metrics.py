"""Anomaly metrics."""
import numpy as np
from scipy import ndimage
from sklearn import metrics


DEFAULT_HIT_IOU_THRESHOLD = 0.10
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

    raise ValueError("Expected a 2D mask or a mask with a singleton/channel dimension.")


def _connected_component_bboxes(mask, min_component_area):
    mask = _as_2d_mask(mask).astype(bool)
    labels, num_labels = ndimage.label(mask)
    objects = ndimage.find_objects(labels)

    bboxes = []
    for label_id, slices in enumerate(objects, start=1):
        if slices is None:
            continue

        ys, xs = slices
        area = int(np.sum(labels[slices] == label_id))
        if area < min_component_area:
            continue

        bboxes.append(
            (
                float(xs.start),
                float(ys.start),
                float(xs.stop),
                float(ys.stop),
            )
        )

    if num_labels == 0:
        return []
    return bboxes


def _intersection_area(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    return inter_w * inter_h


def _bbox_iou(a, b):
    inter = _intersection_area(a, b)
    if inter <= 0:
        return 0.0

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _evaluate_hit_miss(pred_bboxes, gt_bboxes, hit_iou_threshold):
    gt_hit = [False] * len(gt_bboxes)
    pred_hit = [False] * len(pred_bboxes)

    for gt_idx, gt_bbox in enumerate(gt_bboxes):
        for pred_bbox in pred_bboxes:
            if _bbox_iou(gt_bbox, pred_bbox) >= hit_iou_threshold:
                gt_hit[gt_idx] = True
                break

    for pred_idx, pred_bbox in enumerate(pred_bboxes):
        for gt_bbox in gt_bboxes:
            if _bbox_iou(pred_bbox, gt_bbox) >= hit_iou_threshold:
                pred_hit[pred_idx] = True
                break

    return gt_hit, pred_hit


def compute_objectwise_detection_metrics(
    anomaly_segmentations,
    ground_truth_masks,
    threshold,
    hit_iou_threshold=DEFAULT_HIT_IOU_THRESHOLD,
    min_component_area=DEFAULT_MIN_COMPONENT_AREA,
):
    """
    Computes object-level hit, miss and over-detection metrics.

    The matching rule follows the provided deep-defect batch script:
    a GT object is hit if any predicted component bbox reaches the IoU threshold,
    and a predicted object is over-detection if it matches no GT bbox.
    """
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
        pred_bboxes = _connected_component_bboxes(
            _as_2d_mask(segmentation) >= threshold,
            min_component_area=min_component_area,
        )
        gt_bboxes = _connected_component_bboxes(
            _as_2d_mask(gt_mask) > 0,
            min_component_area=min_component_area,
        )

        gt_hit, pred_hit = _evaluate_hit_miss(
            pred_bboxes,
            gt_bboxes,
            hit_iou_threshold=hit_iou_threshold,
        )

        hit = int(np.sum(gt_hit))
        miss = len(gt_bboxes) - hit
        over = int(len(pred_hit) - np.sum(pred_hit))

        total_gt += len(gt_bboxes)
        total_pred += len(pred_bboxes)
        total_hit += hit
        total_miss += miss
        total_over += over

        per_image.append(
            {
                "gt_count": len(gt_bboxes),
                "pred_count": len(pred_bboxes),
                "hit": hit,
                "miss": miss,
                "over": over,
                "hit_rate": hit / len(gt_bboxes) if gt_bboxes else 0.0,
                "miss_rate": miss / len(gt_bboxes) if gt_bboxes else 0.0,
                "over_rate": over / len(pred_bboxes) if pred_bboxes else 0.0,
            }
        )

    return {
        "gt_total": total_gt,
        "pred_total": total_pred,
        "hit": total_hit,
        "miss": total_miss,
        "over": total_over,
        "hit_rate": total_hit / total_gt if total_gt > 0 else 0.0,
        "miss_rate": total_miss / total_gt if total_gt > 0 else 0.0,
        "over_rate": total_over / total_pred if total_pred > 0 else 0.0,
        "threshold": float(threshold),
        "hit_iou_threshold": hit_iou_threshold,
        "min_component_area": min_component_area,
        "per_image": per_image,
    }
