import csv
import logging
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import tqdm

LOGGER = logging.getLogger(__name__)


def _as_numpy(value):
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _to_display_image(image):
    image = _as_numpy(image)
    image = np.squeeze(image)
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = image.transpose(1, 2, 0)
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[:, :, 0]
    return image


def _to_score_map(segmentation):
    segmentation = np.asarray(segmentation)
    segmentation = np.squeeze(segmentation)
    if segmentation.ndim == 2:
        return segmentation
    if segmentation.ndim == 3 and segmentation.shape[0] in (1, 3):
        return np.max(segmentation, axis=0)
    if segmentation.ndim == 3 and segmentation.shape[-1] in (1, 3):
        return np.max(segmentation, axis=-1)
    raise ValueError("Expected a 2D anomaly score map.")


def _thresholded_anomaly_mask(score_map, threshold):
    return _to_score_map(score_map) >= threshold


def _padded_contour_grid(binary_mask):
    height, width = binary_mask.shape
    padded_mask = np.pad(binary_mask.astype(np.float32), 1, constant_values=0.0)
    x_coords = np.arange(-1, width + 1)
    y_coords = np.arange(-1, height + 1)
    return x_coords, y_coords, padded_mask


def _compute_anomaly_contours(score_map, threshold):
    binary_mask = _thresholded_anomaly_mask(score_map, threshold)
    if not np.any(binary_mask):
        return []

    height, width = binary_mask.shape
    x_coords, y_coords, padded_mask = _padded_contour_grid(binary_mask)
    figure, axis = plt.subplots()
    contour_set = axis.contour(x_coords, y_coords, padded_mask, levels=[0.5])
    contours = []
    for segment in contour_set.allsegs[0]:
        contour = np.stack([segment[:, 1], segment[:, 0]], axis=1)
        contour[:, 0] = np.clip(contour[:, 0], 0, height - 1)
        contour[:, 1] = np.clip(contour[:, 1], 0, width - 1)
        contours.append(contour)
    plt.close(figure)
    return contours


def _draw_anomaly_contours(axis, score_map, threshold, color="red", linewidth=1.2):
    score_map = _to_score_map(score_map)
    height, width = score_map.shape
    binary_mask = _thresholded_anomaly_mask(score_map, threshold)
    if np.any(binary_mask):
        x_coords, y_coords, padded_mask = _padded_contour_grid(binary_mask)
        axis.contour(
            x_coords,
            y_coords,
            padded_mask,
            levels=[0.5],
            colors=[color],
            linewidths=linewidth,
        )
    axis.set_xlim(-0.5, width - 0.5)
    axis.set_ylim(height - 0.5, -0.5)


def plot_segmentation_images(
    savefolder,
    image_paths,
    segmentations,
    anomaly_scores=None,
    mask_paths=None,
    image_transform=lambda x: x,
    mask_transform=lambda x: x,
    anomaly_score_threshold=None,
    save_depth=4,
):
    """Generate anomaly segmentation images.

    Args:
        image_paths: List[str] List of paths to images.
        segmentations: [List[np.ndarray]] Generated anomaly segmentations.
        anomaly_scores: [List[float]] Anomaly scores for each image.
        mask_paths: [List[str]] List of paths to ground truth masks.
        image_transform: [function or lambda] Optional transformation of images.
        mask_transform: [function or lambda] Optional transformation of masks.
        anomaly_score_threshold: [float or None] Optional threshold used to draw
            anomaly contours on the response map and input image.
        save_depth: [int] Number of path-strings to use for image savenames.
    """
    if mask_paths is None:
        mask_paths = ["-1" for _ in range(len(image_paths))]
    masks_provided = mask_paths[0] != "-1"
    if anomaly_scores is None:
        anomaly_scores = ["-1" for _ in range(len(image_paths))]

    os.makedirs(savefolder, exist_ok=True)

    for image_path, mask_path, anomaly_score, segmentation in tqdm.tqdm(
        zip(image_paths, mask_paths, anomaly_scores, segmentations),
        total=len(image_paths),
        desc="Generating Segmentation Images...",
        leave=False,
    ):
        image = PIL.Image.open(image_path).convert("RGB")
        image = image_transform(image)
        image = _to_display_image(image)

        if masks_provided:
            if mask_path is not None:
                mask = PIL.Image.open(mask_path).convert("RGB")
                mask = mask_transform(mask)
                mask = _to_display_image(mask)
            else:
                mask = np.zeros(image.shape[:2], dtype=np.uint8)

        segmentation = _to_score_map(segmentation)

        savename = os.path.normpath(image_path).split(os.sep)
        savename = "_".join(savename[-save_depth:])
        savename = os.path.splitext(savename)[0] + ".png"
        savename = os.path.join(savefolder, savename)
        plot_count = 2 + int(masks_provided) + int(anomaly_score_threshold is not None)
        f, axes = plt.subplots(1, plot_count)
        axes = np.asarray(axes).reshape(-1)

        axis_idx = 0
        axes[axis_idx].imshow(image)
        axes[axis_idx].set_title("Image")
        axis_idx += 1

        if masks_provided:
            axes[axis_idx].imshow(mask, cmap="gray")
            axes[axis_idx].set_title("Ground Truth")
            axis_idx += 1

        axes[axis_idx].imshow(segmentation, cmap="jet", vmin=0.0, vmax=1.0)
        axes[axis_idx].set_title("Anomaly Response")
        if anomaly_score_threshold is not None:
            _draw_anomaly_contours(
                axes[axis_idx],
                segmentation,
                anomaly_score_threshold,
            )
        axis_idx += 1

        if anomaly_score_threshold is not None:
            axes[axis_idx].imshow(image)
            axes[axis_idx].set_title("Anomaly Contour")
            _draw_anomaly_contours(
                axes[axis_idx],
                segmentation,
                anomaly_score_threshold,
            )

        for axis in axes:
            axis.axis("off")

        f.set_size_inches(3 * plot_count, 3)
        f.tight_layout()
        f.savefig(savename)
        plt.close()


def create_storage_folder(
    main_folder_path,
    project_folder,
    group_folder,
    mode="iterate",
    resolution_folder=None,
):
    os.makedirs(main_folder_path, exist_ok=True)
    project_path = os.path.join(main_folder_path, project_folder)
    if resolution_folder is not None:
        project_path = os.path.join(project_path, resolution_folder)
    os.makedirs(project_path, exist_ok=True)
    save_path = os.path.join(project_path, group_folder)
    if mode == "iterate":
        counter = 0
        while os.path.exists(save_path):
            save_path = os.path.join(project_path, group_folder + "_" + str(counter))
            counter += 1
        os.makedirs(save_path)
    elif mode == "overwrite":
        os.makedirs(save_path, exist_ok=True)

    return save_path


def resolution_folder_name(imagesize):
    height, width = imagesize[-2:]
    return "{}x{}".format(height, width)


def set_torch_device(gpu_ids):
    """Returns correct torch.device.

    Args:
        gpu_ids: [list] list of gpu ids. If empty, cpu is used.
    """
    if len(gpu_ids):
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        return torch.device("cuda:{}".format(gpu_ids[0]))
    return torch.device("cpu")


def fix_seeds(seed, with_torch=True, with_cuda=True):
    """Fixed available seeds for reproducibility.

    Args:
        seed: [int] Seed value.
        with_torch: Flag. If true, torch-related seeds are fixed.
        with_cuda: Flag. If true, torch+cuda-related seeds are fixed
    """
    random.seed(seed)
    np.random.seed(seed)
    if with_torch:
        torch.manual_seed(seed)
    if with_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def compute_and_store_final_results(
    results_path,
    results,
    row_names=None,
    column_names=[
        "Instance AUROC",
        "Full Pixel AUROC",
        "Full PRO",
        "Anomaly Pixel AUROC",
        "Anomaly PRO",
    ],
):
    """Store computed results as CSV file.

    Args:
        results_path: [str] Where to store result csv.
        results: [List[List]] List of lists containing results per dataset,
                 with results[i][0] == 'dataset_name' and results[i][1:6] =
                 [instance_auroc, full_pixelwisew_auroc, full_pro,
                 anomaly-only_pw_auroc, anomaly-only_pro]
    """
    if row_names is not None:
        assert len(row_names) == len(results), "#Rownames != #Result-rows."

    mean_metrics = {}
    for i, result_key in enumerate(column_names):
        mean_metrics[result_key] = np.mean([x[i] for x in results])
        LOGGER.info("{0}: {1:3.3f}".format(result_key, mean_metrics[result_key]))

    savename = os.path.join(results_path, "results.csv")
    with open(savename, "w") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=",")
        header = column_names
        if row_names is not None:
            header = ["Row Names"] + header

        csv_writer.writerow(header)
        for i, result_list in enumerate(results):
            csv_row = result_list
            if row_names is not None:
                csv_row = [row_names[i]] + result_list
            csv_writer.writerow(csv_row)
        mean_scores = list(mean_metrics.values())
        if row_names is not None:
            mean_scores = ["Mean"] + mean_scores
        csv_writer.writerow(mean_scores)

    mean_metrics = {"mean_{0}".format(key): item for key, item in mean_metrics.items()}
    return mean_metrics
