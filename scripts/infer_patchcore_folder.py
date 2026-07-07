import argparse
import contextlib
import os
import sys

import numpy as np
import PIL.Image
import torch
from torchvision import transforms

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

import patchcore.common
import patchcore.metrics
import patchcore.patchcore
import patchcore.utils
from patchcore.datasets.image_size import channel_image_size
from patchcore.datasets.image_size import resize_crop_transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


class ImageFolderDataset(torch.utils.data.Dataset):
    def __init__(self, input_path, resize, imagesize, recursive=True):
        self.input_path = os.path.abspath(input_path)
        self.recursive = recursive
        self.image_paths = self._collect_images(input_path, recursive)
        if not self.image_paths:
            raise ValueError("No images found in {}".format(input_path))

        self.transform_img = transforms.Compose(
            [
                *resize_crop_transforms(resize, imagesize),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        self.transform_mask = transforms.Compose(
            [
                *resize_crop_transforms(resize, imagesize),
                transforms.ToTensor(),
            ]
        )
        self.transform_std = IMAGENET_STD
        self.transform_mean = IMAGENET_MEAN
        self.imagesize = channel_image_size(imagesize)

    @staticmethod
    def _collect_images(input_path, recursive):
        input_path = os.path.abspath(input_path)
        if os.path.isfile(input_path):
            ext = os.path.splitext(input_path)[1].lower()
            return [input_path] if ext in IMAGE_EXTENSIONS else []

        image_paths = []
        walker = os.walk(input_path)
        for root, _, filenames in walker:
            for filename in filenames:
                ext = os.path.splitext(filename)[1].lower()
                if ext in IMAGE_EXTENSIONS:
                    image_paths.append(os.path.join(root, filename))
            if not recursive:
                break
        return sorted(image_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)
        return {
            "image": image,
            "mask": torch.zeros([1, *image.size()[1:]]),
            "classname": "inference",
            "anomaly": "unknown",
            "is_anomaly": 0,
            "image_name": os.path.basename(image_path),
            "image_path": image_path,
        }


def load_patchcores(args, device):
    model_path = os.path.abspath(args.model_path)
    index_files = [name for name in os.listdir(model_path) if name.endswith(".faiss")]
    if not index_files:
        raise ValueError("No .faiss index found in {}".format(model_path))

    patchcores = []
    n_patchcores = len(index_files)
    for index in range(n_patchcores):
        prepend = (
            "Ensemble-{}-{}_".format(index + 1, n_patchcores)
            if n_patchcores > 1
            else ""
        )
        nn_method = patchcore.common.FaissNN(args.faiss_on_gpu, args.faiss_num_workers)
        patchcore_instance = patchcore.patchcore.PatchCore(device)
        patchcore_instance.load_from_path(
            load_path=model_path,
            device=device,
            nn_method=nn_method,
            prepend=prepend,
            score_gamma=args.score_gamma,
            fastref_enabled=args.fastref,
            fastref_lambda=args.fastref_lambda,
            fastref_iterations=args.fastref_iterations,
            fastref_sinkhorn_iterations=args.fastref_sinkhorn_iterations,
            fastref_epsilon=args.fastref_epsilon,
            fastref_ridge=args.fastref_ridge,
            fastref_chunk_size=args.fastref_chunk_size,
        )
        patchcores.append(patchcore_instance)
    return patchcores


def normalize_segmentations(segmentations):
    segmentations = np.asarray(segmentations, dtype=np.float32)
    minimum = float(np.min(segmentations))
    maximum = float(np.max(segmentations))
    if maximum <= minimum:
        return np.zeros_like(segmentations, dtype=np.float32)
    return (segmentations - minimum) / (maximum - minimum)


def aggregate_predictions(patchcores, dataloader):
    all_scores = []
    all_segmentations = []
    image_paths = dataloader.dataset.image_paths

    for patchcore_instance in patchcores:
        scores, segmentations, _, _ = patchcore_instance.predict(dataloader)
        all_scores.append(np.asarray(scores, dtype=np.float32))
        all_segmentations.append(np.asarray(segmentations, dtype=np.float32))

    if len(all_scores) == 1:
        return all_scores[0], all_segmentations[0], image_paths, False

    normalized_scores = []
    for scores in all_scores:
        score_min = np.min(scores)
        score_max = np.max(scores)
        if score_max <= score_min:
            normalized_scores.append(np.zeros_like(scores))
        else:
            normalized_scores.append((scores - score_min) / (score_max - score_min))

    normalized_segmentations = []
    for segmentations in all_segmentations:
        seg_min = np.min(segmentations)
        seg_max = np.max(segmentations)
        if seg_max <= seg_min:
            normalized_segmentations.append(np.zeros_like(segmentations))
        else:
            normalized_segmentations.append(
                (segmentations - seg_min) / (seg_max - seg_min)
            )

    return (
        np.mean(np.stack(normalized_scores), axis=0),
        np.mean(np.stack(normalized_segmentations), axis=0),
        image_paths,
        True,
    )


def _unique_extensions(first_extension):
    extensions = []
    for extension in [
        first_extension,
        ".png",
        ".bmp",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
    ]:
        if extension and extension not in extensions:
            extensions.append(extension)
    return extensions


def _candidate_mask_paths(mask_root, input_root, image_path):
    image_path = os.path.abspath(image_path)
    rel_path = os.path.basename(image_path)
    if os.path.isdir(input_root):
        rel_path = os.path.relpath(image_path, input_root)

    rel_dir = os.path.dirname(rel_path)
    image_name = os.path.basename(image_path)
    stem, extension = os.path.splitext(image_name)
    directories = [os.path.join(mask_root, rel_dir), mask_root]
    stems = [stem, "{}_mask".format(stem)]

    candidates = []
    for directory in directories:
        for candidate_stem in stems:
            for candidate_extension in _unique_extensions(extension.lower()):
                candidates.append(
                    os.path.join(directory, candidate_stem + candidate_extension)
                )
    return candidates


def _mask_lookup_keys(path):
    basename = os.path.basename(path).lower()
    stem = os.path.splitext(basename)[0]
    keys = {basename, stem}
    if stem.endswith("_mask"):
        keys.add(stem[: -len("_mask")])
    return keys


def _build_mask_lookup(mask_root, recursive):
    lookup = {}
    ambiguous = set()
    for mask_path in ImageFolderDataset._collect_images(mask_root, recursive):
        for key in _mask_lookup_keys(mask_path):
            if key in lookup and lookup[key] != mask_path:
                ambiguous.add(key)
            lookup[key] = mask_path

    for key in ambiguous:
        lookup.pop(key, None)
    return lookup


def _is_good_image_path(image_path):
    path_parts = os.path.normpath(image_path).lower().split(os.sep)
    return "good" in path_parts


def resolve_mask_paths(mask_input, input_path, image_paths, recursive=True):
    mask_input = os.path.abspath(mask_input)
    input_path = os.path.abspath(input_path)
    if os.path.isfile(mask_input):
        if len(image_paths) != 1:
            raise ValueError(
                "--mask_input points to a file, but input contains {} images.".format(
                    len(image_paths)
                )
            )
        return [mask_input]

    if not os.path.isdir(mask_input):
        raise ValueError("--mask_input does not exist: {}".format(mask_input))

    lookup = _build_mask_lookup(mask_input, recursive)
    mask_paths = []
    missing_images = []
    for image_path in image_paths:
        mask_path = None
        for candidate in _candidate_mask_paths(mask_input, input_path, image_path):
            if os.path.exists(candidate):
                mask_path = os.path.abspath(candidate)
                break

        if mask_path is None:
            image_name = os.path.basename(image_path)
            stem = os.path.splitext(image_name)[0].lower()
            mask_path = lookup.get(image_name.lower()) or lookup.get(stem)

        if mask_path is None and _is_good_image_path(image_path):
            mask_paths.append(None)
        elif mask_path is None:
            missing_images.append(image_path)
        else:
            mask_paths.append(mask_path)

    if missing_images:
        preview = "\n".join(missing_images[:5])
        raise ValueError(
            "Could not find GT masks for {} image(s). First missing:\n{}".format(
                len(missing_images), preview
            )
        )
    return mask_paths


def load_ground_truth_masks(mask_paths, dataset, target_shape):
    masks = []
    for mask_path in mask_paths:
        if mask_path is None:
            masks.append(np.zeros(target_shape, dtype=np.uint8))
            continue
        mask = PIL.Image.open(mask_path).convert("L")
        mask = dataset.transform_mask(mask).numpy()
        mask = np.squeeze(mask)
        masks.append((mask > 0).astype(np.uint8))
    return np.stack(masks)


def compute_pixel_optimal_threshold(segmentations, masks_gt):
    flat_masks = np.asarray(masks_gt).ravel()
    if len(np.unique(flat_masks)) < 2:
        raise ValueError(
            "Cannot compute a pixel-optimal threshold because GT masks do not "
            "contain both normal and anomaly pixels."
        )

    pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(
        segmentations,
        masks_gt,
    )
    return float(pixel_scores["optimal_threshold"])


def resolve_threshold_and_masks(args, dataset, image_paths, visual_segmentations):
    mask_paths = None
    if args.mask_input is not None:
        mask_paths = resolve_mask_paths(
            args.mask_input,
            args.input,
            image_paths,
            recursive=not args.no_recursive,
        )

    if args.anomaly_score_threshold is not None:
        return float(args.anomaly_score_threshold), mask_paths, "manual"

    if mask_paths is None:
        return None, None, "none"

    masks_gt = load_ground_truth_masks(
        mask_paths,
        dataset,
        target_shape=visual_segmentations.shape[-2:],
    )
    threshold = compute_pixel_optimal_threshold(visual_segmentations, masks_gt)
    return threshold, mask_paths, "pixel_optimal_f1"


def save_outputs(
    args,
    dataset,
    image_paths,
    scores,
    segmentations,
    segmentations_normalized,
):
    os.makedirs(args.output_dir, exist_ok=True)

    raw_map_dir = os.path.join(args.output_dir, "score_maps")
    index_path = patchcore.utils.save_segmentation_maps(
        raw_map_dir,
        image_paths,
        segmentations,
        scores,
    )

    visual_segmentations = (
        np.asarray(segmentations, dtype=np.float32)
        if segmentations_normalized
        else normalize_segmentations(segmentations)
    )
    anomaly_score_threshold, mask_paths, threshold_source = resolve_threshold_and_masks(
        args,
        dataset,
        image_paths,
        visual_segmentations,
    )

    if not args.skip_visualizations:
        visual_dir = os.path.join(args.output_dir, "visualizations")

        def image_transform(image):
            in_std = np.array(dataset.transform_std).reshape(-1, 1, 1)
            in_mean = np.array(dataset.transform_mean).reshape(-1, 1, 1)
            image = dataset.transform_img(image)
            return np.clip((image.numpy() * in_std + in_mean) * 255, 0, 255).astype(
                np.uint8
            )

        def mask_transform(mask):
            return dataset.transform_mask(mask).numpy()

        patchcore.utils.plot_segmentation_images(
            visual_dir,
            image_paths,
            visual_segmentations,
            scores,
            mask_paths=mask_paths,
            image_transform=image_transform,
            mask_transform=mask_transform,
            anomaly_score_threshold=anomaly_score_threshold,
        )

    print("Processed {} image(s).".format(len(image_paths)))
    print("Scores and raw score maps: {}".format(index_path))
    if anomaly_score_threshold is None:
        print("Anomaly contour threshold: none")
    else:
        print(
            "Anomaly contour threshold: {:.6f} ({})".format(
                anomaly_score_threshold,
                threshold_source,
            )
        )
    if not args.skip_visualizations:
        print("Visualizations: {}".format(os.path.join(args.output_dir, "visualizations")))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a saved PatchCore model on a folder of unlabeled images."
    )
    parser.add_argument("--model_path", required=True, help="Saved PatchCore model folder.")
    parser.add_argument("--input", required=True, help="Image file or folder to infer.")
    parser.add_argument("--output_dir", required=True, help="Folder for inference outputs.")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--resize", default="448x416")
    parser.add_argument("--imagesize", default="448x416")
    parser.add_argument("--no_recursive", action="store_true")
    parser.add_argument("--skip_visualizations", action="store_true")
    parser.add_argument("--anomaly_score_threshold", type=float, default=None)
    parser.add_argument(
        "--mask_input",
        default=None,
        help=(
            "Optional GT mask file or folder. If provided and "
            "--anomaly_score_threshold is omitted, compute the same "
            "pixel-optimal F1 threshold as bin/run_patchcore.py."
        ),
    )
    parser.add_argument("--score_gamma", type=float, default=None)
    parser.add_argument("--fastref", action="store_true", default=None)
    parser.add_argument("--fastref_lambda", type=float, default=None)
    parser.add_argument("--fastref_iterations", type=int, default=None)
    parser.add_argument("--fastref_sinkhorn_iterations", type=int, default=None)
    parser.add_argument("--fastref_epsilon", type=float, default=None)
    parser.add_argument("--fastref_ridge", type=float, default=None)
    parser.add_argument("--fastref_chunk_size", type=int, default=None)
    parser.add_argument("--faiss_on_gpu", action="store_true")
    parser.add_argument("--faiss_num_workers", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    device = patchcore.utils.set_torch_device([args.gpu])
    dataset = ImageFolderDataset(
        args.input,
        resize=args.resize,
        imagesize=args.imagesize,
        recursive=not args.no_recursive,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    patchcores = load_patchcores(args, device)
    device_context = (
        torch.cuda.device("cuda:{}".format(device.index))
        if "cuda" in device.type.lower()
        else contextlib.suppress()
    )
    with device_context:
        scores, segmentations, image_paths, segmentations_normalized = aggregate_predictions(
            patchcores, dataloader
        )

    save_outputs(
        args,
        dataset,
        image_paths,
        scores,
        segmentations,
        segmentations_normalized,
    )


if __name__ == "__main__":
    main()
