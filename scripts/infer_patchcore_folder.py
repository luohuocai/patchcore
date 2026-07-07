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
import patchcore.patchcore
import patchcore.utils
from patchcore.datasets.image_size import channel_image_size
from patchcore.datasets.image_size import resize_crop_transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


class ImageFolderDataset(torch.utils.data.Dataset):
    def __init__(self, input_path, resize, imagesize, recursive=True):
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


def normalize_map(score_map):
    score_map = np.asarray(score_map, dtype=np.float32)
    minimum = float(np.min(score_map))
    maximum = float(np.max(score_map))
    if maximum <= minimum:
        return np.zeros_like(score_map, dtype=np.float32)
    return (score_map - minimum) / (maximum - minimum)


def aggregate_predictions(patchcores, dataloader):
    all_scores = []
    all_segmentations = []
    image_paths = dataloader.dataset.image_paths

    for patchcore_instance in patchcores:
        scores, segmentations, _, _ = patchcore_instance.predict(dataloader)
        all_scores.append(np.asarray(scores, dtype=np.float32))
        all_segmentations.append(np.asarray(segmentations, dtype=np.float32))

    if len(all_scores) == 1:
        return all_scores[0], all_segmentations[0], image_paths

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
    )


def save_outputs(args, dataset, image_paths, scores, segmentations):
    os.makedirs(args.output_dir, exist_ok=True)

    raw_map_dir = os.path.join(args.output_dir, "score_maps")
    index_path = patchcore.utils.save_segmentation_maps(
        raw_map_dir,
        image_paths,
        segmentations,
        scores,
    )

    normalized_segmentations = [normalize_map(segmentation) for segmentation in segmentations]
    if not args.skip_visualizations:
        visual_dir = os.path.join(args.output_dir, "visualizations")

        def image_transform(image):
            in_std = np.array(dataset.transform_std).reshape(-1, 1, 1)
            in_mean = np.array(dataset.transform_mean).reshape(-1, 1, 1)
            image = dataset.transform_img(image)
            return np.clip((image.numpy() * in_std + in_mean) * 255, 0, 255).astype(
                np.uint8
            )

        patchcore.utils.plot_segmentation_images(
            visual_dir,
            image_paths,
            normalized_segmentations,
            scores,
            mask_paths=None,
            image_transform=image_transform,
            anomaly_score_threshold=args.anomaly_score_threshold,
        )

    print("Processed {} image(s).".format(len(image_paths)))
    print("Scores and raw score maps: {}".format(index_path))
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
        scores, segmentations, image_paths = aggregate_predictions(patchcores, dataloader)

    save_outputs(args, dataset, image_paths, scores, segmentations)


if __name__ == "__main__":
    main()
