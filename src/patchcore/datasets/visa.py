import json
import os
from enum import Enum

import PIL
import torch
from torchvision import transforms

from patchcore.datasets.image_size import channel_image_size
from patchcore.datasets.image_size import resize_crop_transforms


_CLASSNAMES = [
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class VisaDataset(torch.utils.data.Dataset):
    """PatchCore dataset wrapper for VisA meta.json format."""

    def __init__(
        self,
        source,
        classname,
        resize=732,
        imagesize=640,
        split=DatasetSplit.TRAIN,
        train_val_split=1.0,
        k_shot=-1,
        seed=10,
        **kwargs,
    ):
        super().__init__()
        self.source = source
        self.split = split
        self.train_val_split = train_val_split
        self.k_shot = k_shot
        self.seed = seed

        meta_path = os.path.join(self.source, "meta.json")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"VisA meta.json not found: {meta_path}")
        with open(meta_path, "r") as f:
            self.meta = json.load(f)

        if classname == "all":
            split_key = "test" if split == DatasetSplit.TEST else "train"
            self.classnames_to_use = sorted(self.meta[split_key].keys())
        else:
            self.classnames_to_use = [classname] if classname is not None else _CLASSNAMES

        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

        self.transform_img = [
            *resize_crop_transforms(resize, imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

        self.transform_mask = [
            *resize_crop_transforms(resize, imagesize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

        self.transform_std = IMAGENET_STD
        self.transform_mean = IMAGENET_MEAN
        self.imagesize = channel_image_size(imagesize)

    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        if self.split == DatasetSplit.TEST and mask_path is not None:
            mask = PIL.Image.open(mask_path).convert("L")
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros([1, *image.size()[1:]])

        return {
            "image": image,
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "good"),
            "image_name": os.path.join(*image_path.split(os.sep)[-4:]),
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def _resolve_path(self, relative_path):
        if not relative_path:
            return None
        return os.path.normpath(os.path.join(self.source, relative_path))

    def get_image_data(self):
        split_key = self.split.value
        if self.split == DatasetSplit.VAL:
            split_key = DatasetSplit.TRAIN.value

        imgpaths_per_class = {}
        data_to_iterate = []

        for classname in self.classnames_to_use:
            items = list(self.meta[split_key][classname])

            if self.train_val_split < 1.0 and self.split in (DatasetSplit.TRAIN, DatasetSplit.VAL):
                n_images = len(items)
                train_val_split_idx = int(n_images * self.train_val_split)
                if self.split == DatasetSplit.TRAIN:
                    items = items[:train_val_split_idx]
                else:
                    items = items[train_val_split_idx:]

            if self.split == DatasetSplit.TRAIN:
                items = [item for item in items if int(item.get("anomaly", 0)) == 0]
                if self.k_shot > 0 and len(items) > self.k_shot:
                    import random

                    random.seed(self.seed)
                    items = random.sample(items, self.k_shot)

            imgpaths_per_class[classname] = {}
            for item in items:
                image_path = self._resolve_path(item["img_path"])
                is_anomaly = int(item.get("anomaly", 0)) != 0
                anomaly = item.get("specie_name", "") if is_anomaly else "good"
                if is_anomaly and not anomaly:
                    anomaly = "anomaly"
                mask_path = self._resolve_path(item.get("mask_path", "")) if is_anomaly else None

                imgpaths_per_class[classname].setdefault(anomaly, []).append(image_path)
                data_to_iterate.append([classname, anomaly, image_path, mask_path])

        return imgpaths_per_class, data_to_iterate
