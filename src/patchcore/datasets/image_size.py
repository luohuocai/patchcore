import re

from torchvision import transforms


def parse_size(size, name="size"):
    """Parse a size as an int or (height, width) tuple.

    Accepted rectangular forms include "640x1152", "640*1152", and
    "640,1152". A single value keeps the original square-crop behavior.
    """
    if isinstance(size, int):
        if size <= 0:
            raise ValueError(f"{name} must be positive, got {size}.")
        return size

    if isinstance(size, (tuple, list)):
        if len(size) == 1:
            return parse_size(size[0], name)
        if len(size) == 2:
            height, width = (int(size[0]), int(size[1]))
            if height <= 0 or width <= 0:
                raise ValueError(f"{name} must be positive, got {size}.")
            return (height, width)
        raise ValueError(f"{name} must have one or two values, got {size}.")

    if isinstance(size, str):
        value = size.strip().lower()
        value = (
            value.replace("(", "")
            .replace(")", "")
            .replace("[", "")
            .replace("]", "")
            .replace("*", "x")
            .replace(",", "x")
        )
        parts = [part for part in re.split(r"[x\s]+", value) if part]
        if len(parts) == 1:
            return parse_size(int(parts[0]), name)
        if len(parts) == 2:
            return parse_size((int(parts[0]), int(parts[1])), name)

    raise ValueError(
        f"{name} must be an int or HxW string like '640x1152', got {size!r}."
    )


def resize_crop_transforms(resize, imagesize):
    resize = parse_size(resize, "resize")
    imagesize = parse_size(imagesize, "imagesize")

    if isinstance(imagesize, tuple):
        return [transforms.Resize(imagesize)]

    return [
        transforms.Resize(resize),
        transforms.CenterCrop(imagesize),
    ]


def channel_image_size(imagesize):
    imagesize = parse_size(imagesize, "imagesize")
    if isinstance(imagesize, tuple):
        height, width = imagesize
    else:
        height = width = imagesize
    return (3, height, width)
