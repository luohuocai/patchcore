from patchcore.datasets.visa import DatasetSplit
from patchcore.datasets.visa import VisaDataset


_CLASSNAMES = [
    "breakfast_box",
    "juice_bottle",
    "pushpins",
    "screw_bag",
    "splicing_connectors",
]


class MVTecLOCODataset(VisaDataset):
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
        if classname is None:
            classname = "all"
        super().__init__(
            source=source,
            classname=classname,
            resize=resize,
            imagesize=imagesize,
            split=split,
            train_val_split=train_val_split,
            k_shot=k_shot,
            seed=seed,
            **kwargs,
        )
