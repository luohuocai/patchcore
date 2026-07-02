from PIL import Image

from patchcore.datasets.mvtec import DatasetSplit
from patchcore.datasets.mvtec import MVTecDataset


def _save_rgb(path, size=(16, 12), color=(20, 40, 60)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path)


def _save_mask(path, size=(16, 12)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, color=255).save(path)


def test_mvtec_dataset_accepts_direct_class_root(tmp_path):
    class_root = tmp_path / "srs_ex"
    _save_rgb(class_root / "train" / "good" / "train_good.bmp")
    _save_rgb(class_root / "test" / "good" / "test_good.bmp")
    _save_rgb(class_root / "test" / "defect" / "test_bad.bmp")
    _save_mask(class_root / "ground_truth" / "defect" / "test_bad_mask.png")

    train_dataset = MVTecDataset(
        str(class_root),
        classname="srs_ex",
        resize=16,
        imagesize=16,
        split=DatasetSplit.TRAIN,
    )
    test_dataset = MVTecDataset(
        str(class_root),
        classname="srs_ex",
        resize=16,
        imagesize=16,
        split=DatasetSplit.TEST,
    )

    assert train_dataset.classnames_to_use == ["srs_ex"]
    assert len(train_dataset) == 1
    assert len(test_dataset) == 2
    assert any(
        item[1] == "defect" and item[3].endswith("_mask.png")
        for item in test_dataset.data_to_iterate
    )
