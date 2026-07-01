import os
import subprocess
import sys
import json

# Configuration
device = "0"
# Path to datasets
data_root_mvtec = "../../data/mvtec_anomaly_detection"
data_root_visa = r"C:\Users\Administrator\Desktop\dataset\visa"
data_root_btad = r"C:\Users\Administrator\Desktop\dataset\BTech_Dataset_transformed"
data_root_mvtec_loco = r"C:\Users\Administrator\Desktop\dataset\MVTec_loco"
data_root_microled = "../../data/microled_AD"
data_root_miniled = "../../data/miniled_AD"
data_root_srs = r"C:\Users\Administrator\Desktop\dataset\SRS"

# Output directory (Project root / output / patchcore)
output_dir = "../../output/PatchCore"

# Define test configurations
# Note: PatchCore natively uses the "run_patchcore.py" script which heavily relies on the "click" library.
# We map the datasets as "mvtec" to reuse the MVTecDataset class structure for
# SRS/microled/miniled as well.
test_configs = [
    # {"dataset": "mvtec", "path": data_root_mvtec, "class_name": "transistor"}, 
    # {"dataset": "visa", "path": data_root_visa, "class_name": "all"},
    # {"dataset": "btad", "path": data_root_btad, "class_name": "all"},
    # {"dataset": "mvtec_loco", "path": data_root_mvtec_loco, "class_name": "all"},
    # {"dataset": "microled", "path": data_root_microled, "class_name": "all"},
    # {"dataset": "miniled", "path": data_root_miniled, "class_name": "all"},
    {"dataset": "srs", "path": data_root_srs, "class_name": "srs_cropped"},
]

# We map all to "mvtec" format in patchcore, but pass the different paths.
# PatchCore takes multiple commands chained together in click.
# The general structure is: python bin/run_patchcore.py <save_path> [options] patch_core [options] sampler [options] dataset [options]

backbone_name = "wideresnet50" # PatchCore default backbone
score_gamma = 1.5  # >1 suppresses weak patch-score responses in heatmaps.


def patchcore_dataset_name(dataset_name):
    if dataset_name.lower() in ["visa", "btad", "mvtec_loco"]:
        return dataset_name
    return "mvtec"


def is_mvtec_style_class(data_root, class_name):
    class_root = os.path.join(data_root, class_name)
    return (
        os.path.isdir(os.path.join(class_root, "train", "good"))
        and os.path.isdir(os.path.join(class_root, "test"))
    )


def count_files(folder):
    if not os.path.isdir(folder):
        return 0
    return sum(
        1
        for name in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, name))
    )


def get_subdatasets(test_dataset, data_root, target_class):
    if target_class.lower() != "all":
        return [target_class]

    meta_path = os.path.join(data_root, "meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path, "r") as f:
            return sorted(json.load(f)["test"].keys())

    classes = [
        d
        for d in sorted(os.listdir(data_root))
        if os.path.isdir(os.path.join(data_root, d))
        and is_mvtec_style_class(data_root, d)
    ]
    if not classes:
        raise RuntimeError(
            f"No MVTec-style classes found under {data_root} for {test_dataset}."
        )
    return classes


def validate_mvtec_style_dataset(data_root, classes):
    for class_name in classes:
        if not is_mvtec_style_class(data_root, class_name):
            raise RuntimeError(
                "Expected MVTec-style structure for class "
                f"'{class_name}', but did not find train/good and test folders "
                f"under {os.path.join(data_root, class_name)}."
            )

        train_good_dir = os.path.join(data_root, class_name, "train", "good")
        test_good_dir = os.path.join(data_root, class_name, "test", "good")
        test_defect_dir = os.path.join(data_root, class_name, "test", "defect")
        train_good_count = count_files(train_good_dir)
        test_good_count = count_files(test_good_dir)
        test_defect_count = count_files(test_defect_dir)

        print(
            f"SRS/MVTec-style class={class_name}: "
            f"train/good={train_good_count}, "
            f"test/good={test_good_count}, "
            f"test/defect={test_defect_count}"
        )

        if train_good_count == 0:
            raise RuntimeError(
                f"PatchCore full-shot training requires normal samples in "
                f"{train_good_dir}, but found 0."
            )

for config in test_configs:
    test_dataset = config["dataset"]
    data_root = config["path"]
    target_class = config["class_name"]

    # Check if data root exists
    if not os.path.exists(data_root):
        print(f"Error: Data root not found at {os.path.abspath(data_root)} for dataset {test_dataset}")
        continue

    classes = get_subdatasets(test_dataset, data_root, target_class)
    if patchcore_dataset_name(test_dataset) == "mvtec":
        validate_mvtec_style_dataset(data_root, classes)

    # Setup the paths
    save_dir = os.path.join(output_dir, test_dataset)
    os.makedirs(save_dir, exist_ok=True)
    
    # Construct the command using click chaining syntax for patchcore
    cmd = [
        sys.executable, "bin/run_patchcore.py",
        "--gpu", str(device),
        "--seed", "0",
        "--save_patchcore_model",
        "--save_segmentation_images",
        save_dir,
        
        "patch_core",
        "-b", backbone_name,
        "-le", "layer2", "-le", "layer3",  # default patchcore layers for wide_resnet50
        # "--faiss_on_gpu",
        "--pretrain_embed_dimension", "1024",
        "--target_embed_dimension", "1024",
        "--anomaly_scorer_num_nn", "1",
        "--patchsize", "3",
        "--score_gamma", str(score_gamma),
        
        "sampler",
        "-p", "0.1", # default coreset sampling ratio (10%)
        "approx_greedy_coreset",
        
        "dataset",
        "--resize", "640x1152",
        "--imagesize", "640x1152"
    ]
    
    # Subdatasets argument
    for c in classes:
        cmd.extend(["-d", c])

    cmd.extend([
        patchcore_dataset_name(test_dataset),
        data_root
    ])
    
    # Set environment variable to ensure correct working directory for imports
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')) + os.pathsep + env.get("PYTHONPATH", "")
    
    print(f"\n{'='*50}")
    print(f"Running PatchCore for dataset={test_dataset}, class={target_class}...")
    print(f"{'='*50}\n")
    
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command for {target_class}: {e}")
