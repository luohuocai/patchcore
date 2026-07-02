import os
import subprocess
import sys
import json
import csv

# Configuration
device = "0"
# Path to datasets

data_root_srs = r"C:\Users\Administrator\Desktop\dataset\SRS"

# Output directories on D drive.
output_dir = r"D:\lhc\patchcore\red_line\output"
visualization_output_dir = r"D:\lhc\patchcore\red_line\visualizations"
# Define test configurations
# We map the datasets as "mvtec" to reuse the MVTecDataset class structure for
# SRS/microled/miniled as well.
test_configs = [
    # {"dataset": "mvtec", "path": data_root_mvtec, "class_name": "bottle"},
    #
    {"dataset": "srs", "path": data_root_srs, "class_name": "srs_squa"},
]

few_shots = [5]  # Define the number of few-shots to evaluate

resolutions = [
    {"resize": "448x416", "imagesize": "448x416"},
    # {"resize": "768x320", "imagesize": "768x320"},
    # {"resize": "1024x448", "imagesize": "1024x448"},
]

# 如果是srs_line,




backbone_name = "wideresnet50" # PatchCore default backbone
score_gamma = 1  # >1 suppresses weak patch-score responses in heatmaps.

SUMMARY_COLUMNS = [
    "dataset",
    "class_name",
    "few_shot",
    "resize",
    "imagesize",
    "score_gamma",
    "input_height",
    "input_width",
    "input_pixels",
    "input_aspect_ratio_w_h",
    "instance_auroc",
    "full_pixel_auroc",
    "anomaly_pixel_auroc",
    "pixel_optimal_threshold",
    "object_defect_count",
    "object_hit_count",
    "object_miss_count",
    "object_over_count",
    "object_hit_rate",
    "object_miss_rate",
    "overall_miss_rate",
    "object_over_rate",
    "avg_inference_time_ms_per_image",
    "peak_gpu_memory_mb",
    "num_test_images",
    "segmentation_images_path",
    "results_csv",
]


def patchcore_dataset_name(dataset_name):
    if dataset_name.lower() in ["visa", "btad", "mvtec_loco"]:
        return dataset_name
    return "mvtec"


def is_mvtec_style_root(path):
    return (
        os.path.isdir(os.path.join(path, "train", "good"))
        and os.path.isdir(os.path.join(path, "test"))
    )


def class_root_path(data_root, class_name):
    if is_mvtec_style_root(data_root):
        return data_root
    return os.path.join(data_root, class_name)


def is_mvtec_style_class(data_root, class_name):
    return is_mvtec_style_root(class_root_path(data_root, class_name))


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

    if is_mvtec_style_root(data_root):
        return [os.path.basename(os.path.normpath(data_root))]

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


def validate_mvtec_style_dataset(data_root, classes, few_shot):
    for class_name in classes:
        class_root = class_root_path(data_root, class_name)
        if not is_mvtec_style_class(data_root, class_name):
            raise RuntimeError(
                "Expected MVTec-style structure for class "
                f"'{class_name}', but did not find train/good and test folders "
                f"under {class_root}."
            )

        train_good_dir = os.path.join(class_root, "train", "good")
        test_good_dir = os.path.join(class_root, "test", "good")
        test_defect_dir = os.path.join(class_root, "test", "defect")
        train_good_count = count_files(train_good_dir)
        test_good_count = count_files(test_good_dir)
        test_defect_count = count_files(test_defect_dir)

        print(
            f"SRS/MVTec-style class={class_name}: "
            f"train/good={train_good_count}, "
            f"test/good={test_good_count}, "
            f"test/defect={test_defect_count}, "
            f"k_shot={few_shot}"
        )

        if train_good_count < few_shot:
            raise RuntimeError(
                f"PatchCore {few_shot}-shot training requires at least "
                f"{few_shot} normal samples in {train_good_dir}, but found "
                f"{train_good_count}."
            )


def find_latest_results_csv(root_dir):
    candidates = []
    for current_root, _, files in os.walk(root_dir):
        if "results.csv" in files:
            path = os.path.join(current_root, "results.csv")
            candidates.append(path)
    if not candidates:
        raise RuntimeError(f"No results.csv found under {root_dir}.")
    return max(candidates, key=os.path.getmtime)


def read_mean_metrics(results_csv):
    with open(results_csv, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No result rows found in {results_csv}.")

    for row in rows:
        if row.get("Row Names") == "Mean":
            return row
    return rows[-1]


def resolution_folder_name(imagesize):
    imagesize = str(imagesize)
    if "x" in imagesize.lower() or "*" in imagesize:
        return imagesize.lower().replace("*", "x")
    return f"{imagesize}x{imagesize}"


def parse_input_size(imagesize):
    imagesize = str(imagesize).strip().lower().replace("*", "x").replace(",", "x")
    if "x" in imagesize:
        height, width = imagesize.split("x", 1)
        return int(height), int(width)
    size = int(imagesize)
    return size, size


def input_size_summary(imagesize):
    height, width = parse_input_size(imagesize)
    return {
        "input_height": height,
        "input_width": width,
        "input_pixels": height * width,
        "input_aspect_ratio_w_h": round(width / height, 6),
    }


def summary_csv_filename(resolutions):
    resolution_names = []
    for resolution in resolutions:
        resolution_name = resolution_folder_name(resolution["imagesize"])
        if resolution_name not in resolution_names:
            resolution_names.append(resolution_name)

    suffix = "_".join(resolution_names)
    if suffix:
        return f"srs_few_shot_resolution_summary_{suffix}.csv"
    return "srs_few_shot_resolution_summary.csv"


def write_summary_csv(summary_path, rows):
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in SUMMARY_COLUMNS})


summary_rows = []


for config in test_configs:
    test_dataset = config["dataset"]
    data_root = config["path"]
    target_class = config["class_name"]

    # Check if data root exists
    if not os.path.exists(data_root):
        print(f"Error: Data root not found at {os.path.abspath(data_root)} for dataset {test_dataset}")
        continue

    for resolution in resolutions:
        resize = resolution["resize"]
        imagesize = resolution["imagesize"]
        resolution_name = resolution_folder_name(imagesize)
        input_summary = input_size_summary(imagesize)

        for few_shot in few_shots:
            classes = get_subdatasets(test_dataset, data_root, target_class)
            if patchcore_dataset_name(test_dataset) == "mvtec":
                validate_mvtec_style_dataset(data_root, classes, few_shot)

            # Setup the paths
            save_dir = os.path.join(
                output_dir,
                test_dataset,
                resolution_name,
                f"few_shot_{few_shot}",
            )
            segmentation_images_path = os.path.join(
                visualization_output_dir,
                test_dataset,
                resolution_name,
                f"few_shot_{few_shot}",
            )
            os.makedirs(save_dir, exist_ok=True)
            os.makedirs(segmentation_images_path, exist_ok=True)

            # Construct the command using click chaining syntax for patchcore
            cmd = [
                sys.executable, "bin/run_patchcore.py",
                "--gpu", str(device),
                "--seed", "0",
                "--save_patchcore_model",
                "--save_segmentation_images",
                "--segmentation_images_path", segmentation_images_path,
                "--log_project", f"few_shot_summary_{resolution_name}",
                "--log_group", f"shot_{few_shot}",
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
                "--num_workers", "0",
                "--resize", str(resize),
                "--imagesize", str(imagesize),
                "--k_shot", str(few_shot) # Pass the k_shot parameter here
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
            print(
                "Running PatchCore for "
                f"dataset={test_dataset}, class={target_class}, "
                f"k_shot={few_shot}, imagesize={resolution_name}..."
            )
            print(f"{'='*50}\n")

            try:
                subprocess.run(cmd, env=env, check=True)
                results_csv = find_latest_results_csv(save_dir)
                mean_metrics = read_mean_metrics(results_csv)
                summary_rows.append(
                    {
                        "dataset": test_dataset,
                        "class_name": target_class,
                        "few_shot": few_shot,
                        "resize": resize,
                        "imagesize": imagesize,
                        "score_gamma": score_gamma,
                        **input_summary,
                        "segmentation_images_path": os.path.join(
                            segmentation_images_path,
                            f"{patchcore_dataset_name(test_dataset)}_{target_class}",
                            resolution_name,
                        ),
                        "results_csv": results_csv,
                        **mean_metrics,
                    }
                )
            except subprocess.CalledProcessError as e:
                print(
                    "Error running command for "
                    f"{target_class} ({few_shot}-shot, {resolution_name}): {e}"
                )


summary_csv = os.path.join(output_dir, summary_csv_filename(resolutions))
write_summary_csv(summary_csv, summary_rows)
print(f"\nSaved few-shot summary CSV to: {summary_csv}")
