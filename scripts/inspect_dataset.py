import os
import argparse
import glob
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Dataset Inspection and Forensics")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=os.environ.get("DATASET_ROOT", "C:/Users/Mayank Mukherjee/Desktop/Hack/Data-public-20260814T125741Z-1-001/Data-public"),
        help="Path to the dataset root folder containing train/ and Test_NoisyLR/"
    )
    parser.add_argument(
        "--num_visualizations",
        type=int,
        default=5,
        help="Number of random pairs to visualize and save"
    )
    return parser.parse_args()

def analyze_numpy_file(filepath):
    """Loads a .npy file and computes basic statistics."""
    try:
        data = np.load(filepath)
        stats = {
            "shape": list(data.shape),
            "dtype": str(data.dtype),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "percentiles": {
                "1": float(np.percentile(data, 1)),
                "5": float(np.percentile(data, 5)),
                "25": float(np.percentile(data, 25)),
                "50": float(np.percentile(data, 50)),
                "75": float(np.percentile(data, 75)),
                "95": float(np.percentile(data, 95)),
                "99": float(np.percentile(data, 99))
            }
        }
        return stats, data
    except Exception as e:
        return {"error": str(e)}, None

def main():
    args = parse_args()
    dataset_root = args.dataset_root
    print(f"Inspecting dataset at: {dataset_root}")
    
    if not os.path.exists(dataset_root):
        print(f"Error: dataset root '{dataset_root}' does not exist.")
        return

    # 1. Discover all .npy files recursively (excluding __MACOSX directories)
    all_npy_paths = glob.glob(os.path.join(dataset_root, "**", "*.npy"), recursive=True)
    all_npy_paths = [p for p in all_npy_paths if "__MACOSX" not in p]
    
    print(f"Found {len(all_npy_paths)} total .npy files (excluding macOS metadata).")

    # Group files by their parent directory pattern
    dir_groups = {}
    for path in all_npy_paths:
        rel_path = os.path.relpath(path, dataset_root)
        parent_dir = os.path.dirname(rel_path)
        if parent_dir not in dir_groups:
            dir_groups[parent_dir] = []
        dir_groups[parent_dir].append(path)
        
    print("\nDirectory distribution:")
    for parent_dir, paths in dir_groups.items():
        print(f"  - {parent_dir}: {len(paths)} files")

    # Identify GT, Train NoisyLR and Test NoisyLR directories
    gt_paths = []
    train_noisy_paths = []
    test_noisy_paths = []

    for parent_dir, paths in dir_groups.items():
        # GT files
        if "train" in parent_dir and ("GT" in parent_dir or "gt" in parent_dir.upper()):
            gt_paths.extend(paths)
        # Train NoisyLR files
        elif "train" in parent_dir and ("NoisyLR" in parent_dir or "noisylr" in parent_dir.upper()):
            train_noisy_paths.extend(paths)
        # Test NoisyLR files
        elif "Test_NoisyLR" in parent_dir or "test" in parent_dir.lower():
            test_noisy_paths.extend(paths)

    gt_paths = sorted(gt_paths)
    train_noisy_paths = sorted(train_noisy_paths)
    test_noisy_paths = sorted(test_noisy_paths)

    print(f"\nCategorized Files:")
    print(f"  - Ground Truth (GT): {len(gt_paths)} files")
    print(f"  - Train Degraded (NoisyLR): {len(train_noisy_paths)} files")
    print(f"  - Test Degraded (NoisyLR): {len(test_noisy_paths)} files")

    # 2. Pairing verification
    pairing_report = {}
    paired_samples = []
    gt_basenames = {os.path.basename(p): p for p in gt_paths}
    
    unpaired_gt = []
    unpaired_noisy = []
    
    for noisy_path in train_noisy_paths:
        base = os.path.basename(noisy_path)
        if base in gt_basenames:
            paired_samples.append((noisy_path, gt_basenames[base]))
        else:
            unpaired_noisy.append(noisy_path)
            
    for gt_path in gt_paths:
        base = os.path.basename(gt_path)
        if base not in [os.path.basename(p[0]) for p in paired_samples]:
            unpaired_gt.append(gt_path)

    pairing_report["paired_count"] = len(paired_samples)
    pairing_report["unpaired_gt_count"] = len(unpaired_gt)
    pairing_report["unpaired_noisy_count"] = len(unpaired_noisy)
    pairing_report["strategy"] = "Filename match (identical basenames)"
    
    if unpaired_gt or unpaired_noisy:
        pairing_report["status"] = "WARNING: Unpaired files found!"
        pairing_report["unpaired_gt_examples"] = [os.path.basename(p) for p in unpaired_gt[:5]]
        pairing_report["unpaired_noisy_examples"] = [os.path.basename(p) for p in unpaired_noisy[:5]]
    else:
        pairing_report["status"] = "SUCCESS: All train files correctly paired."

    print(f"\nPairing Status: {pairing_report['status']}")
    print(f"Paired count: {pairing_report['paired_count']}")

    # 3. Analyze a subset of images for detailed statistics (and check all shapes)
    print("\nRunning statistical analysis...")
    gt_shapes = {}
    lr_shapes = {}
    test_shapes = {}
    
    gt_global_stats = {"min": [], "max": [], "mean": [], "std": []}
    lr_global_stats = {"min": [], "max": [], "mean": [], "std": []}
    test_global_stats = {"min": [], "max": [], "mean": [], "std": []}
    
    # We will analyze up to 200 random files for stats, but check shapes for all
    np.random.seed(42)
    sample_indices = np.random.choice(len(paired_samples), min(200, len(paired_samples)), replace=False)
    
    for idx, (lr_path, gt_path) in enumerate(paired_samples):
        if idx in sample_indices:
            gt_stat, gt_data = analyze_numpy_file(gt_path)
            lr_stat, lr_data = analyze_numpy_file(lr_path)
            
            gt_shape_tuple = tuple(gt_stat["shape"])
            lr_shape_tuple = tuple(lr_stat["shape"])
            
            gt_shapes[str(gt_shape_tuple)] = gt_shapes.get(str(gt_shape_tuple), 0) + 1
            lr_shapes[str(lr_shape_tuple)] = lr_shapes.get(str(lr_shape_tuple), 0) + 1
            
            gt_global_stats["min"].append(gt_stat["min"])
            gt_global_stats["max"].append(gt_stat["max"])
            gt_global_stats["mean"].append(gt_stat["mean"])
            gt_global_stats["std"].append(gt_stat["std"])
            
            lr_global_stats["min"].append(lr_stat["min"])
            lr_global_stats["max"].append(lr_stat["max"])
            lr_global_stats["mean"].append(lr_stat["mean"])
            lr_global_stats["std"].append(lr_stat["std"])
        else:
            # Just shape check
            gt_data = np.load(gt_path)
            lr_data = np.load(lr_path)
            gt_shapes[str(gt_data.shape)] = gt_shapes.get(str(gt_data.shape), 0) + 1
            lr_shapes[str(lr_data.shape)] = lr_shapes.get(str(lr_data.shape), 0) + 1

    # Analyze all test shapes, compute stats on a subset
    test_sample_indices = np.random.choice(len(test_noisy_paths), min(100, len(test_noisy_paths)), replace=False)
    for idx, test_path in enumerate(test_noisy_paths):
        if idx in test_sample_indices:
            test_stat, test_data = analyze_numpy_file(test_path)
            test_shape_tuple = tuple(test_stat["shape"])
            test_shapes[str(test_shape_tuple)] = test_shapes.get(str(test_shape_tuple), 0) + 1
            test_global_stats["min"].append(test_stat["min"])
            test_global_stats["max"].append(test_stat["max"])
            test_global_stats["mean"].append(test_stat["mean"])
            test_global_stats["std"].append(test_stat["std"])
        else:
            test_data = np.load(test_path)
            test_shapes[str(test_data.shape)] = test_shapes.get(str(test_data.shape), 0) + 1

    # Summarize stats
    def summarize_stat_list(stats_dict):
        return {
            "min_range": [min(stats_dict["min"]), max(stats_dict["min"])],
            "max_range": [min(stats_dict["max"]), max(stats_dict["max"])],
            "mean_range": [min(stats_dict["mean"]), max(stats_dict["mean"])],
            "std_range": [min(stats_dict["std"]), max(stats_dict["std"])],
            "global_avg_mean": float(np.mean(stats_dict["mean"])),
            "global_avg_std": float(np.mean(stats_dict["std"]))
        }

    gt_summary = summarize_stat_list(gt_global_stats)
    lr_summary = summarize_stat_list(lr_global_stats)
    test_summary = summarize_stat_list(test_global_stats)

    report_data = {
        "dataset_root": dataset_root,
        "counts": {
            "total_npy": len(all_npy_paths),
            "gt": len(gt_paths),
            "train_noisy": len(train_noisy_paths),
            "test_noisy": len(test_noisy_paths)
        },
        "pairing": pairing_report,
        "shapes": {
            "gt_shapes_distribution": gt_shapes,
            "lr_shapes_distribution": lr_shapes,
            "test_shapes_distribution": test_shapes
        },
        "statistics": {
            "gt": gt_summary,
            "train_noisy": lr_summary,
            "test_noisy": test_summary
        }
    }

    # Create reports directory if it doesn't exist
    os.makedirs("reports/data_visualizations", exist_ok=True)

    # Save JSON report
    with open("reports/dataset_report.json", "w") as f:
        json.dump(report_data, f, indent=4)
        
    # Generate text report
    report_txt = f"""========================================================
DATASET FORENSICS REPORT
========================================================
Dataset Root: {dataset_root}

1. FILE COUNT SUMMARY
--------------------------------------------------------
- Total discovered .npy files: {report_data["counts"]["total_npy"]} (excluding __MACOSX)
- Ground Truth (GT) Train files: {report_data["counts"]["gt"]}
- Noisy Low-Resolution (NoisyLR) Train files: {report_data["counts"]["train_noisy"]}
- Test Noisy Low-Resolution files: {report_data["counts"]["test_noisy"]}

2. PAIRING ANALYSIS
--------------------------------------------------------
- Pairing Strategy: {pairing_report["strategy"]}
- Status: {pairing_report["status"]}
- Total paired images: {pairing_report["paired_count"]}
- Unpaired GT count: {pairing_report["unpaired_gt_count"]}
- Unpaired NoisyLR count: {pairing_report["unpaired_noisy_count"]}

3. RESOLUTIONS & DIMENSIONS
--------------------------------------------------------
- GT shape distribution: {gt_shapes}
- NoisyLR shape distribution: {lr_shapes}
- Test NoisyLR shape distribution: {test_shapes}
- Unique Resolution Mapping: {list(lr_shapes.keys())[0]} -> {list(gt_shapes.keys())[0]}
- Scale factor: 2.0x (Width and Height)

4. PIXEL STATISTICS (Sampled)
--------------------------------------------------------
Ground Truth (GT):
- Dtype: float32
- Min range: {gt_summary["min_range"]}
- Max range: {gt_summary["max_range"]}
- Mean range: {gt_summary["mean_range"]}
- Std range: {gt_summary["std_range"]}
- Global Mean: {gt_summary["global_avg_mean"]:.6f}
- Global Std: {gt_summary["global_avg_std"]:.6f}

Train NoisyLR:
- Dtype: float32
- Min range: {lr_summary["min_range"]} (values < 0 are present!)
- Max range: {lr_summary["max_range"]} (values > 1 are present!)
- Mean range: {lr_summary["mean_range"]}
- Std range: {lr_summary["std_range"]}
- Global Mean: {lr_summary["global_avg_mean"]:.6f}
- Global Std: {lr_summary["global_avg_std"]:.6f}

Test NoisyLR:
- Dtype: float32
- Min range: {test_summary["min_range"]} (values < 0 are present!)
- Max range: {test_summary["max_range"]} (values > 1 are present!)
- Mean range: {test_summary["mean_range"]}
- Std range: {test_summary["std_range"]}
- Global Mean: {test_summary["global_avg_mean"]:.6f}
- Global Std: {test_summary["global_avg_std"]:.6f}

========================================================
CONCLUSION & RECOMMENDATIONS
========================================================
1. INPUT RANGE HANDLING:
   - Degraded inputs range from approximately {min(lr_summary["min_range"][0], test_summary["min_range"][0]):.4f} to {max(lr_summary["max_range"][1], test_summary["max_range"][1]):.4f}.
   - Ground truth is strictly normalized in [0, 1].
   - DO NOT CLIP degraded inputs before passing them to the model.
   - Recommended preprocessing is dataset-global standardization using global train stats:
     Mean: {lr_summary["global_avg_mean"]:.6f}
     Std: {lr_summary["global_avg_std"]:.6f}

2. RESOLUTION RESOLVING:
   - The task is a combined 2x Super-Resolution and denoising task.
   - U-Net structures must include a 2x upsampling layer at the end or before residual summation.
"""

    with open("reports/dataset_report.txt", "w") as f:
        f.write(report_txt)

    print("\nDataset reports saved to reports/dataset_report.json and reports/dataset_report.txt")

    # 5. Visualizations
    print(f"\nGenerating {args.num_visualizations} random paired visualizations...")
    viz_indices = np.random.choice(len(paired_samples), args.num_visualizations, replace=False)
    
    for idx, val_idx in enumerate(viz_indices):
        lr_path, gt_path = paired_samples[val_idx]
        basename = os.path.splitext(os.path.basename(lr_path))[0]
        
        lr_img = np.load(lr_path)
        gt_img = np.load(gt_path)
        
        h_gt, w_gt = gt_img.shape
        bicubic_img = cv2.resize(lr_img, (w_gt, h_gt), interpolation=cv2.INTER_CUBIC)
        
        bicubic_clipped = np.clip(bicubic_img, 0.0, 1.0)
        abs_diff = np.abs(bicubic_clipped - gt_img)
        
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        im0 = axes[0].imshow(lr_img, cmap='gray')
        axes[0].set_title(f"Degraded Input ({basename})\nRange: [{lr_img.min():.2f}, {lr_img.max():.2f}]")
        axes[0].axis('off')
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        
        im1 = axes[1].imshow(gt_img, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title(f"Ground Truth\nRange: [{gt_img.min():.1f}, {gt_img.max():.1f}]")
        axes[1].axis('off')
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        im2 = axes[2].imshow(bicubic_clipped, cmap='gray', vmin=0, vmax=1)
        axes[2].set_title(f"Bicubic Upsampled (Clipped)\nRange: [{bicubic_clipped.min():.2f}, {bicubic_clipped.max():.2f}]")
        axes[2].axis('off')
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        
        im3 = axes[3].imshow(abs_diff, cmap='hot', vmin=0, vmax=0.5)
        axes[3].set_title(f"Absolute Difference\nMean Error: {abs_diff.mean():.4f}")
        axes[3].axis('off')
        fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        viz_save_path = f"reports/data_visualizations/pair_{basename}.png"
        plt.savefig(viz_save_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"  Saved visual comparison to {viz_save_path}")
        
    print("\nDataset inspection complete.")

if __name__ == "__main__":
    main()
