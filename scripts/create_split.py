import os
import argparse
import glob
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Train/Validation Splits")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="C:/Users/Mayank Mukherjee/Desktop/Hack/Data-public-20260814T125741Z-1-001/Data-public",
        help="Path to dataset root"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.90,
        help="Ratio of training samples"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    gt_dir = os.path.join(args.dataset_root, "train", "train", "GT")
    if not os.path.exists(gt_dir):
        print(f"Error: GT directory not found at {gt_dir}")
        return
        
    gt_files = sorted([os.path.basename(f) for f in glob.glob(os.path.join(gt_dir, "*.npy"))])
    num_files = len(gt_files)
    print(f"Total available training files: {num_files}")
    
    # Deterministic shuffling
    np.random.seed(args.seed)
    indices = np.arange(num_files)
    np.random.shuffle(indices)
    
    num_train = int(num_files * args.train_ratio)
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]
    
    train_files = [gt_files[i] for i in train_indices]
    val_files = [gt_files[i] for i in val_indices]
    
    print(f"Train split size: {len(train_files)} ({len(train_files)/num_files:.1%})")
    print(f"Val split size: {len(val_files)} ({len(val_files)/num_files:.1%})")
    
    # Verify no overlap
    train_set = set(train_files)
    val_set = set(val_files)
    overlap = train_set.intersection(val_set)
    
    if len(overlap) > 0:
        print(f"ERROR: Found overlap between train and validation splits: {overlap}")
        return
    else:
        print("SUCCESS: Zero overlap between train and validation splits verified.")
        
    # Write to files
    os.makedirs("data", exist_ok=True)
    
    with open("data/train_split.txt", "w") as f:
        for fname in sorted(train_files):
            f.write(fname + "\n")
            
    with open("data/val_split.txt", "w") as f:
        for fname in sorted(val_files):
            f.write(fname + "\n")
            
    print("Split files successfully written to data/train_split.txt and data/val_split.txt")

if __name__ == "__main__":
    main()
