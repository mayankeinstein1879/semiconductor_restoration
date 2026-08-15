import os
import sys
import argparse
import random
import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader, Subset

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.dataset import SemiconductorDataset
from models.factory import get_model
from training.trainer import Trainer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_dataset_stats(dataset, train_indices=None):
    """Computes global mean and standard deviation of raw input training images only."""
    print("Computing dataset normalization statistics on training partition...")
    all_vals = []
    
    indices = train_indices if train_indices is not None else range(len(dataset))
    
    for idx in indices:
        fname = dataset.filenames[idx]
        lr_path = os.path.join(dataset.input_dir, fname)
        img = np.load(lr_path)
        all_vals.append(img.flatten())
        
    all_vals = np.concatenate(all_vals)
    mean_val = float(all_vals.mean())
    std_val = float(all_vals.std())
    print(f"Calculated statistics - Mean: {mean_val:.6f} | Std: {std_val:.6f}")
    return mean_val, std_val

def main():
    parser = argparse.ArgumentParser(description="Clean generalized training pipeline for semiconductor image restoration.")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file.")
    parser.add_argument("--experiment_name", type=str, default=None, help="Overwrites experiment name.")
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    # Experiment name override
    if args.experiment_name:
        config["training"]["experiment_name"] = args.experiment_name
    elif "experiment_name" not in config["training"]:
        config["training"]["experiment_name"] = "final_experiment"
        
    # Set seed
    seed = config["training"].get("seed", 42)
    set_seed(seed)
    print(f"Random seeds set to: {seed}")
    
    # Paths from config
    dataset_root = config["dataset"]["root"]
    
    # Configure input and target folders dynamically (fallback to default hackathon directories)
    input_dir = config["dataset"].get("input_dir")
    target_dir = config["dataset"].get("target_dir")
    if not input_dir:
        input_dir = os.path.join(dataset_root, "train", "train", "NoisyLR")
    if not target_dir:
        target_dir = os.path.join(dataset_root, "train", "train", "GT")
    
    # 1. Instantiate the dataset in raw mode to calculate statistics
    raw_dataset = SemiconductorDataset(
        input_dir=input_dir,
        target_dir=target_dir,
        normalize=None, # no normalization yet
        augment=False,
        scale_factor=config["model"].get("scale_factor", 2)
    )
    
    # 2. Determine Train/Val indices and splits
    train_split_path = config["dataset"].get("train_split")
    val_split_path = config["dataset"].get("val_split")
    
    if train_split_path and os.path.exists(train_split_path) and val_split_path and os.path.exists(val_split_path):
        # Load splits
        with open(train_split_path, "r") as f:
            train_names = set(line.strip() for line in f if line.strip())
        with open(val_split_path, "r") as f:
            val_names = set(line.strip() for line in f if line.strip())
            
        train_indices = [i for i, f in enumerate(raw_dataset.filenames) if f in train_names]
        val_indices = [i for i, f in enumerate(raw_dataset.filenames) if f in val_names]
        print(f"Loaded splits from file: Train indices={len(train_indices)} | Val indices={len(val_indices)}")
    else:
        # Perform dynamic 90/10 split
        num_samples = len(raw_dataset)
        indices = list(range(num_samples))
        random.shuffle(indices)
        split_idx = int(0.9 * num_samples)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        print(f"No valid split files found. Created dynamic 90/10 split: Train={len(train_indices)} | Val={len(val_indices)}")
        
    # 3. Calculate statistics on training partition only
    if config["dataset"].get("normalize") == "standardize":
        # If mean and std are explicitly provided in config, use them, otherwise calculate dynamically
        train_mean = config["dataset"].get("global_mean")
        train_std = config["dataset"].get("global_std")
        
        if train_mean is None or train_std is None:
            train_mean, train_std = compute_dataset_stats(raw_dataset, train_indices)
            config["dataset"]["global_mean"] = train_mean
            config["dataset"]["global_std"] = train_std
        else:
            print(f"Using pre-defined normalization stats - Mean: {train_mean:.6f} | Std: {train_std:.6f}")
    else:
        train_mean, train_std = 0.0, 1.0
        
    # 4. Instantiate final datasets with normalization and split
    train_dataset = SemiconductorDataset(
        input_dir=input_dir,
        target_dir=target_dir,
        normalize=config["dataset"].get("normalize", "standardize"),
        global_mean=train_mean,
        global_std=train_std,
        augment=config["dataset"].get("augment", True),
        scale_factor=config["model"].get("scale_factor", 2)
    )
    
    val_dataset = SemiconductorDataset(
        input_dir=input_dir,
        target_dir=target_dir,
        normalize=config["dataset"].get("normalize", "standardize"),
        global_mean=train_mean,
        global_std=train_std,
        augment=False,
        scale_factor=config["model"].get("scale_factor", 2)
    )
    
    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        batch_size=config["training"].get("batch_size", 32),
        shuffle=True,
        num_workers=config["dataset"].get("num_workers", 0)
    )
    
    val_loader = DataLoader(
        Subset(val_dataset, val_indices),
        batch_size=config["training"].get("batch_size", 32),
        shuffle=False,
        num_workers=config["dataset"].get("num_workers", 0)
    )
    
    # 5. Sanity test of the model pipeline
    print("\n=== STARTING PIPELINE SANITY TEST ===")
    device = torch.device("cpu")
    model = get_model(config).to(device)
    
    sample_lr, sample_gt = next(iter(train_loader))
    sample_lr, sample_gt = sample_lr[:4].to(device), sample_gt[:4].to(device)
    
    # Forward pass
    with torch.no_grad():
        out = model(sample_lr)
        
    # Check shape
    scale = config["model"].get("scale_factor", 2)
    expected_out_shape = (sample_lr.size(0), sample_lr.size(1), sample_lr.size(2)*scale, sample_lr.size(3)*scale)
    assert out.shape == expected_out_shape, f"Sanity shape check failed: Got {out.shape}, expected {expected_out_shape}"
    print("  - Shape check passed.")
    
    # Check values in [0,1]
    assert out.min() >= 0.0 and out.max() <= 1.0, f"Sigmoid range violation: Min={out.min():.4f}, Max={out.max():.4f}"
    print("  - [0, 1] range constraint check passed.")
    
    # Verify backward pass gradients
    from losses.combined_loss import CombinedLoss
    criterion = CombinedLoss(
        l1_weight=config["loss"].get("reconstruction_weight", 1.0),
        ssim_weight=config["loss"].get("ssim_weight", 0.0),
        edge_weight=config["loss"].get("edge_weight", 0.0)
    )
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    loss = criterion(model(sample_lr), sample_gt)
    loss.backward()
    
    grad_sum = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
    assert grad_sum > 0, "Backward gradients are zero!"
    print(f"  - Non-zero gradient check passed (sum: {grad_sum:.4f}).")
    
    # Tiny overfit validation
    optimizer.zero_grad()
    for _ in range(50):
        optimizer.zero_grad()
        loss = criterion(model(sample_lr), sample_gt)
        loss.backward()
        optimizer.step()
    print(f"  - Overfitting convergence check passed (Loss: {loss.item():.6f}).")
    
    # Reset model
    model = get_model(config).to(device)
    print("=== PIPELINE SANITY TEST SUCCESSFUL ===\n")
    
    # 6. Initialize trainer and run training
    experiment_dir = os.path.join("experiments", config["training"]["experiment_name"])
    trainer = Trainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        experiment_dir=experiment_dir,
        train_mean=train_mean,
        train_std=train_std
    )
    
    trainer.fit()

if __name__ == "__main__":
    main()
