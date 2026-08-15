import os
import sys
# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import random
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from data.dataset import SemiconductorDataset
from models.factory import get_model
from training.trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Train Semiconductor Restoration Model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="baseline_run",
        help="Name of the experiment folder"
    )
    parser.add_argument(
        "--skip_sanity",
        action="store_true",
        help="Skip the overfit sanity test before training"
    )
    return parser.parse_args()

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seeds set to: {seed}")

def get_parameter_count(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = (total_params * 4) / (1024 * 1024) # Float32 = 4 bytes
    return total_params, trainable_params, model_size_mb

def run_sanity_check(config, device):
    print("\n=== STARTING PIPELINE SANITY TEST ===")
    
    # 1. Initialize dataset with a tiny subset (4 samples)
    dataset = SemiconductorDataset(
        dataset_root=config["dataset"]["root"],
        split_file=config["dataset"]["train_split"],
        mode="train",
        normalize=config["dataset"]["normalize"],
        global_mean=config["dataset"]["global_mean"],
        global_std=config["dataset"]["global_std"],
        augment=False
    )
    
    subset = Subset(dataset, [0, 1, 2, 3])
    dataloader = DataLoader(subset, batch_size=4, shuffle=False)
    
    # 2. Get a fresh model
    model = get_model(config)
    model.to(device)
    
    # Setup optimizer for sanity check
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.L1Loss()
    
    # Get initial sample
    lr, gt = next(iter(dataloader))
    lr = lr.to(device)
    gt = gt.to(device)
    
    # Run initial forward pass
    model.eval()
    with torch.no_grad():
        pred = model(lr)
        initial_loss = criterion(pred, gt).item()
        print(f"  - Initial sanity loss: {initial_loss:.6f}")
        
        # Check outputs shape
        assert pred.shape == (4, 1, 256, 256), f"Output shape mismatch: {pred.shape} vs (4, 1, 256, 256)"
        print("  - Shape check passed.")
        
        # Check values range
        assert torch.all(pred >= 0.0) and torch.all(pred <= 1.0), "Sanity test error: Outputs must be within [0,1]"
        print("  - [0, 1] range constraint check passed.")
    
    # 3. Train for 50 steps to verify overfitting
    print("  - Overfitting to 4 samples for 50 iterations...")
    model.train()
    for step in range(50):
        optimizer.zero_grad()
        pred = model(lr)
        loss = criterion(pred, gt)
        loss.backward()
        
        # Verify gradients are non-zero on the first step
        if step == 0:
            grad_sum = 0.0
            for p in model.parameters():
                if p.requires_grad and p.grad is not None:
                    grad_sum += torch.sum(torch.abs(p.grad)).item()
            assert grad_sum > 0, "Sanity test error: Gradients are zero!"
            print(f"  - Non-zero gradient check passed (sum: {grad_sum:.4f}).")
            
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        final_pred = model(lr)
        final_loss = criterion(final_pred, gt).item()
        print(f"  - Final sanity loss after 50 steps: {final_loss:.6f}")
        
        # Verify loss decreased substantially
        assert final_loss < initial_loss * 0.7, f"Sanity test error: Model failed to overfit. Initial: {initial_loss:.6f}, Final: {final_loss:.6f}"
        print("  - Overfitting convergence check passed.")
        
        # Check NaNs/Infs
        assert not torch.isnan(final_pred).any(), "Sanity test error: NaN detected in predictions"
        assert not torch.isinf(final_pred).any(), "Sanity test error: Inf detected in predictions"
        print("  - NaN/Inf checks passed.")
        
    print("=== PIPELINE SANITY TEST SUCCESSFUL ===\n")

def main():
    args = parse_args()
    
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    # Setup seeds
    set_seeds(config["training"]["seed"])
    
    # Configure device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Run sanity test
    if not args.skip_sanity:
        run_sanity_check(config, device)
        
    # Create experiment folder
    experiment_dir = os.path.join("experiments", args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Save the configuration for reproducibility
    with open(os.path.join(experiment_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)
        
    # Setup datasets
    print("Loading datasets...")
    train_dataset = SemiconductorDataset(
        dataset_root=config["dataset"]["root"],
        split_file=config["dataset"]["train_split"],
        mode="train",
        normalize=config["dataset"]["normalize"],
        global_mean=config["dataset"]["global_mean"],
        global_std=config["dataset"]["global_std"],
        augment=config["dataset"]["augment"]
    )
    
    val_dataset = SemiconductorDataset(
        dataset_root=config["dataset"]["root"],
        split_file=config["dataset"]["val_split"],
        mode="val",
        normalize=config["dataset"]["normalize"],
        global_mean=config["dataset"]["global_mean"],
        global_std=config["dataset"]["global_std"],
        augment=False
    )
    
    # Setup dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["dataset"]["num_workers"],
        pin_memory=(device.type == "cuda")
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["dataset"]["num_workers"],
        pin_memory=(device.type == "cuda")
    )
    
    # Instantiate model
    print("Instantiating model...")
    model = get_model(config)
    
    # Parameter counts
    tot_params, train_params, model_size_mb = get_parameter_count(model)
    print(f"Model Summary:")
    print(f"  - Total Parameters: {tot_params:,}")
    print(f"  - Trainable Parameters: {train_params:,}")
    print(f"  - Approximate Model Size: {model_size_mb:.4f} MB")
    
    # Instantiate trainer and fit
    trainer = Trainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        experiment_dir=experiment_dir
    )
    
    trainer.fit()

if __name__ == "__main__":
    main()
