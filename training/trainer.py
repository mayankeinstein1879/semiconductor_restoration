import os
import time
import csv
import torch
import torch.nn as nn
import numpy as np
from metrics.metrics import compute_all_metrics

class Trainer:
    """
    Trainer class to handle model training, validation, checkpointing, and logging.
    """
    def __init__(self, config, model, train_loader, val_loader, experiment_dir, train_mean=0.432994, train_std=0.202202):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.experiment_dir = experiment_dir
        self.train_mean = train_mean
        self.train_std = train_std
        
        # Configure device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Training on device: {self.device}")
        
        # Setup optimization
        self.lr = float(config["training"]["learning_rate"])
        self.weight_decay = float(config["training"]["weight_decay"])
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        self.epochs = int(config["training"]["epochs"])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs,
            eta_min=1e-6
        )
        
        # Loss function - Loads combined loss based on config weights
        from losses.combined_loss import CombinedLoss
        l1_w = float(config.get("loss", {}).get("reconstruction_weight", 1.0))
        ssim_w = float(config.get("loss", {}).get("ssim_weight", 0.0))
        edge_w = float(config.get("loss", {}).get("edge_weight", 0.0))
        self.criterion = CombinedLoss(l1_weight=l1_w, ssim_weight=ssim_w, edge_weight=edge_w)
        
        # AMP Configuration (only if cuda is available)
        self.use_amp = config["training"].get("amp", False) and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Log path
        self.metrics_log_path = os.path.join(self.experiment_dir, "metrics.csv")
        self._init_log_file()
        
        # Best validation PSNR for checkpointing
        self.best_psnr = -1.0
        
    def _init_log_file(self):
        os.makedirs(self.experiment_dir, exist_ok=True)
        # Create CSV header
        with open(self.metrics_log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "train_loss", "val_l1", "val_psnr", 
                "val_ssim", "val_mae", "val_mse", "lr", "epoch_time_sec"
            ])
            
    def log_epoch(self, epoch, train_loss, val_metrics, lr, epoch_time):
        with open(self.metrics_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{train_loss:.6f}",
                f"{val_metrics['L1']:.6f}",
                f"{val_metrics['PSNR']:.4f}",
                f"{val_metrics['SSIM']:.4f}",
                f"{val_metrics['MAE']:.6f}",
                f"{val_metrics['MSE']:.6f}",
                f"{lr:.8f}",
                f"{epoch_time:.2f}"
            ])
            
    def save_checkpoint(self, name):
        checkpoint_dir = os.path.join(self.experiment_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config,
            "best_psnr": self.best_psnr,
            "normalization": {
                "mean": float(self.train_mean),
                "std": float(self.train_std)
            }
        }
        
        save_path = os.path.join(checkpoint_dir, f"{name}.pth")
        torch.save(checkpoint, save_path)
        print(f"Saved checkpoint to {save_path}")
        
    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        
        for lr_tensor, gt_tensor in self.train_loader:
            lr_tensor = lr_tensor.to(self.device)
            gt_tensor = gt_tensor.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.scaler:
                with torch.cuda.amp.autocast():
                    pred = self.model(lr_tensor)
                    loss = self.criterion(pred, gt_tensor)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred = self.model(lr_tensor)
                loss = self.criterion(pred, gt_tensor)
                loss.backward()
                self.optimizer.step()
                
            running_loss += loss.item() * lr_tensor.size(0)
            
        return running_loss / len(self.train_loader.dataset)
        
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        
        val_l1 = 0.0
        val_psnr = []
        val_ssim = []
        val_mae = []
        val_mse = []
        
        for lr_tensor, gt_tensor in self.val_loader:
            lr_tensor = lr_tensor.to(self.device)
            gt_tensor = gt_tensor.to(self.device)
            
            pred = self.model(lr_tensor)
            loss = self.criterion(pred, gt_tensor)
            val_l1 += loss.item() * lr_tensor.size(0)
            
            # Compute evaluation metrics (needs numpy)
            pred_np = pred.squeeze(1).cpu().numpy()
            gt_np = gt_tensor.squeeze(1).cpu().numpy()
            
            # Compute metrics for each sample in the batch
            for i in range(pred_np.shape[0]):
                m = compute_all_metrics(gt_np[i], pred_np[i])
                val_psnr.append(m["PSNR"])
                val_ssim.append(m["SSIM"])
                val_mae.append(m["MAE"])
                val_mse.append(m["MSE"])
                
        metrics = {
            "L1": val_l1 / len(self.val_loader.dataset),
            "PSNR": float(np.mean(val_psnr)),
            "SSIM": float(np.mean(val_ssim)),
            "MAE": float(np.mean(val_mae)),
            "MSE": float(np.mean(val_mse))
        }
        return metrics
        
    def fit(self):
        print("Starting training loop...")
        for epoch in range(1, self.epochs + 1):
            t_start = time.perf_counter()
            
            train_loss = self.train_epoch()
            val_metrics = self.validate()
            
            self.scheduler.step()
            lr_curr = self.optimizer.param_groups[0]["lr"]
            
            t_end = time.perf_counter()
            epoch_time = t_end - t_start
            
            # Print epoch summary
            print(
                f"Epoch [{epoch}/{self.epochs}] - "
                f"Train Loss: {train_loss:.6f} | "
                f"Val L1: {val_metrics['L1']:.6f} | "
                f"Val PSNR: {val_metrics['PSNR']:.4f} dB | "
                f"Val SSIM: {val_metrics['SSIM']:.4f} | "
                f"LR: {lr_curr:.6e} | "
                f"Time: {epoch_time:.2f}s"
            )
            
            # Log metrics to CSV
            self.log_epoch(epoch, train_loss, val_metrics, lr_curr, epoch_time)
            
            # Checkpoint management
            # Save latest checkpoint
            self.save_checkpoint("latest")
            
            # Save best checkpoint
            if val_metrics["PSNR"] > self.best_psnr:
                self.best_psnr = val_metrics["PSNR"]
                print(f"New best validation PSNR: {self.best_psnr:.4f} dB. Saving best checkpoint.")
                self.save_checkpoint("best")
                
        print(f"Training completed. Best PSNR: {self.best_psnr:.4f} dB")
