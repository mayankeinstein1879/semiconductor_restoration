import os
import numpy as np
import torch
from torch.utils.data import Dataset

class SemiconductorDataset(Dataset):
    """
    Generalized PyTorch Dataset for paired semiconductor inspection images.
    Discovers files dynamically from input and target directories.
    """
    def __init__(
        self,
        input_dir,
        target_dir=None,
        normalize="standardize",
        global_mean=0.432994,
        global_std=0.202202,
        augment=False,
        scale_factor=2
    ):
        super().__init__()
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.normalize = normalize.lower() if normalize else None
        self.global_mean = global_mean
        self.global_std = global_std
        self.augment = augment
        self.scale_factor = scale_factor
        
        # 1. Discover files
        if not os.path.exists(self.input_dir):
            raise FileNotFoundError(f"Input directory does not exist: {self.input_dir}")
            
        self.filenames = sorted([
            f for f in os.listdir(self.input_dir)
            if f.endswith(".npy")
        ])
        
        if len(self.filenames) == 0:
            raise ValueError(f"No .npy files found in input directory: {self.input_dir}")
            
        # 2. Check targets if provided
        if self.target_dir:
            if not os.path.exists(self.target_dir):
                raise FileNotFoundError(f"Target directory does not exist: {self.target_dir}")
                
            # Verify file pairing
            target_files = set(os.listdir(self.target_dir))
            for f in self.filenames:
                if f not in target_files:
                    raise FileNotFoundError(
                        f"Mismatched pair: File '{f}' exists in input directory but not in target directory."
                    )
                    
        # 3. Perform a quick structural validation check on the first sample
        self._validate_sample(self.filenames[0])
        print(f"Initialized SemiconductorDataset with {len(self.filenames)} samples.")
        
    def __len__(self):
        return len(self.filenames)
        
    def _validate_sample(self, fname):
        """Verifies file integrity, shape scale alignment, dtypes, and NaN presence."""
        lr_path = os.path.join(self.input_dir, fname)
        if os.path.getsize(lr_path) == 0:
            raise IOError(f"Input file '{fname}' is empty/corrupt.")
            
        try:
            lr_img = np.load(lr_path)
        except Exception as e:
            raise IOError(f"Could not load input file '{fname}': {str(e)}")
            
        if np.isnan(lr_img).any() or np.isinf(lr_img).any():
            raise ValueError(f"NaN or Inf found in input image: {fname}")
            
        if lr_img.ndim != 2:
            raise ValueError(f"Expected 2D grayscale input (height, width), got ndim={lr_img.ndim} for {fname}")
            
        # Target validations
        if self.target_dir:
            gt_path = os.path.join(self.target_dir, fname)
            if os.path.getsize(gt_path) == 0:
                raise IOError(f"Target file '{fname}' is empty/corrupt.")
                
            try:
                gt_img = np.load(gt_path)
            except Exception as e:
                raise IOError(f"Could not load target file '{fname}': {str(e)}")
                
            if np.isnan(gt_img).any() or np.isinf(gt_img).any():
                raise ValueError(f"NaN or Inf found in target image: {fname}")
                
            if gt_img.ndim != 2:
                raise ValueError(f"Expected 2D grayscale target (height, width), got ndim={gt_img.ndim} for {fname}")
                
            # Verify scale factor alignment
            expected_h = lr_img.shape[0] * self.scale_factor
            expected_w = lr_img.shape[1] * self.scale_factor
            if gt_img.shape != (expected_h, expected_w):
                raise ValueError(
                    f"Scale factor mismatch for '{fname}': Input shape {lr_img.shape} "
                    f"scaled by {self.scale_factor}x does not match Target shape {gt_img.shape} "
                    f"(expected {(expected_h, expected_w)})."
                )
                
    def __getitem__(self, idx):
        fname = self.filenames[idx]
        
        # Load input image
        lr_path = os.path.join(self.input_dir, fname)
        lr_img = np.load(lr_path).astype(np.float32)
        
        # Apply dataset-aware normalization
        if self.normalize == "standardize":
            lr_img = (lr_img - self.global_mean) / self.global_std
            
        # Target image loader
        if self.target_dir:
            gt_path = os.path.join(self.target_dir, fname)
            gt_img = np.load(gt_path).astype(np.float32)
            
            # Apply augmentations for training
            if self.augment:
                lr_img, gt_img = self._apply_augmentation(lr_img, gt_img)
                
            lr_tensor = torch.from_numpy(lr_img).unsqueeze(0)
            gt_tensor = torch.from_numpy(gt_img).unsqueeze(0)
            return lr_tensor, gt_tensor
        else:
            lr_tensor = torch.from_numpy(lr_img).unsqueeze(0)
            return lr_tensor, fname
            
    def _apply_augmentation(self, lr, gt):
        """Applies matching spatial augmentations to keep structures aligned."""
        # Random horizontal flip
        if np.random.rand() > 0.5:
            lr = np.fliplr(lr).copy()
            gt = np.fliplr(gt).copy()
            
        # Random vertical flip
        if np.random.rand() > 0.5:
            lr = np.flipud(lr).copy()
            gt = np.flipud(gt).copy()
            
        # Random 90 degree rotations
        rot_k = np.random.randint(0, 4)
        if rot_k > 0:
            lr = np.rot90(lr, rot_k).copy()
            gt = np.rot90(gt, rot_k).copy()
            
        # Random transpose
        if np.random.rand() > 0.5:
            lr = lr.T.copy()
            gt = gt.T.copy()
            
        return lr, gt
