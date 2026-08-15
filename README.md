# AI-Based Semiconductor Inspection Image Restoration

This repository contains an end-to-end, generalized deep-learning solution for the restoration and super-resolution of degraded grayscale semiconductor inspection images.

Given a degraded, low-resolution grayscale image ($128 \times 128$), the system removes speckle and Gaussian noise, restores high-frequency details, and upsamples it to a clean ground-truth resolution ($256 \times 256$).

---

## 1. Directory Structure

```
semiconductor_restoration/
├── configs/
│   └── final_model.yaml           # Final experiment config file
├── data/
│   └── dataset.py                 # Generalized dataset loader with data validation
├── models/
│   ├── blocks.py                  # Standard neural building blocks
│   ├── residual_unet.py           # Baseline Residual U-Net
│   ├── advanced_restoration.py    # Restormer-inspired quality model
│   └── factory.py                 # Model factory registrator
├── losses/
│   ├── ssim_loss.py               # Differentiable PyTorch SSIM loss
│   ├── gradient_loss.py           # Differentiable Sobel gradient loss
│   └── combined_loss.py           # Combined L1 + SSIM + Sobel Edge loss
├── metrics/
│   └── metrics.py                 # Evaluation metrics calculations
├── training/
│   └── trainer.py                 # Modular training loop runner
├── train.py                       # Root training execution script
├── infer.py                       # Standalone inference CLI
├── evaluate.py                    # Root evaluation script
├── reproducibility_test.py        # Independent reproducibility test script
├── requirements.txt               # Package dependencies list
└── README.md                      # This documentation
```

---

## 2. Environment Setup

This project runs on standard CPU/GPU Python environments.

```bash
# Clone or open the repository
cd semiconductor_restoration

# Install required dependencies
pip install -r requirements.txt
```

---

## 3. Training CLI
To train the model from scratch using the final configuration (which automatically calculates dynamic dataset statistics on the training partition and saves them inside the checkpoint):

```bash
python train.py --config configs/final_model.yaml
```

### Training Highlights:
- **Splits**: Canonical 2,880 train / 320 validation samples.
- **Normalization**: Computes training-only mean and standard deviation dynamically and registers them under the `"normalization"` key of the saved checkpoint.
- **Pipeline Sanity Test**: Automatically runs shape, range `[0,1]`, gradient flow, and overfitting convergence tests before beginning training.

---

## 4. Standalone Inference CLI
To run inference on any directory containing unseen degraded semiconductor images:

```bash
python infer.py \
    --input_dir <path_to_noisy_lr_npy_directory> \
    --output_dir <path_to_output_directory> \
    --checkpoint <path_to_checkpoint_pth>
```

### Preprocessing & Execution:
- Automatically extracts the custom `mean` and `std` normalization values stored inside the checkpoint file to standardise input.
- Automatically instantiates the correct model architecture based on the checkpoint metadata.
- Outputs are saved as grayscale `.npy` files at 2x spatial resolution.
- Reports CPU latency and throughput (FPS) diagnostics.

---

## 5. Standalone Evaluation CLI
To compute quality metrics on a paired validation dataset:

```bash
python evaluate.py --config configs/final_model.yaml --checkpoint <path_to_checkpoint_pth>
```

Calculates and prints Mean, Median, and Standard Deviation for:
- **PSNR** (Peak Signal-to-Noise Ratio)
- **SSIM** (Structural Similarity Index Measure)
- **MAE** (Mean Absolute Error)
- **MSE** (Mean Squared Error)
- **Gradient MAE** (Sobel gradient fidelity difference)

---

## 6. Reproducibility Test
To run determinism and validation constraints tests on a small subset of unseen test images:

```bash
python reproducibility_test.py
```

---

## 7. Model Architecture Details

### Advanced Restoration v1 (Quality Champion)
- **Restormer-inspired blocks**: Replaces standard convolutional blocks with LayerNorm, Multi-Dilation Transposed Attention (MDTA), and Gated Dilation Feed-Forward Networks (GDFN).
- **Contiguous LayerNorm Optimization**: Permutes tensors to channel-last layout `(B, H, W, C)` before executing PyTorch's native `F.layer_norm`. Squeezes parameter weights to maintain 100% backward shape compatibility. This achieves a **71% increase in CPU throughput** (increasing speed from 29.5 FPS to **50.6 FPS**).
- **Parameters**: 135,243 (24% smaller than baseline U-Net).
- **Combined Loss**:
  \[
  L_{\text{total}} = L_{\text{L1}} + 0.10 \times L_{\text{SSIM}} + 0.05 \times L_{\text{edge}}
  \]
  where $L_{\text{edge}}$ computes the L1 difference between the horizontal and vertical Sobel gradients.
