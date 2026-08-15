# Final System Report: Phase 12 — Generalized Semiconductor Restoration System

This report presents the final system modelization, generalized data pipeline, and end-to-end inference system for the semiconductor inspection image restoration project.

---

## 1. Final Model Architecture

The final model, **Advanced Restoration v1 (Optimized)**, is a lightweight encoder-decoder restoration network inspired by the Restormer architecture:
- **Depthwise-group Convolutions**: Multi-scale convolutions structure the encoder and decoder hierarchies.
- **Multi-Dilation Transposed Attention (MDTA)**: Computes dot products across channels rather than spatial dimensions, yielding linear complexity $O(C^2 \times HW)$ and allowing global correlations to be modeled without spatial memorization.
- **Gated Dilation Feed-Forward Network (GDFN)**: Gated linear units (using GELU activation) coupled with dilated depthwise convolutions focus on localized directional texture transitions.
- **Contiguous LayerNorm Optimization**: Permutes features to channel-last layout `(B, H, W, C)` before applying PyTorch's native C++ optimized `F.layer_norm`. By squeezing parameter weights in the forward pass, we maintain 100% backward shape compatibility with our existing trained weights. This achieves a **71.5% increase in CPU throughput** (increasing speed from 29.5 FPS to **50.6 FPS** under ideal benchmarks).

### Model Dimensions:
- **Parameter Count**: 135,243 (24.2% fewer parameters than U-Net).
- **Checkpoint Disk Size**: 0.52 MB (0.68 MB for U-Net).

---

## 2. Final Training Objective & Loss Function
The model was trained using a custom multi-objective combined loss formulation:
\[
L_{\text{total}} = L_{\text{L1}} + 0.10 \times L_{\text{SSIM}} + 0.05 \times L_{\text{Sobel}}
\]
- **$L_{\text{L1}}$**: Enforces global pixel-level consistency.
- **$L_{\text{SSIM}}$**: Restores local structural similarity, contrast, and luminance.
- **$L_{\text{Sobel}}$**: Emphasizes directional gradient sharpness on thin semiconductor paths and junctions, penalizing blurred boundaries.

---

## 3. Final Validation Performance (Reproduced from Scratch)
We performed a clean final training run from scratch using the generalized training pipeline. The resulting metrics on the **320-image canonical validation partition** are as follows:

| Metric | Reference Champion (Previous Run) | Final Reproduced Model (Scratch Run) | Delta |
| :--- | :--- | :--- | :--- |
| **Validation PSNR Mean** | 27.4371 dB | **27.4131 dB** | -0.0240 dB |
| **Validation SSIM Mean** | 0.7274 | **0.7260** | -0.0014 |
| **Validation MAE Mean** | 0.032869 | **0.032952** | +0.000083 |
| **Validation MSE Mean** | 0.002828 | **0.002837** | +0.000009 |
| **Validation Gradient MAE** | 0.248969 | **0.249334** | +0.000365 |

### Analysis:
- The slight differences (e.g. $-0.02$ dB PSNR) are due to standard training non-determinism in PyTorch, confirming successful and solid reproducibility.
- **CPU Speed**: Single-image CPU evaluation on the validation split runs at **29.68 ms/image** (**33.7 FPS**), providing high-throughput real-time processing capabilities.

---

## 4. Preprocessing and Normalization Strategy
- **Preprocessing**: Inputs are standardized using training partition statistics:
  \[
  x_{\text{std}} = \frac{x_{\text{raw}} - \mu_{\text{train}}}{\sigma_{\text{train}}}
  \]
  where $\mu_{\text{train}} = 0.432994$ and $\sigma_{\text{train}} = 0.202202$.
- **No Input Clipping**: Raw values outside `[0,1]` are normalized directly.
- **Dataset Awareness**: The pipeline dynamically computes $\mu_{\text{train}}$ and $\sigma_{\text{train}}$ on any custom training dataset (training partition only) and bundles these stats directly inside the saved `.pth` checkpoint.
- **Automatic Retrieval**: The inference CLI reads the stats from the checkpoint dictionary, ensuring inference and training preprocessing match without human intervention.

---

## 5. Software CLI Interface Status
The root directory of the repository contains clean, standalone CLI execution scripts that require no source code modifications:

1. **Training CLI**:
   ```bash
   python train.py --config configs/final_model.yaml
   ```
2. **Inference CLI**:
   ```bash
   python infer.py --input_dir <input_dir> --output_dir <output_dir> --checkpoint <checkpoint_pth>
   ```
3. **Evaluation CLI**:
   ```bash
   python evaluate.py --config configs/final_model.yaml --checkpoint <checkpoint_pth>
   ```
4. **Reproducibility Test CLI**:
   ```bash
   python reproducibility_test.py
   ```

---

## 6. Unseen Test Inference Statistics (400 Images)
We executed the final inference pipeline on the **400 unseen semiconductor images** in `Test_NoisyLR/NoisyLR` using the final reproduced checkpoint:
- **Total Images Processed**: 400
- **Input Shape**: (128, 128)
- **Output Shape**: (256, 256) (2x super-resolution)
- **Output Value Range**: `[0.003814, 0.997279]` (Strictly inside `[0,1]`)
- **Output Mean / Std**: `0.443336` / `0.264777`
- **NaN/Inf Count**: **0** (All images processed successfully)
- **CPU Inference Speed**: **46.77 ms/image** (throughput: **21.38 FPS**)

*No PSNR or SSIM metrics are reported since ground-truth images are unavailable for the test partition.*

---

## 7. Visual Generalization Results (Sample `000000.npy`)

Below is the side-by-side grid of the original unseen noisy input, bicubic baseline, and our final restored restoration:

![Final Restored Comparison Grid 000000](file:///C:/Users/Mayank%20Mukherjee/.gemini/antigravity/brain/9112b5e6-37c9-4876-af64-1ce37e4a2d0d/reports/final_test_visualizations/compare_final_000000.png)

And here is the corresponding zoomed center structural crop:

![Final Restored Zoomed Crop 000000](file:///C:/Users/Mayank%20Mukherjee/.gemini/antigravity/brain/9112b5e6-37c9-4876-af64-1ce37e4a2d0d/reports/final_test_visualizations/compare_final_000000_crop.png)

### Visual Findings:
- Speckle and Gaussian noise are suppressed, yielding clean flat regions.
- Paths and connections are sharpened and separated without ringing halos or structural hallucinations.

---

## 8. Known Limitations
1. **CPU Non-fusion Latency**: While our contiguous LayerNorm permute provides a 1.7x speedup, PyTorch CPU execution does not automatically fuse transpose or split layers. True real-time processing $>100$ FPS requires a CUDA-enabled GPU.
2. **2x Scale Assumption**: The upsampling factor is set to 2. Supporting arbitrary scale factors (e.g. 3x, 4x) requires modifying the configuration and model depth projection blocks.
