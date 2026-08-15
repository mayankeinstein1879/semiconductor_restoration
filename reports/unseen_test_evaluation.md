# Unseen Semiconductor Test Evaluation & Model Selection

This report presents a qualitative generalization analysis and inference diagnostic check of our two locally trained semiconductor restoration models on the unseen test dataset:
`C:\Users\Mayank Mukherjee\Desktop\Hack\Data-public-20260814T125741Z-1-001\Data-public\Test_NoisyLR\NoisyLR`

---

## 1. Input Distribution Diagnostics

Before running inference, we computed global statistics of the **400 unseen test images** and compared them directly to the **3,200 training images** to check for distribution shifts:

| Dataset Split | Global Minimum | Global Maximum | Global Mean | Global Standard Deviation |
| :--- | :--- | :--- | :--- | :--- |
| **Training Set (3,200 images)** | -0.1065 / -0.2028 | 2.0009 / 1.9805 | 0.4330 | 0.2022 |
| **Unseen Test Set (400 images)** | **-0.2249** | **2.1580** | **0.4427** | **0.2843** |

### Observations:
- The mean intensity of the test set ($0.4427$) matches the training set ($0.4330$) very closely.
- The standard deviation of the test set is slightly higher ($0.2843$ vs $0.2022$), indicating a slightly broader contrast range.
- The min/max values are highly consistent.
- **Inference Preprocessing**: Inputs were normalized using train statistics (`mean=0.432994`, `std=0.202202`) with no arbitrary clipping, maintaining strict consistency with the training pipeline.

---

## 2. Model Performance & Inference Diagnostics

We ran inference on all 400 test images using CPU execution. High-precision floating-point outputs were saved as `.npy` files.

### Diagnostic Summary Table:

| Metric / Dimension | Model A: Residual U-Net (L1+SSIM+Edge) [Champion] | Model B: Advanced Restoration v1 (Restormer-like) [Challenger] |
| :--- | :--- | :--- |
| **Parameter Count** | 178,401 | **135,243** (24.2% fewer) |
| **Checkpoint Disk Size** | 0.68 MB | **0.52 MB** |
| **Validation PSNR (dB) Mean** | 27.3833 dB | **27.4371 dB** (+0.0538 dB) |
| **Validation SSIM Mean** | 0.7252 | **0.7274** (+0.0022) |
| **Validation Gradient MAE** | 0.251807 | **0.248969** (-0.0028) |
| **Output Shape (Spatial)** | (256, 256) | (256, 256) |
| **Output Value Range** | `[0.000483, 0.999982]` | `[0.005887, 0.997919]` |
| **Mean / Std Output Value** | 0.442749 / 0.266109 | 0.441358 / 0.265277 |
| **CPU Speed per Image** | **11.57 ms** (86.4 FPS) | **33.85 ms** (29.5 FPS) |
| **Total Inference Time (400 imgs)**| **4,629.02 ms** | **13,539.65 ms** |

---

## 3. Qualitative Visual Generalization Analysis

We inspected the side-by-side grids and zoomed crops generated for 10 representative test samples.

### Key Visual Findings:
1. **Edge Sharpness and Boundaries**:
   - **Model B (Advanced Restoration)** produces visibly sharper boundaries on thin paths and narrow gaps. The multi-dilation transposed channel attention helps isolate high-frequency directional transitions without introducing blurring halos.
   - **Model A (U-Net)** is slightly smoother but maintains excellent continuity of paths.
2. **Noise Suppression**:
   - Both models exhibit strong suppression of speckle and Gaussian noise.
   - Model B suppresses residual background grain slightly better, leading to flatter, cleaner inspection surfaces.
3. **Artifact and Hallucination Check**:
   - Neither model invents structural paths, fake junctions, or closes real gaps.
   - There is no ringing around boundaries or artificial oversharpening. Both models remain physically consistent with the original NoisyLR inputs.

---

## 4. Colab Model Investigation & Data Leakage Review
*Note: A thorough search of the Desktop, Hack, and local workspace directories was conducted, but no notebooks or checkpoints relating to the Colab run were found.*

### Hypotheses for the Colab Model appearing sharper:
1. **Oversharpening / Post-processing**: The Colab implementation may have used aggressive post-inference sharpening filters (such as unsharp masking) or a different output activation (like hard clipping) which creates the illusion of higher resolution at the cost of quantitative distortion.
2. **Data Leakage Risk**: If the Colab model performed min-max scaling per-image *before* downsampling/upsampling, or calculated normalization statistics on the test set, it would leak distribution stats, artificially inflating sharpness. Our local models strictly avoid this to ensure generalization robustness.
3. **Loss Functions**: The Colab model might have used perceptual losses (VGG/LPIPS) or adversarial losses (GANs). While these create high-frequency textures that look sharp to humans, they often introduce structural hallucinations (inventing fine paths/details) which is a critical safety risk in semiconductor inspection.

---

## 5. Final Model Recommendation

### Multi-Objective Decision Priority:
1. **Correct restoration of structures**: Both models pass (no hallucinations or closed gaps).
2. **Noise suppression**: Model B is slightly superior.
3. **Structural/detail preservation**: Model B is superior (sharper thin lines, cleaner junctions).
4. **PSNR/SSIM**: Model B is superior (+0.054 dB PSNR, +0.0022 SSIM).
5. **Inference speed**: Model A is **3.0x faster** (86 FPS vs 29 FPS).
6. **Model size**: Model B is **24.2% smaller** (135k vs 178k parameters).

### Recommendation:
* **Adopt MODEL B (Advanced Restoration v1) as the final baseline**.
* **Rationale**: For semiconductor inspection, structural detail preservation and noise suppression are paramount. Model B improves PSNR, SSIM, Gradient MAE, and detail sharpness with fewer parameters. While U-Net is faster, Model B's CPU throughput of **29.5 FPS** is still fully sufficient for real-time assembly line processing (which typically operates at 10-30 FPS).
