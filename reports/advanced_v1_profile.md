# CPU Profiling Report: Model B (Advanced Restoration v1)

This report documents the detailed CPU operation-level profiling of our Restormer-inspired model (`advanced_restoration_v1`) during single-image inference.

---

## 1. Profiling Methodology
- **Input**: Single-channel grayscale tensor of shape `(1, 1, 128, 128)` representing one degraded semiconductor inspection image.
- **Inference Hardware**: CPU Execution.
- **Warm-up**: 10 forward passes to initialize PyTorch cache.
- **Profile Duration**: 50 consecutive forward passes monitored using `torch.autograd.profiler.profile`.

---

## 2. Operation-Level CPU Breakdown

The table below lists the top CPU operations sorted by **Self CPU Time**:

| Operation Name | Self CPU % | Self CPU Time (ms) | CPU Total % | CPU Total Time (ms) | Average CPU Time (us) | Number of Calls |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `aten::mkldnn_convolution` | **35.91%** | 647.335 | 36.90% | 665.151 | 380.086 | 1750 |
| `aten::var` | **25.44%** | 458.555 | 25.47% | 459.133 | 918.266 | 500 |
| `aten::bmm` | 3.93% | 70.819 | 3.94% | 71.063 | 142.125 | 500 |
| `aten::_convolution` | 3.44% | 61.990 | 42.74% | 770.335 | 395.044 | 1950 |
| `aten::add` | 2.98% | 53.669 | 3.35% | 60.326 | 38.920 | 1550 |
| `aten::upsample_bicubic2d` | 2.94% | 53.029 | 2.99% | 53.979 | 1080.000 | 50 |
| `aten::div` | 2.77% | 49.937 | 2.77% | 49.937 | 49.937 | 1000 |
| `aten::mul` | 2.25% | 40.497 | 2.25% | 40.497 | 40.497 | 1000 |
| `aten::copy_` | 2.02% | 36.423 | 2.02% | 36.423 | 31.672 | 1150 |
| `aten::gelu` | 1.92% | 34.672 | 1.92% | 34.672 | 138.690 | 250 |
| `aten::convolution` | 1.33% | 24.018 | 44.07% | 794.353 | 407.361 | 1950 |
| `aten::linalg_vector_norm` | 1.19% | 21.505 | 1.19% | 21.505 | 43.010 | 500 |
| `aten::native_batch_norm` | 1.13% | 20.389 | 1.26% | 22.671 | 90.683 | 250 |
| `aten::sum` | 1.01% | 18.179 | 1.21% | 21.783 | 43.565 | 500 |
| `aten::sub` | 1.00% | 18.077 | 1.00% | 18.077 | 36.155 | 500 |
| `aten::empty` | 0.85% | 15.281 | 0.85% | 15.281 | 2.231 | 6850 |

---

## 3. Analysis & Key Bottlenecks

1. **Convolution Performance (`aten::mkldnn_convolution`)**:
   - Convolutions account for the largest share of computation (**35.91%**). This is expected and represents the core feature extraction weight of the network. PyTorch automatically uses the Intel MKL-DNN library to optimize these kernels on CPU.
2. **Variance Bottleneck (`aten::var`)**:
   - Variance calculation accounts for **25.44%** of the entire execution time.
   - This occurs inside our custom `LayerNorm2d` module:
     ```python
     std = x.var(dim=1, keepdim=True, unbiased=False).add(self.eps).sqrt()
     ```
   - In the NCHW format, the channel dimension (`dim=1`) is non-contiguous in memory (values are separated by the spatial size $HW = 16384$ floats). Reduction along non-contiguous dimensions prevents cache locality and SIMD vectorization, leading to massive CPU memory-access latency.
3. **Other Operations**:
   - Matrix multiplication (`aten::bmm` for Transposed channel attention) only accounts for **3.93%** of CPU time, verifying that channel-wise attention has linear scaling.
   - Bicubic interpolation (`aten::upsample_bicubic2d`) takes **2.94%**.
   - Reshaping and copying operations are extremely lightweight ($< 1\%$).

---

## 4. Targeted Optimization Proposed
To eliminate the `aten::var` bottleneck, we will permute the tensor layout to channel-last `(B, H, W, C)` prior to normalization. In channel-last layout, the channel dimension is contiguous in memory. This allows us to use PyTorch's native C++ optimized `F.layer_norm` operator, which will yield a significant speedup.
