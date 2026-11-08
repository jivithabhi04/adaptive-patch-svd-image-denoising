# Adaptive Patch-Based SVD Image Denoising

An image-denoising project that uses **Singular Value Decomposition (SVD)** on overlapping image patches to reduce additive Gaussian noise. The implementation compares three singular-value processing strategies:

1. **Rank-3 truncation** – retains the first three singular values and removes the rest.
2. **Adaptive soft thresholding** – shrinks singular values using a threshold adjusted according to local patch variance.
3. **Hybrid 1-6-1** – keeps the first singular value, soft-thresholds the next six, and removes the final singular value.

The project was completed for **EC6103E – Applied Mathematics for Electronics Design**, NIT Calicut (Monsoon 2026).

## Repository contents

| File | Description |
|---|---|
| `SVD.py` | Main Python implementation |
| `cameraman.jpg` | 512 × 512 grayscale test image |
| `Report.pdf` | Project report |
| `SVD_Denoising_Presentation_Graphite_Amber.pptx` | Project presentation |
| `main.tex` | LaTeX source included with the project |
| `fig_*.png` | Generated result and analysis figures |
| `console_output.png` | Program output screenshot |
| `Math Project 1.docx` | Supplementary document |
| `Math Project 1 video.pdf` | Supplementary PDF |

> The recorded presentation video is not included because it exceeds GitHub's standard 100 MB per-file limit. Add it separately using Git LFS or a video-hosting link if you want to share it.

## Method overview

### 1. Image preparation and noise model

The input image is converted to grayscale and normalized to the range `[0, 1]`. Zero-mean Gaussian noise is added:

\[
Y = X + N,\qquad N \sim \mathcal{N}(0,\sigma^2)
\]

The experiment uses `sigma = 0.05` and random seed `42` for reproducibility. The noisy image is clipped to `[0, 1]`.

### 2. Patch extraction and SVD

The 512 × 512 image is processed using overlapping **8 × 8 patches** with a stride of **4 pixels**. This produces 127 × 127 = **16,129 patches**.

Each patch is decomposed as:

\[
P = U\Sigma V^T
\]

The singular values are modified according to the selected method, and the patch is reconstructed using:

\[
\hat P = U\hat\Sigma V^T
\]

Overlapping reconstructed patches are accumulated and averaged to obtain the denoised image.

### 3. Adaptive threshold

For adaptive and hybrid methods, the base threshold is:

\[
\tau_0 = \alpha\sigma(\sqrt{P}+\sqrt{P})
\]

where the patch dimension is \(P=8\). For each patch:

\[
A_p = \frac{\max(\operatorname{var}(p)-\sigma^2,0)}
{\operatorname{var}(p)+\epsilon}
\]

\[
\tau_p = \tau_0(1-\gamma A_p)
\]

The implementation uses `alpha = 0.50`, `gamma = 0.45`, and `epsilon = 1e-12`. Singular values are soft-thresholded as:

\[
\hat{s}_i = \max(s_i-\tau_p,0)
\]

## Reported results

The following values are taken from the submitted project report for the 512 × 512 `cameraman.jpg` experiment with Gaussian noise standard deviation `0.05`.

| Method | MSE | PSNR (dB) | Singular-value energy retained |
|---|---:|---:|---:|
| Noisy image | 0.002382 | 26.231 | — |
| Rank-3 truncation | 0.001574 | 28.029 | 98.22% |
| Adaptive soft thresholding | 0.000999 | 30.006 | 83.94% |
| Hybrid 1-6-1 | **0.000834** | **30.789** | 94.60% |

In this reported experiment, the hybrid 1-6-1 method achieved the highest PSNR and lowest MSE among the compared methods.

## Requirements

- Python 3.9 or later
- NumPy
- Matplotlib
- Pillow

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Run the project

Clone or download this repository. Keep `SVD.py` and `cameraman.jpg` in the same directory, then run:

```bash
python SVD.py
```

The script prints MSE, PSNR, and average singular-value energy-retention results, and displays image comparisons and analysis plots.

## Parameters

The main experiment settings are defined near the top of `SVD.py`:

```python
PATCH_SIZE = 8
STRIDE = 4
SIGMA = 0.05
RANDOM_SEED = 42
ALPHA = 0.50
GAMMA = 0.45
EPSILON = 1e-12
```

You can change these values to explore different patch sizes, noise levels, and threshold settings.

## Evaluation metrics

**Mean Squared Error (MSE)** measures the average squared difference between the original and reconstructed images:

\[
\mathrm{MSE}=\frac{1}{MN}\sum_{i=1}^{M}\sum_{j=1}^{N}
(X_{ij}-\hat X_{ij})^2
\]

**Peak Signal-to-Noise Ratio (PSNR)** is calculated using peak intensity 1 because the images are normalized to `[0, 1]`:

\[
\mathrm{PSNR}=10\log_{10}\left(\frac{1}{\mathrm{MSE}}\right)\ \mathrm{dB}
\]

## Limitations and notes

- The experiment uses a known synthetic Gaussian noise level; it does not estimate an unknown noise level from a real input.
- The adaptive activity measure is based on local patch variance and is a heuristic.
- Pixels not covered by the patch extraction grid retain their noisy input values.
- Results depend on image content, noise realization, patch size, stride, and threshold parameters.
- The reported table is reproduced from the project report. Running the included script is the appropriate way to generate results for the exact code and environment in this repository.

## Tools and libraries

- Python
- NumPy
- Matplotlib
- Pillow
- Singular Value Decomposition (SVD)

## Author

**R. Jivith Abhishek**  
M.Tech, Electronics and Communication Engineering  
National Institute of Technology Calicut

---

This repository contains academic project material and is shared for learning and reference.
