from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# Image and algorithm parameters


IMAGE_PATH = str(Path(__file__).resolve().parent / "cameraman.jpg")

PATCH_SIZE = 8
STRIDE = 4

SIGMA = 0.05
RANDOM_SEED = 42

ALPHA = 0.50
GAMMA = 0.45

EPSILON = 1e-12


# Load the image

image = Image.open(IMAGE_PATH)
image = image.convert("L")

original = np.asarray(image, dtype=np.float64)
original = original / 255.0

print("Image dimensions:", original.shape)
print("Minimum pixel value:", original.min())
print("Maximum pixel value:", original.max())


# Generate Gaussian noise

rng = np.random.default_rng(RANDOM_SEED)

noise = rng.normal(
    loc=0.0,
    scale=SIGMA,
    size=original.shape
)

noisy = original + noise
noisy = np.clip(noisy, 0.0, 1.0)


# Patch-based SVD denoising

def svd_denoise(
    noisy_image,
    method,
    patch_size=8,
    stride=4,
    sigma=0.05,
    alpha=0.50,
    gamma=0.45
):

    height, width = noisy_image.shape

    # Store reconstructed patch estimates
    result_sum = np.zeros_like(noisy_image)

    # Count how many patches contribute to each pixel
    count = np.zeros_like(noisy_image)

    # Store singular values for analysis
    singular_value_sum = np.zeros(patch_size)
    modified_value_sum = np.zeros(patch_size)

    # Store energy retention for each patch
    energy_retention = []

    # Base threshold for adaptive soft thresholding
    tau0 = alpha * sigma * (
        np.sqrt(patch_size) + np.sqrt(patch_size)
    )

    # Process overlapping patches
    for row in range(0, height - patch_size + 1, stride):

        for col in range(0, width - patch_size + 1, stride):

            patch = noisy_image[
                row:row + patch_size,
                col:col + patch_size
            ]

            # Calculate SVD
            U, s, VT = np.linalg.svd(
                patch,
                full_matrices=False
            )

            # Rank-3 truncation
            if method == "rank3":

                s_new = np.zeros_like(s)
                s_new[:3] = s[:3]

            # Calculate adaptive threshold for the patch
            elif method in ("adaptive", "hybrid"):

                variance = np.var(patch)

                numerator = max(
                    variance - sigma**2,
                    0.0
                )

                denominator = variance + EPSILON

                activity = numerator / denominator

                threshold = tau0 * (
                    1.0 - gamma * activity
                )

                # Adaptive soft thresholding
                if method == "adaptive":

                    s_new = np.maximum(
                        s - threshold,
                        0.0
                    )

                # Hybrid 1-6-1 method
                elif method == "hybrid":

                    s_new = np.zeros_like(s)

                    # Keep the first singular value unchanged
                    s_new[0] = s[0]

                    # Soft-threshold the middle six singular values
                    s_new[1:7] = np.maximum(
                        s[1:7] - threshold,
                        0.0
                    )

                    # The final singular value remains zero

            else:
                raise ValueError(
                    "Method must be 'rank3', 'adaptive', or 'hybrid'"
                )

            # Collect singular values
            singular_value_sum += s
            modified_value_sum += s_new

            # Calculate the fraction of singular-value energy retained
            original_energy = np.sum(s**2)
            retained_energy = np.sum(s_new**2)

            if original_energy > EPSILON:

                energy_retention.append(
                    retained_energy / original_energy
                )

            # Reconstruct the denoised patch
            patch_result = (U * s_new) @ VT

            # Accumulate overlapping patch estimates
            result_sum[
                row:row + patch_size,
                col:col + patch_size
            ] += patch_result

            # Count how many patches contribute to each pixel
            count[
                row:row + patch_size,
                col:col + patch_size
            ] += 1

    # Average overlapping estimates
    denoised = result_sum / np.maximum(count, 1)

    # Keep original noisy pixels where no patch was processed
    denoised[count == 0] = noisy_image[count == 0]

    denoised = np.clip(denoised, 0.0, 1.0)

    # Calculate average singular values
    total_patches = len(energy_retention)

    average_singular_values = (
        singular_value_sum / total_patches
    )

    average_modified_values = (
        modified_value_sum / total_patches
    )

    average_energy_retention = np.mean(
        energy_retention
    )

    return (
        denoised,
        average_singular_values,
        average_modified_values,
        average_energy_retention
    )


# Apply rank-3 truncation

(
    rank3_result,
    rank3_singular_values,
    rank3_modified_values,
    rank3_energy
) = svd_denoise(
    noisy_image=noisy,
    method="rank3",
    patch_size=PATCH_SIZE,
    stride=STRIDE,
    sigma=SIGMA
)


# Apply adaptive soft thresholding

(
    adaptive_result,
    adaptive_singular_values,
    adaptive_modified_values,
    adaptive_energy
) = svd_denoise(
    noisy_image=noisy,
    method="adaptive",
    patch_size=PATCH_SIZE,
    stride=STRIDE,
    sigma=SIGMA,
    alpha=ALPHA,
    gamma=GAMMA
)


# Apply hybrid 1-6-1 method

(
    hybrid_result,
    hybrid_singular_values,
    hybrid_modified_values,
    hybrid_energy
) = svd_denoise(
    noisy_image=noisy,
    method="hybrid",
    patch_size=PATCH_SIZE,
    stride=STRIDE,
    sigma=SIGMA,
    alpha=ALPHA,
    gamma=GAMMA
)


# Calculate mean squared error

def calculate_mse(reference, reconstructed):

    difference = reference - reconstructed

    squared_difference = difference**2

    mse = np.mean(squared_difference)

    return mse


# Calculate peak signal-to-noise ratio

def calculate_psnr(reference, reconstructed):

    mse = calculate_mse(
        reference,
        reconstructed
    )

    if mse == 0:
        return float("inf")

    psnr = 10 * np.log10(1.0 / mse)

    return psnr


# Calculate image quality metrics

results = {
    "Noisy image": noisy,
    "Rank-3 truncation": rank3_result,
    "Adaptive soft thresholding": adaptive_result,
    "Hybrid 1-6-1": hybrid_result
}

mse_values = {}
psnr_values = {}

print("\nImage Denoising Results")

for name, result in results.items():

    mse = calculate_mse(
        original,
        result
    )

    psnr = calculate_psnr(
        original,
        result
    )

    mse_values[name] = mse
    psnr_values[name] = psnr

    print("\nMethod:", name)
    print("MSE:", f"{mse:.6f}")
    print("PSNR:", f"{psnr:.3f}", "dB")


print("\nAverage Singular-Value Energy Retention")

print(
    "Rank-3 truncation:",
    f"{rank3_energy * 100:.2f}%"
)

print(
    "Adaptive soft thresholding:",
    f"{adaptive_energy * 100:.2f}%"
)

print(
    "Hybrid 1-6-1:",
    f"{hybrid_energy * 100:.2f}%"
)


# PLOT 1: Original, noisy and rank-3 truncation

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(original, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Original Image")

axes[1].imshow(noisy, cmap="gray", vmin=0, vmax=1)
axes[1].set_title("Noisy Image")

axes[2].imshow(rank3_result, cmap="gray", vmin=0, vmax=1)
axes[2].set_title("Rank-3 Truncation")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()


# PLOT 2: Original, noisy and adaptive soft thresholding

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(original, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Original Image")

axes[1].imshow(noisy, cmap="gray", vmin=0, vmax=1)
axes[1].set_title("Noisy Image")

axes[2].imshow(adaptive_result, cmap="gray", vmin=0, vmax=1)
axes[2].set_title("Adaptive Soft Thresholding")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()


# PLOT 3: Original, noisy and hybrid 1-6-1

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(original, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Original Image")

axes[1].imshow(noisy, cmap="gray", vmin=0, vmax=1)
axes[1].set_title("Noisy Image")

axes[2].imshow(hybrid_result, cmap="gray", vmin=0, vmax=1)
axes[2].set_title("Hybrid 1-6-1")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()


# PLOT 4: Average singular values before and after processing

singular_indices = np.arange(1, PATCH_SIZE + 1)

plt.figure(figsize=(9, 5))

plt.plot(
    singular_indices,
    rank3_singular_values,
    marker="o",
    label="Input patches"
)

plt.plot(
    singular_indices,
    rank3_modified_values,
    marker="s",
    label="After rank-3 truncation"
)

plt.plot(
    singular_indices,
    adaptive_modified_values,
    marker="^",
    label="After adaptive thresholding"
)

plt.plot(
    singular_indices,
    hybrid_modified_values,
    marker="d",
    label="After hybrid 1-6-1"
)

plt.xlabel("Singular Value Index")
plt.ylabel("Average Singular Value")

plt.title("Average Singular Values Before and After Denoising")

plt.xticks(singular_indices)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# PLOT 5: Singular-value energy retention

energy_labels = [
    "Rank-3 truncation",
    "Adaptive soft thresholding",
    "Hybrid 1-6-1"
]

energy_values = [
    rank3_energy * 100,
    adaptive_energy * 100,
    hybrid_energy * 100
]

plt.figure(figsize=(9, 5))

bars = plt.bar(
    energy_labels,
    energy_values
)

plt.ylabel("Average Energy Retained (%)")
plt.title("Singular-Value Energy Retention")

plt.ylim(0, 100)

for bar, value in zip(bars, energy_values):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.2f}%",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.show()


# PLOT 6: MSE comparison

method_names = list(mse_values.keys())
mse_results = list(mse_values.values())

plt.figure(figsize=(10, 5))

bars = plt.bar(method_names, mse_results)

plt.ylabel("Mean Squared Error")
plt.title("MSE Comparison")

plt.xticks(rotation=15, ha="right")

for bar, value in zip(bars, mse_results):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.6f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.show()


# PLOT 7: PSNR comparison

method_names = list(psnr_values.keys())
psnr_results = list(psnr_values.values())

plt.figure(figsize=(10, 5))

bars = plt.bar(method_names, psnr_results)

plt.ylabel("PSNR (dB)")
plt.title("PSNR Comparison")

plt.xticks(rotation=15, ha="right")

for bar, value in zip(bars, psnr_results):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.3f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.show()