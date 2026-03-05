""""
Program 6. Image Restoration: Simulate image degradation (motion blur, Gaussian noise) and restore
images using Wiener filtering and inverse filtering techniques. 

Anthony Pinto Robinson
29/01/2026
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from numpy.fft import fft2, ifft2, fftshift, ifftshift

# -----------------------------
# Noise Function
# -----------------------------
def add_gaussian_noise(image, sigma):
    noise = np.random.normal(0, sigma, image.shape)
    noisy = image + noise
    return np.clip(noisy, 0, 1)

# -----------------------------
# Motion Blur PSF (Corrected)
# -----------------------------
def motion_blur_kernel(size, angle):
    kernel = np.zeros((size, size))
    center = size // 2

    # Symmetric line across center
    x1 = center - int(np.cos(np.deg2rad(angle)) * center)
    y1 = center + int(np.sin(np.deg2rad(angle)) * center)
    x2 = center + int(np.cos(np.deg2rad(angle)) * center)
    y2 = center - int(np.sin(np.deg2rad(angle)) * center)

    cv2.line(kernel, (x1, y1), (x2, y2), 1, 1)

    kernel /= np.sum(kernel)
    return kernel

# -----------------------------
# Pad PSF correctly for FFT
# -----------------------------
def psf2otf(psf, shape):
    psf_padded = np.zeros(shape)
    psf_padded[:psf.shape[0], :psf.shape[1]] = psf
    psf_padded = fftshift(psf_padded)
    return fft2(psf_padded)

# -----------------------------
# Inverse Filter (Improved)
# -----------------------------
def inverse_filter(degraded, psf, threshold=1e-3):
    G = fft2(degraded)
    H = psf2otf(psf, degraded.shape)

    # Avoid division where |H| is too small
    H_abs = np.abs(H)
    H[H_abs < threshold] = threshold

    F_hat = G / H
    restored = np.abs(ifft2(F_hat))
    return np.clip(restored, 0, 1)

# -----------------------------
# Wiener Filter
# -----------------------------
def wiener_filter(degraded, psf, K):
    G = fft2(degraded)
    H = psf2otf(psf, degraded.shape)
    H_conj = np.conj(H)

    F_hat = (H_conj / (np.abs(H)**2 + K)) * G
    restored = np.abs(ifft2(F_hat))
    return np.clip(restored, 0, 1)

# -----------------------------
# Main Program
# -----------------------------
original = cv2.imread('black-and-white-eye.jpg', 0)

if original is None:
    raise FileNotFoundError("Image 'black-and-white-eye.jpg' not found.")

original = original.astype(np.float64) / 255.0

# Create motion blur PSF
psf = motion_blur_kernel(21, 15)

# Blur image
blurred = convolve2d(original, psf, mode='same', boundary='symm')

# Add noise
degraded = add_gaussian_noise(blurred, 0.01)

# Restore
inverse_restored = inverse_filter(degraded, psf)
wiener_restored = wiener_filter(degraded, psf, K=0.001)

# -----------------------------
# Display Results
# -----------------------------
images = [original, degraded, inverse_restored, wiener_restored]
titles = ['Original', 'Degraded (Blur+Noise)', 'Inverse Filter', 'Wiener Filter']

plt.figure(figsize=(12, 8))
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()