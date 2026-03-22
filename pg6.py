"""
Program 6. Image Restoration: Simulate image degradation (motion blur, Gaussian noise) and restore
images using Wiener filtering and inverse filtering techniques. 
    
Anthony Pinto Robinson
26/02/2026
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from numpy.fft import fft2, ifft2, ifftshift

def add_gaussian_noise(image, sigma):
    noise = np.random.normal(0, sigma, image.shape)
    return np.clip(image + noise, 0, 1)

def motion_blur_kernel(size, angle):
    kernel = np.zeros((size, size))
    center = size // 2
    cv2.line(kernel,(center, center),
             (center + int(np.cos(np.deg2rad(angle)) * center),
              center - int(np.sin(np.deg2rad(angle)) * center)),
             1, 1)
    return kernel / np.sum(kernel)

def inverse_filter(degraded, psf):
    G = fft2(degraded)
    
    psf_padded = np.zeros_like(degraded)
    psf_padded[:psf.shape[0], :psf.shape[1]] = psf
    psf_padded = ifftshift(psf_padded)
    
    H = fft2(psf_padded)
    epsilon = 1e-8
    
    F = G / (H + epsilon)
    return np.clip(np.abs(ifft2(F)), 0, 1)

def wiener_filter(degraded, psf, K):
    G = fft2(degraded)
    
    psf_padded = np.zeros_like(degraded)
    psf_padded[:psf.shape[0], :psf.shape[1]] = psf
    psf_padded = ifftshift(psf_padded)
    
    H = fft2(psf_padded)
    H_conj = np.conj(H)
    
    F = (H_conj / (np.abs(H)**2 + K)) * G
    return np.clip(np.abs(ifft2(F)), 0, 1)

original = cv2.imread('Images/someflower.png')

if original is None:
    print("Image not found")
    exit()

original = original.astype(np.float64) / 255.0

psf = motion_blur_kernel(21, 11)

print("PSF:\n", psf)

blurred = convolve2d(original, psf, mode='same', boundary='symm')

degraded = add_gaussian_noise(blurred, 0.01)

inverse_restored = inverse_filter(degraded, psf)

wiener_restored = wiener_filter(degraded, psf, 0.001)

# Display results
images = [original, degraded, inverse_restored, wiener_restored]
titles = ['Original', 'Degraded', 'Inverse Filter', 'Wiener Filter']

plt.figure(figsize=(12, 8))
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
