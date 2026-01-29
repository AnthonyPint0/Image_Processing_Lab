""" 
Write a Python Program to Perform Image Enhancement Using Homomorphic Filtering

Anthony Pinto Robinson
29/01/2026
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def homomorphic_filter(image, sigma=10, gamma_l=0.3, gamma_h=1.5, c=1):
    # 1. Log transform
    img_log = np.log1p(np.array(image, dtype="float"))
    
    # 2. Fourier Transform
    fft = np.fft.fft2(img_log)
    fft_shift = np.fft.fftshift(fft)
    
    # 3. Create High-Pass Filter (Gaussian)
    rows, cols = image.shape
    crow, ccol = rows//2 , cols//2
    x = np.linspace(-ccol, ccol, cols)
    y = np.linspace(-crow, crow, rows)
    X, Y = np.meshgrid(x, y)
    d2 = X**2 + Y**2
    # Gaussian High-Pass Filter
    H = (gamma_h - gamma_l) * (1 - np.exp(-c * d2 / (2 * sigma**2))) + gamma_l
    
    # 4. Apply filter
    filtered_fft = fft_shift * H
    
    # 5. Inverse Fourier Transform
    ifft_shift = np.fft.ifftshift(filtered_fft)
    ifft = np.fft.ifft2(ifft_shift)
    
    # 6. Inverse Log Transform
    result = np.real(np.exp(ifft) - 1)
    return np.clip(result, 0, 255).astype("uint8")

img = cv2.imread('img2.jpeg', 0)

# Apply filter
enhanced_img = homomorphic_filter(img, sigma=20, gamma_l=0.2, gamma_h=1.8)

# Visualization
plt.figure(figsize=(10, 5))
plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Original')
plt.subplot(122), plt.imshow(enhanced_img, cmap='gray'), plt.title('Enhanced')
plt.show()