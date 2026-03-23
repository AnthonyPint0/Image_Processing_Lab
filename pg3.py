import cv2
import numpy as np
import matplotlib.pyplot as plt

def quantize(img, bits):
    levels = 2 ** bits
    step = 256 // levels
    return (img // step) * step

# 1. Read image
img = cv2.imread('Images/eye.png', 0)  # directly grayscale

# 2. Quantization levels
bits_list = [8, 4, 2, 1]

# 3. Process + Display
plt.figure(figsize=(10,5))

for i, b in enumerate(bits_list):
    q = quantize(img, b)

    plt.subplot(1, 4, i+1)
    plt.imshow(q, cmap='gray', vmin=0, vmax=255)
    plt.title(f"{b} bits")
    plt.axis('off')

plt.tight_layout()
plt.show()