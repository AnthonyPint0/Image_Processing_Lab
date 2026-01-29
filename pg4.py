""" 
Write a Python Program to Perform Image Enhancement Using Intensity Transformations

Anthony Pinto Robinson
29/01/2026
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('img2gray.jpeg')
if img is None:
    print("Error: Image not found!")
    exit()
if len(img.shape) == 2:
    is_gray = True
    img_proc = img.copy()
else:
    is_gray = False
    img_proc = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_float = img_proc.astype(np.float32)

def negative(image):
    return 255 - image

def log_transform(image):
    c = 255 / np.log(1 + np.max(image))
    log_img = c * np.log(1 + image)
    return np.clip(log_img, 0, 255).astype(np.uint8)

def power_law(image, gamma=0.5):
    power = 255 * ((image / 255) ** gamma)
    return np.clip(power, 0, 255).astype(np.uint8)

def contrast_stretch(image):
    r_min = np.min(image)
    r_max = np.max(image)
    stretched = ((image - r_min) / (r_max - r_min)) * 255
    return np.clip(stretched, 0, 255).astype(np.uint8)

negative_img = negative(img_proc)
log_img = log_transform(img_float)
power_img = power_law(img_float, gamma=0.5)
stretch_img = contrast_stretch(img_float)

plt.figure(figsize=(12, 10))

plt.subplot(2, 3, 1)
plt.imshow(img_proc, cmap='gray' if is_gray else None)
plt.title("Original Image")
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(negative_img, cmap='gray' if is_gray else None)
plt.title("Negative")
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(log_img, cmap='gray' if is_gray else None)
plt.title("Log Transformation")
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(power_img, cmap='gray' if is_gray else None)
plt.title("Power-Law Transformation")
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(stretch_img, cmap='gray' if is_gray else None)
plt.title("Contrast Stretching")
plt.axis('off') 

plt.tight_layout()
plt.show()

if not is_gray:
    cv2.imwrite('negative.jpg', cv2.cvtColor(negative_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite('log.jpg', cv2.cvtColor(log_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite('power_law.jpg', cv2.cvtColor(power_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite('contrast_stretch.jpg', cv2.cvtColor(stretch_img, cv2.COLOR_RGB2BGR))
else:
    cv2.imwrite('negative.jpg', negative_img)
    cv2.imwrite('log.jpg', log_img)
    cv2.imwrite('power_law.jpg', power_img)
    cv2.imwrite('contrast_stretch.jpg', stretch_img)