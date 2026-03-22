""" 
Program 2. Gray-Level Transformations: Implement contrast stretching, logarithmic, power-law, and
negative transformations on images. 

Anthony Pinto Robinson
21/01/2026
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('Images/sunflower.jpg')

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

images = [img_proc, negative_img, log_img, power_img, stretch_img]
titles = [
    "Original Image",
    "Negative",
    "Log Transformation",
    "Power-Law Transformation",
    "Contrast Stretching"
]

for i in range(len(images)):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], cmap='gray' if is_gray else None)
    plt.title(titles[i])
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