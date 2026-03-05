"""
Program 16. Noise Removal and Image Enhancement: Apply noise removal techniques like Gaussian,
median, and bilateral filters to enhance image quality. 

Anthony Pinto Robinson
04/03/2026
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('img2.jpeg')

if image is None:
    print("Image not found")
    exit()
    
# Convert BGR to RGB 
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#  Gaussian Filter
gaussian = cv2.GaussianBlur(image_rgb, (5, 5), 0)

#  Median Filter
median = cv2.medianBlur(image_rgb, 5)

#  Bilateral Filter
bilateral = cv2.bilateralFilter(image_rgb, 9, 75, 75)

titles = ["Original Image", "Gaussian Filter", "Median Filter", "Bilateral Filter"]
images = [image_rgb, gaussian, median, bilateral]

plt.figure(figsize=(10, 8))

for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis('off')
    
plt.tight_layout()
plt.show()