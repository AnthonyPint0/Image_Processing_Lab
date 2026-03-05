"""
Program 8. Thresholding for Image Segmentation: Perform global thresholding (Otsu’s method) and
adaptive thresholding for segmenting images. 

Anthony Pinto Robinson
29/01/2026
"""

import cv2
import matplotlib.pyplot as plt
image = cv2.imread('lenagray.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Image not found.")
    exit()
# 1. Global Thresholding using Otsu's Method
_, otsu_thresh = cv2.threshold(image, 0, 255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# 2. Adaptive Thresholding
adaptive_thresh = cv2.adaptiveThreshold( image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,     cv2.THRESH_BINARY, 11, 2)
# Display Results
titles = ['Original Image',
          'Otsu Global Threshold',
          'Adaptive Threshold']
images = [image, otsu_thresh, adaptive_thresh]
plt.figure(figsize=(10, 6))
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()

