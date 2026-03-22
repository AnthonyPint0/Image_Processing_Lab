""" 
Program 4. Spatial Domain Filtering: Apply smoothing filters (mean, median, Gaussian) and sharpening
filters (Laplacian, high-pass) for noise reduction and enhancement. 

Anthony Pinto Robinson
29/01/2026
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('Images/sunflower.jpg')

if img is None:
    print("Error: Image not found!")
    exit()
    
def mean_filter(image):
    kernel = np.ones((5, 5), np.float32) / 25
    return cv2.filter2D(image, -1, kernel)

def median_filter(image, kernel_size=5):
    return cv2.medianBlur(image, kernel_size)

def gaussian_filter(image, kernel_size=5, sigma=0):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

def laplacian_filter(image):
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    return np.uint8(np.absolute(laplacian))

def high_pass_filter(image):
    kernel = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)

images = [
    img,
    mean_filter(img),
    median_filter(img),
    gaussian_filter(img),
    laplacian_filter(img),
    high_pass_filter(img)
]

titles = [
    "Original Image",
    "Mean Filter",
    "Median Filter",
    "Gaussian Filter",
    "Laplacian Filter",
    "High-Pass Filter"
]

plt.figure(figsize=(18, 12))
for i in range(len(images)):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()