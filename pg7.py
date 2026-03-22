"""
Program 7. Edge Detection: Implement edge detection algorithms like Sobel, Prewitt, Roberts, and
Canny. 

Anthony Pinto Robinson
18/02/2026
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('Images/bw_flower.png', cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: Image not found.")
    exit()

def roberts_edge_detection(image):
    kernel_x = np.array([[ 1, 0],
                         [ 0,-1]])  
    kernel_y = np.array([[ 0, 1],
                         [-1, 0]])
    roberts_x = cv2.filter2D(image, cv2.CV_64F, kernel_x)
    roberts_y = cv2.filter2D(image, cv2.CV_64F, kernel_y)
    roberts = cv2.magnitude(np.float32(roberts_x), np.float32(roberts_y))
    return roberts

def prewitt_edge_detection(image):
    kernel_x = np.array([[ -1, 0, 1],
                         [ -1, 0, 1],
                         [ -1, 0, 1]])
    kernel_y = np.array([[ 1, 1, 1],
                         [ 0, 0, 0],
                         [-1,-1,-1]])
    prewitt_x = cv2.filter2D(image, cv2.CV_64F, kernel_x)
    prewitt_y = cv2.filter2D(image, cv2.CV_64F, kernel_y)
    prewitt = cv2.magnitude(np.float32(prewitt_x), np.float32(prewitt_y))
    return prewitt

def sobel_edge_detection(image):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(np.float32(sobel_x), np.float32(sobel_y))
    return sobel

def canny_edge_detection(image):
    return cv2.Canny(image, 100, 200)

images = [image,
          roberts_edge_detection(image),
          prewitt_edge_detection(image),
          sobel_edge_detection(image),
          canny_edge_detection(image)]

titles = ['Original',
          'Sobel',
          'Prewitt',
          'Roberts',
          'Canny']

plt.figure(figsize=(12, 8))

for i in range(5):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
