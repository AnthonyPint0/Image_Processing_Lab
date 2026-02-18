# Write a Python program to apply and compare Sobel, Prewitt, Roberts, and Canny edge detection techniques on a grayscale image.

import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('lenagray.jpg', cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: Image not found.")
    exit()

# 1. Sobel Edge Detection
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.magnitude(sobel_x, sobel_y)

# 2. Prewitt Edge Detection (Manual Kernel)
kernelx = np.array([[1, 0, -1],
                    [1, 0, -1],
                    [1, 0, -1]])

kernely = np.array([[1,  1,  1],
                    [0,  0,  0],
                    [-1, -1, -1]])

prewitt_x = cv2.filter2D(image, -1, kernelx)
prewitt_y = cv2.filter2D(image, -1, kernely)
prewitt = cv2.magnitude(np.float32(prewitt_x), np.float32(prewitt_y))

# 3. Roberts Edge Detection
roberts_x = np.array([[1, 0],
                      [0, -1]])

roberts_y = np.array([[0, 1],
                      [-1, 0]])

roberts_x_img = cv2.filter2D(image, -1, roberts_x)
roberts_y_img = cv2.filter2D(image, -1, roberts_y)
roberts = cv2.magnitude(np.float32(roberts_x_img),
                        np.float32(roberts_y_img))
# 4. Canny Edge Detection
canny = cv2.Canny(image, 100, 200)

# Display Results
titles = ['Original',
          'Sobel',
          'Prewitt',
          'Roberts',
          'Canny']

images = [image,
          sobel,
          prewitt,
          roberts,
          canny]

plt.figure(figsize=(12, 8))

for i in range(5):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
