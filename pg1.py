""" 
Write a python program to perform basic image processing operation using OpenCV

Anthony Pinto Robinson
18/12/2025
"""

import cv2
import numpy as np

def read_image(path):
    img = cv2.imread(path)
    if img is None:
        print("Image not found")
        quit()
    return img

def pixel_operations(img):
    print("Pixel(30,30):", img[30, 30])
    img[30, 30] = [255, 0, 0]  # BGR
    return img

def resize_image(img):
    return cv2.resize(img, (300, 300))

def rotate_image(img, angle):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1)
    return cv2.warpAffine(img, matrix, (w, h))

def transform_image(img):
    h, w = img.shape[:2]
    trans = np.float32([[1, 0, 40], [0, 1, 40]])
    transl = cv2.warpAffine(img, trans, (w, h))
    flip = cv2.flip(img, 0)
    return transl, flip

img = read_image("sunflower.jpg")

cv2.imshow("Original", img)
cv2.waitKey(0)

cv2.imwrite("beach.jpg", img)

img = pixel_operations(img)
cv2.imshow("Pixel Changed", img)
cv2.waitKey(0)

res = resize_image(img)
rot = rotate_image(img, 45)
trans, flip = transform_image(img)

cv2.imshow("Resized", res)
cv2.imshow("Rotated", rot)
cv2.imshow("Translated", trans)
cv2.imshow("Flipped", flip)

cv2.waitKey(0)
cv2.destroyAllWindows()
