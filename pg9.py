"""
Program 9. Region-Based Segmentation: Implement region-growing and splitting-merging algorithms
for image segmentation

Anthony Pinto Robinson
04/03/2026
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

image = cv2.imread('Images/landscape.png', cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Image not found")
    sys.exit()
    
# 1️. REGION GROWING
def region_growing(img, seed, threshold=10):
    rows, cols = img.shape
    segmented = np.zeros((rows, cols), np.uint8)
    visited = np.zeros((rows, cols), bool)
    seed_value = img[seed]
    stack = [seed]
    while stack:
        x, y = stack.pop()
        if visited[x, y]:
            continue
        visited[x, y] = True
        if abs(int(img[x, y]) - int(seed_value)) <= threshold:
            segmented[x, y] = 255

            # 4-connected neighbors
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols:
                    stack.append((nx, ny))
    return segmented

# 2️. REGION SPLITTING (Quadtree)
def split_region(img, threshold=100):
    segmented = np.zeros_like(img)
    def split(x, y, size):
        region = img[x:x+size, y:y+size]
        variance = np.var(region)

        if variance < threshold or size < 8:
            segmented[x:x+size, y:y+size] = np.mean(region)
        else:
            half = size // 2
            split(x, y, half)
            split(x, y+half, half)
            split(x+half, y, half)
            split(x+half, y+half, half)
    h, w = img.shape
    size = min(h, w)
    split(0, 0, size)
    return segmented

#  REGION MERGING (Simple)
def merge_regions(img, threshold=10):
    merged = img.copy()
    rows, cols = img.shape
    for i in range(rows - 1):
        for j in range(cols - 1):
            if abs(int(img[i,j]) - int(img[i,j+1])) < threshold:
                avg = (int(img[i,j]) + int(img[i,j+1])) // 2
                merged[i,j] = avg
                merged[i,j+1] = avg
    return merged

# Apply Algorithms
seed_point = (200, 200)
rg_result = region_growing(image, seed_point, threshold=15)
split_result = split_region(image, threshold=100)
merge_result = merge_regions(split_result, threshold=10)

# Display Results
images = [image, rg_result, split_result, merge_result]
titles = ["Original Image", "Region Growing", "Region Splitting", "Region Merging"]

plt.figure(figsize=(12,8))

for i in range(len(images)):
    plt.subplot(2, 2, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()