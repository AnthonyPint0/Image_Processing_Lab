# ----------------------------------------------------------------------------------------------
# 13. Clustering Algorithms: Apply unsupervised clustering algorithms like K-Means and  
#       hierarchical clustering for segmenting image data.

#      ANTHONY PINTO ROBINSON (253MCA37)
#      11/03/2026
# --------------------------------------------------------------------------------------------

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster

image = cv2.imread('Images/cheetha.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

pixel_values = image.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

# 1. K-Means Clustering
k = 3
criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    100,
    0.2
)

_, labels, centers = cv2.kmeans(
    pixel_values,
    k,
    None,
    criteria,
    10,
    cv2.KMEANS_RANDOM_CENTERS
)

centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]
kmeans_image = segmented_data.reshape(image.shape)

# 2. Hierarchical Clustering
sample_pixels = pixel_values[:2000]

Z = linkage(sample_pixels, method='ward')
clusters = fcluster(Z, t=3, criterion='maxclust')

clustered = clusters.reshape(-1, 1)

hierarchical_image = image.copy()
hierarchical_image = hierarchical_image.reshape((-1, 3))
hierarchical_image[:2000] = clusters.reshape(-1, 1) * 80
hierarchical_image = hierarchical_image.reshape(image.shape)

titles = [
    'Original Image',
    'K-Means Segmentation',
    'Hierarchical Clustering'
]

images = [
    image,
    kmeans_image,
    hierarchical_image
]

plt.figure(figsize=(12, 6))

for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis('off')

plt.show()