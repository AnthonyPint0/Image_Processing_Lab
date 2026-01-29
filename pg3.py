""" 
Write a Python Program to Analyze Image Quantization and Visual Degradation at Different Bit Depths

Anthony Pinto Robinson
28/01/2026
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
import os

def quantize_image(image, bits):
  if bits == 8:
    return image.copy()

  num_levels = 2 ** bits
  step_size = 256 / num_levels

  quantized = np.floor(image / step_size) * step_size
  return quantized.astype(np.uint8)

def analyze_quantization(image_path):

  if not os.path.exists(image_path):
    print("Error: Image file not found!")
    return

  img = imread(image_path)

  if img.ndim == 3:
    if img.shape[2] == 4: # Remove alpha channel if present
      img = img[:, :, :3]
    img = rgb2gray(img)
    img = (img * 255).astype(np.uint8)

  bits_per_pixel = [8, 4, 2, 1]

  fig, axes = plt.subplots(1, len(bits_per_pixel), figsize=(18, 5))

  for i, bits in enumerate(bits_per_pixel):
    q_img = quantize_image(img, bits)

    axes[i].imshow(q_img, cmap='gray', vmin=0, vmax=255)
    axes[i].set_title(f"{bits} bits/pixel\n({2**bits} levels)")
    axes[i].axis('off')

    if bits == 8:
      print("8 bpp: Original image (no degradation)")
    elif bits == 4:
      print("4 bpp: Slight banding visible")
    elif bits == 2:
      print("2 bpp: High degradation, posterization visible")
    elif bits == 1:
      print("1 bpp: Severe degradation, binary appearance")

  plt.suptitle("Image Quantization and Visual Degradation Analysis")
  plt.tight_layout()
  plt.show()

image_path = "sunflower.jpg"
analyze_quantization(image_path)