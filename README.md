Image Processing with Histograms and Intensity Transformations
Overview
This project demonstrates fundamental image processing techniques using histograms, intensity transformations, and thresholding for image segmentation. The techniques covered include histogram generation and visualization, intensity transformations (such as image negation, brightness, and contrast adjustments), and thresholding for image segmentation. These methods enhance image characteristics, making objects easier to analyze and segment.

Objectives
Understand and create histograms for image analysis.
Apply intensity transformations to improve image contrast and brightness.
Perform thresholding for image segmentation.
Setup
Prerequisites
Ensure you have the following libraries installed:

matplotlib
opencv-python
numpy
Installation

You can install the required libraries using pip:
pip install matplotlib opencv-python numpy

Usage
Import Necessary Libraries
import matplotlib.pyplot as plt
import cv2
import numpy as np

Helper Functions
Define helper functions for plotting images and histograms.
def plot_image(image_1, image_2, title_1="Original", title_2="New Image"):
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(image_1, cmap="gray")
    plt.title(title_1)
    plt.subplot(1, 2, 2)
    plt.imshow(image_2, cmap="gray")
    plt.title(title_2)
    plt.show()

def plot_hist(old_image, new_image, title_old="Original", title_new="New Image"):
    intensity_values = np.array([x for x in range(256)])
    plt.subplot(1, 2, 1)
    plt.bar(intensity_values, cv2.calcHist([old_image], [0], None, [256], [0, 256])[:, 0], width=5)
    plt.title(title_old)
    plt.xlabel('intensity')
    plt.subplot(1, 2, 2)
    plt.bar(intensity_values, cv2.calcHist([new_image], [0], None, [256], [0, 256])[:, 0], width=5)
    plt.title(title_new)
    plt.xlabel('intensity')
    plt.show()

Histograms
Histograms display the intensity of an image and are useful for understanding and manipulating images.

Toy Example
Create a toy array and generate its histogram.

toy_image = np.array([[0, 2, 2], [1, 1, 1], [1, 1, 2]], dtype=np.uint8)
plt.imshow(toy_image, cmap="gray")
plt.show()

Gray Scale Histograms
Create a histogram for a grayscale image.

goldhill = cv2.imread("goldhill.bmp", cv2.IMREAD_GRAYSCALE)
plt.figure(figsize=(10, 10))
plt.imshow(goldhill, cmap="gray")
plt.show()

hist = cv2.calcHist([goldhill], [0], None, [256], [0, 256])
intensity_values = np.array([x for x in range(hist.shape[0])])
plt.bar(intensity_values, hist[:, 0], width=5)
plt.title("Bar histogram")
plt.show()

Intensity Transformations
Apply transformations to enhance image characteristics.

Image Negatives
neg_toy_image = 255 - toy_image
plot_image(toy_image, neg_toy_image)

Brightness and Contrast Adjustments
alpha = 1  # Contrast control
beta = 100  # Brightness control   
new_image = cv2.convertScaleAbs(goldhill, alpha=alpha, beta=beta)
plot_image(goldhill, new_image, title_1="Original", title_2="Brightness Control")

Histogram Equalization
Improve image contrast by flattening the histogram.
zelda = cv2.imread("zelda.png", cv2.IMREAD_GRAYSCALE)
new_image = cv2.equalizeHist(zelda)
plot_image(zelda, new_image, title_1="Original", title_2="Histogram Equalization")

Thresholding and Simple Segmentation
Segment objects from images using thresholding.

Toy Example
Apply thresholding to a toy image.
def thresholding(input_img, threshold, max_value=255, min_value=0):
    N, M = input_img.shape
    image_out = np.zeros((N, M), dtype=np.uint8)
    for i in range(N):
        for j in range(M):
            if input_img[i, j] > threshold:
                image_out[i, j] = max_value
            else:
                image_out[i, j] = min_value
    return image_out

thresholding_toy = thresholding(toy_image, threshold=1, max_value=2, min_value=0)
plot_image(toy_image, thresholding_toy)

Real Image
Apply thresholding to the "cameraman" image.
image = cv2.imread("cameraman.jpeg", cv2.IMREAD_GRAYSCALE)
threshold = 87
max_value = 255
min_value = 0
new_image = thresholding(image, threshold, max_value, min_value)
plot_image(image, new_image, title_1="Original", title_2="After Thresholding")


This project demonstrates essential techniques in image processing, including histograms, intensity transformations, and thresholding. These methods are foundational for enhancing and analyzing images in various applications, such as medical imaging and industrial inspections.
