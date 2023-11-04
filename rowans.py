import cv2
import numpy as np
import os

# -*- coding: utf-8 -*-
"""Untitled74.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1UPtYWIksAUiNdrOIYhuPqrOue3-M2bya
"""


def image_preprocessing(image):
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    eroded_image = cv2.erode(binary_image, kernel, iterations=1)
    blurred_image = cv2.medianBlur(eroded_image, 5)
    kernel = np.ones((3, 3), np.uint8)
    eroded_image = cv2.erode(blurred_image, kernel, iterations=1)
    return eroded_image


def ROI_color(image):
    # preprocessing after reading the image
    boundaries = [([20, 100, 100], [40, 255, 255])]
    height, width, _ = image.shape
    crop_height = height // 2
    cropped_image = image[crop_height:, 200:1100]
    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
    output = 0
    # Loop over the boundaries
    for lower, upper in boundaries:
        # Create NumPy arrays from the boundaries
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        # Find the colors within the specified boundaries and apply the mask
        mask = cv2.inRange(cropped_image, lower, upper)
        output = cv2.bitwise_and(cropped_image, cropped_image, mask=mask)
    return output


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny


for img in os.listdir("./imgs"):
    image = cv2.imread(f"./imgs/{img}")
    img_preprocessed = canny(ROI_color(image_preprocessing(image)))
    cv2.imwrite(f"./imgs_processed/{img}_preprocessed.png", img_preprocessed)
