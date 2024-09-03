#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 20:35:28 2024

@author: declanbracken
"""
from PIL import Image
import pytesseract
import numpy as np

filename = '/Users/declanbracken/Development/UofT_Projects/Meng_Project/Transcripts/Test transcripts/transcripts-2-2048.png'
filename = '/Users/declanbracken/Development/UofT_Projects/Meng_Project/Transcripts/Test transcripts/2-2048-cropped.png'
img = Image.open(filename)
# img.show()
# Convert the image to numpy array and get its dimensions
img_array = np.array(img)
height, width = img_array.shape[:2]

# Calculate the center to split the image into two halves
center = width // 2

# Split the image into left an d right halves
left_half = img_array[:, :center]
right_half = img_array[:, center:]

# Convert numpy arrays back to PIL images
left_img = Image.fromarray(left_half)
right_img = Image.fromarray(right_half)

# Apply OCR to each half
full_text = pytesseract.image_to_string(img, config = "--psm 6")
# left_text = pytesseract.image_to_string(left_img)
# right_text = pytesseract.image_to_string(right_img)

print("Full Page")
print(full_text)
# Print the extracted text from both halves
# print("Left Half Text:")
# print(left_text)
# print("\nRight Half Text:")
# print(right_text)

