import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Path to the folder containing images
image_folder = "/Users/declanbracken/Development/UofT_Projects/Meng_Project/code_base/Synthetic_Image_Annotation/Training_Data/images"  # Folder where images are extracted

# Lists to store image widths and heights
widths = []
heights = []
i =0
# Read each image and get its dimensions
for image_name in os.listdir(image_folder):
    i += 1
    if image_name.endswith('.jpg') or image_name.endswith('.jpeg'):  # Consider only .jpg and .jpeg images
        image_path = os.path.join(image_folder, image_name)
        with Image.open(image_path) as img:
            width, height = img.size
            widths.append(width)
            heights.append(height)
print(i)
# Calculate median width and height
median_width = np.median(widths)
median_height = np.median(heights)

# Plot the width vs. height distribution
plt.figure(figsize=(10, 6))
plt.scatter(widths, heights, alpha=0.1, color='blue', label='Image Sizes')  # Semi-transparent dots

# Add median lines
plt.axvline(median_width, color='red', linestyle='--', linewidth=1, label=f'Median Width: {median_width:.0f} px')
plt.axhline(median_height, color='green', linestyle='--', linewidth=1, label=f'Median Height: {median_height:.0f} px')

# Customize the plot
plt.title("Distribution of Image Resolutions (Width vs. Height)")
plt.xlabel("Width (pixels)")
plt.ylabel("Height (pixels)")
plt.grid(True)
plt.legend(loc='upper right')

# Display the plot
plt.show()
