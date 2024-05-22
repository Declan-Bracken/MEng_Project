import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# class ImageProcessor:
#     def __init__(self, image_path, boxes):
#         self.image_path = image_path
#         self.image = cv2.imread(image_path)
#         if self.image is None:
#             print(f"Failed to load image from {image_path}.")
#             return
#         self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
#         self.boxes = boxes
#         self.cropped_images = self.crop_images()

#     def crop_images(self):
#         cropped_images = []
#         for box in self.boxes:
#             x1, y1, x2, y2 = map(int, box[:4])
#             cropped_image = Image.fromarray(self.image_rgb[y1:y2, x1:x2])
#             cropped_images.append(cropped_image)
#         return cropped_images

class ImageProcessor:
    def __init__(self, image_path, boxes):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            print(f"Failed to load image from {image_path}.")
            return
        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.boxes = boxes
        self.cropped_images = self.crop_images()

    def crop_images(self):
        cropped_images = []
        for box in self.boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            cropped_image = self.image_rgb[y1:y2, x1:x2]
            # preprocessed_image = self.preprocess_image(cropped_image)
            cropped_images.append(Image.fromarray(cropped_image))
        return cropped_images

    def preprocess_image(self, image):
        # Apply Gaussian Blur to reduce noise
        blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Convert to grayscale
        gray_image = cv2.cvtColor(blurred_image, cv2.COLOR_RGB2GRAY)
        
        # Enhance contrast
        alpha = 1.5 # Simple contrast control [1.0-3.0]
        beta = 0    # Simple brightness control [0-100]
        contrast_enhanced_image = cv2.convertScaleAbs(gray_image, alpha=alpha, beta=beta)
        
        # Sharpen the image
        kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
        sharpened_image = cv2.filter2D(contrast_enhanced_image, -1, kernel)
        
        # Convert back to RGB (optional, if you want to keep the 3-channel format)
        processed_image = cv2.cvtColor(sharpened_image, cv2.COLOR_GRAY2RGB)
        
        return processed_image
    
    def plot_images(self):
        if not self.cropped_images:
            print("No images to display.")
            return
        
        num_images = len(self.cropped_images)
        fig, axes = plt.subplots(1, num_images, figsize=(15, 15))
        
        if num_images == 1:
            axes = [axes]  # Ensure axes is iterable if there's only one image
        
        for ax, img in zip(axes, self.cropped_images):
            ax.imshow(img)
            ax.axis('off')
        
        plt.show()

if __name__ == "__main__":
    # Example usage:
    image_path = "/Users/declanbracken/Development/UofT_Projects/Meng_Project/Transcripts/Web_Scraped_Transcripts/transcrip-02102015-1-2048.webp"
    boxes = [(50, 50, 200, 200), (300, 300, 450, 450)]  # Example bounding boxes
    processor = ImageProcessor(image_path, boxes)
    processor.plot_images()
