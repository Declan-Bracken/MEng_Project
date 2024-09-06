import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

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
            cropped_image = self.image_rgb[y1:y2, x1:x2]
            # preprocessed_image = self.preprocess_image(cropped_image)
            cropped_images.append(Image.fromarray(cropped_image))
        return cropped_images
    
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
