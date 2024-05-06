import cv2
from PIL import Image

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
            cropped_image = Image.fromarray(self.image_rgb[y1:y2, x1:x2])
            cropped_images.append(cropped_image)
        return cropped_images
