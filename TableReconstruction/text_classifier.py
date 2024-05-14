from pytesseract import image_to_data, Output
import numpy as np

class TextClassifier:
    def __init__(self, cropped_images, classes, boxes):
        self.cropped_images = cropped_images
        self.classes = classes
        self.boxes = boxes
        self.headers = []
        self.tables = []
        self.single_row = []
        
        self._classify_text()

    def _classify_text(self):
        for idx, box in enumerate(self.boxes):
            class_type = self.classes[idx]
            text = image_to_data(self.cropped_images[idx], output_type=Output.DICT, config='--psm 6')
            if class_type == 0:
                self.headers.append({'box': box, 'text': text})
            elif class_type == 1:
                self.tables.append({'box': box, 'text': text})
            else:
                self.single_row.append({'box': box, 'text': text})
