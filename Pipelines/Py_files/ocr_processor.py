from PIL import Image
import pytesseract
import easyocr
import cv2
import os
from tqdm import tqdm

class OCRProcessor:
    def __init__(self):
        self.class_names = {0: 'grade headers', 1: 'grade table', 2: 'single row table'}
    
    def init_easyocr(self, reader = None):
        # Initialize EasyOCR reader; specify language if different from English
        self.reader = reader if reader else easyocr.Reader(['en'])

    def tesseract_get_string(self, image, config):
        # Run Pytesseract
        return pytesseract.image_to_string(image, config=config)
    
    def easyocr_get_string(self,image):
        # Use EasyOCR to extract text from cropped image
        return self.reader.readtext(image, detail=0, paragraph=True)

    def process_images_with_ocr(self, results, image_directory, default_config="--psm 7", use_tesseract = 1):
        """
        Processes images by cropping based on bounding boxes, applying OCR, and organizing the results
        by image name. Each image's results include bounding box coordinates, class name, confidence level,
        and OCR-extracted text.

        Parameters:
            results (list): A list of dictionaries containing bounding boxes and class IDs. Each dictionary
                            should have a 'boxes' key with 'data' and 'cls' sub-keys.
            image_directory (str): Path to the directory containing the images used for detection.

        Returns:
            dict: A dictionary where each key is an image name, and the value is a list of dictionaries
                for each bounding box's data including class name, confidence, OCR text, and coordinates.
        """
        if os.path.isdir(image_directory):
            onlyfiles = [os.path.join(image_directory, file) for file in os.listdir(image_directory)]
        else:
            onlyfiles = [image_directory]  # Handle single image

        processed_results = {}
        for result, image_path in tqdm(zip(results, onlyfiles), desc = "Running OCR Engine"):
            image_name = os.path.basename(image_path)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image from {image_path}.")
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            boxes = result.boxes.data.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            image_results = []
            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box[:4])
                cropped_image = image_rgb[y1:y2, x1:x2]
                class_id = int(classes[idx])
                class_name = self.class_names[class_id]
                confidence = box[4]
                
                # Get cropped Image
                pil_cropped_image = Image.fromarray(cropped_image)
                
                if use_tesseract == 1:
                    # Set configuration for tesseract
                    config = default_config if class_name != 'grade table' else "--psm 6"
                    text_extracted = self.tesseract_get_string(pil_cropped_image, config=config)
                else:
                    text_extracted = self.easyocr_get_string(cropped_image)
                    print(text_extracted)

                image_results.append({
                    "class_name": class_name,
                    "confidence": confidence,
                    "text": text_extracted.strip(),
                    "bounding_box": [x1, y1, x2, y2]
                })

            processed_results[image_name] = image_results

        return processed_results

    def format_strings(self, processed_results):
        """
        Processes OCR results to format text strings for LLM processing.
        """
        formatted_data = {}
        for image_name, results in processed_results.items():
            table_texts = []
            header_texts = []

            for result in results:
                class_name = result['class_name']
                text = result['text']

                if class_name in ['grade table', 'single row table']:
                    table_texts.append(text)
                elif class_name == 'grade headers':
                    header_texts.append(text)

            table_data = '\n'.join(table_texts)
            header_data = '\n'.join(header_texts)

            formatted_data[image_name] = {
                'table_data': table_data,
                'header_data': header_data
            }

        return formatted_data