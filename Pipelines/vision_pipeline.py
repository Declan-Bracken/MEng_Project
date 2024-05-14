from ultralytics import YOLO
from PIL import Image
import pytesseract
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import os
# from ocr_processor import OCRProcessor

class vision_pipeline():
  def __init__(self, path_to_cnn):
    self.path_to_cnn = path_to_cnn
    # Class mapping
    self.class_names = {0: 'grade headers', 1: 'grade table', 2: 'single row table'}
    self.class_colors = {0: 'g', 1: 'r', 2: 'b'}
    # Initialize Model
    self._init_yolo()

  def _init_yolo(self):
    """
    Initializes the YOLO object detection model.

    Raises:
        Exception: Describes the error if the model fails to load.
    """
    print("Loading Vision Model...")
    try:
      self.object_detector = YOLO(self.path_to_cnn)
      print("Model loaded successfully.")
    except Exception as e:
      print("Failed to load model:", e)
  
  # Function for visualizing bounding boxes
  def visualize_boxes(self, image_path, boxes, classes, names):
    """
    Visualizes bounding boxes on an image and displays it.

    Parameters:
        image_path (str): Path to the image file.
        boxes (list of lists): Each sublist contains coordinates [xmin, ymin, xmax, ymax] and confidence scores.
        classes (list of int): List of class indices corresponding to the bounding boxes.
        names (list of str): List of class names corresponding to the indices in classes.
    """

    # Load image
    image = cv2.imread(image_path)
    # print("Image path:", image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create figure and axes
    _, ax = plt.subplots(1, figsize=(8, 8))

    # Display the image
    ax.imshow(image)

    # Add bounding boxes
    for i, box in enumerate(boxes):
      # box = [xmin, ymin, xmax, ymax, confidence, class_confidence]
      x1, y1, x2, y2 = box[:4]
      class_id = int(classes[i])  # Get class index, convert to integer if not already
      label = names[class_id]  # Get class name using class_id
      confidence = box[4]  # Confidence score
      rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=self.class_colors[class_id], 
                               facecolor='none')
      ax.add_patch(rect)
      # Place text inside the bounding box
      ax.text(x1, y1, f'{label} {confidence:.2f}', color='white', fontsize=10, 
              bbox=dict(facecolor=self.class_colors[class_id], alpha=0.5))

    plt.axis('off')
    plt.show()

  def predict(self, image_directory, plot = False, **kwargs):
    """
    Predicts bounding boxes for all images in a specified directory using a preloaded model.

    Parameters:
        image_directory (str): The directory containing images to predict on.
        plot (bool): If True, visualizes the bounding boxes on each image.
        **kwargs: Variable keyword arguments for model prediction settings, including 'iou' and 'conf'.

    Returns:
        list: A list of prediction results, one for each image processed.
    """
    if os.path.isdir(image_directory):
      # Get all files as a list:
      onlyfiles = [image_directory + '/' + file for file in os.listdir(image_directory)]
    else:
      onlyfiles = [image_directory]

    # Model predictions
    results = self.object_detector.predict(onlyfiles, **kwargs)
    if plot:
      for result, image_path in zip(results, onlyfiles):
        classes = result.boxes.cls.cpu().numpy()  # Move to CPU and convert to numpy
        boxes = result.boxes.data.cpu().numpy()   # Move to CPU and convert to numpy
        self.visualize_boxes(image_path, boxes, classes, self.class_names)
    
    return results
  
  def process_images_with_ocr(self, results, image_directory, config="--psm 6"):
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

    Configurations for OCR:
    0    Orientation and script detection (OSD) only.
    1    Automatic page segmentation with OSD.
    2    Automatic page segmentation, but no OSD, or OCR.
    3    Fully automatic page segmentation, but no OSD. (Default)
    4    Assume a single column of text of variable sizes.
    5    Assume a single uniform block of vertically aligned text.
    6    Assume a single uniform block of text.
    7    Treat the image as a single text line.
    8    Treat the image as a single word.
    9    Treat the image as a single word in a circle.
    10   Treat the image as a single character.
    11   Sparse text. Find as much text as possible in no particular order.
    12   Sparse text with OSD.
    13   Raw line. Treat the image as a single text line,
    """
    if os.path.isdir(image_directory):
        onlyfiles = [os.path.join(image_directory, file) for file in os.listdir(image_directory)]
    else:
        onlyfiles = [image_directory]  # Handle single image

    # Dictionary to store results grouped by image name
    processed_results = {}

    # Iterate over each result and corresponding image path
    for result, image_path in zip(results, onlyfiles):
        image_name = os.path.basename(image_path)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image from {image_path}.")
            continue  # Skip files that aren't valid images

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes = result.boxes.data.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        image_results = []  # List to hold results for this image

        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box[:4])
            cropped_image = image_rgb[y1:y2, x1:x2]
            class_id = int(classes[idx])
            class_name = self.class_names[class_id]
            confidence = box[4]

            pil_cropped_image = Image.fromarray(cropped_image)

            if class_name == 'grade table':
                config = "--psm 6"  # Example config for grade tables
            else:
                config = "--psm 7"  # Default config for other classes

            text_extracted = pytesseract.image_to_string(pil_cropped_image, config=config)

            image_results.append({
                "class_name": class_name,
                "confidence": confidence,
                "text": text_extracted.strip(),
                "bounding_box": [x1, y1, x2, y2]
            })

        # Append the results for this image to the dictionary
        processed_results[image_name] = image_results

    return processed_results
    
  def format_strings(self, processed_results):
    """
    Processes OCR results to format text strings for LLM processing.

    This function combines text from 'grade table' and 'single row table' into a single string, and 
    all 'grade headers' data into another string. These formatted strings are intended to be passed 
    to an LLM for further processing.

    Parameters:
        processed_results (dict): A dictionary with image names as keys and lists of dictionaries 
                                  for each bounding box's OCR results.

    Returns:
        dict: A dictionary where each key is an image name and the value is another dictionary with
              two keys 'table_data' and 'header_data' containing formatted strings for each category.
    """

    formatted_data = {}

    for image_name, results in processed_results.items():
        table_texts = []
        header_texts = []

        for result in results:
            class_name = result['class_name']
            text = result['text']
            
            if class_name == 'grade table' or class_name == 'single row table':
                table_texts.append(text)
            elif class_name == 'grade headers':
                header_texts.append(text)

        # Combine all relevant texts into single strings, separated by newline or another separator
        table_data = '\n'.join(table_texts)  # Space or '\n' can be used depending on how the LLM processes data
        header_data = '\n'.join(header_texts)

        # Store formatted text in dictionary
        formatted_data[image_name] = {
            'table_data': table_data,
            'header_data': header_data
        }

    return formatted_data
  
if __name__ == '__main__':
  # image_directory = '/Users/declanbracken/Development/UofT_Projects/Meng_Project/Transcripts/Real Transcripts/'
  image_directory = '/Users/declanbracken/Development/UofT_Projects/Meng_Project/Transcripts/Web_Scraped_Transcripts/'
  # image_name = '3.png'
  image_name = '2015-queens-university-transcript-1-2048.webp'
  image_path = image_directory + image_name
  # model_path = '/Users/declanbracken/Development/UofT_Projects/Meng_Project/code_base/yolo_training/yolo_v8_models/finetune_v4 (3_classes)/best (1).pt'
  model_path = '/Users/declanbracken/Development/UofT_Projects/Meng_Project/code_base/yolo_training/yolo_v8_models/finetune_v5/best.pt'
  pipeline = vision_pipeline(model_path)
  results = pipeline.predict(image_path, plot = True, iou = 0.3, conf = 0.5, agnostic_nms = True)
  ocr_processor = OCRProcessor()
  # ocr_processor.init_easyocr()
  processed_results = ocr_processor.process_images_with_ocr(results, image_path, use_tesseract=True)
  formatted_strings = ocr_processor.format_strings(processed_results)

  first_image_name = next(iter(formatted_strings))
  print(formatted_strings[first_image_name]['header_data'])
  print(formatted_strings[first_image_name]['table_data'])
  