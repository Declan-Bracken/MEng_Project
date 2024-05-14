import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO
import tempfile
import os

class vision_pipeline:
    def __init__(self, path_to_cnn):
        self.path_to_cnn = path_to_cnn
        self.class_names = {0: 'grade headers', 1: 'grade table', 2: 'single row table'}
        self.class_colors = {0: 'g', 1: 'r', 2: 'b'}
        self._init_yolo()

    def _init_yolo(self):
        print("Loading Vision Model...")
        try:
            self.object_detector = YOLO(self.path_to_cnn)
            print("Model loaded successfully.")
        except Exception as e:
            print("Failed to load model:", e)

    def visualize_boxes(self, image_path, boxes, classes, names):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        fig, ax = plt.subplots(1, figsize=(8, 8), dpi = 200)
        ax.imshow(image)

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box[:4]
            class_id = int(classes[i])
            label = names[class_id]
            confidence = box[4]
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor=self.class_colors[class_id], facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1, f'{label} {confidence:.2f}', color='white', fontsize=6, bbox=dict(facecolor=self.class_colors[class_id], alpha=0.3))

        plt.axis('off')
        # plt.show()

        # Save the figure to a temporary file
        temp_img_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
        plt.savefig(temp_img_path, bbox_inches='tight', pad_inches=0.0)
        plt.close(fig)
        
        return temp_img_path
    
    def predict(self, image_directory, plot=False, **kwargs):
        if os.path.isdir(image_directory):
            onlyfiles = [image_directory + '/' + file for file in os.listdir(image_directory)]
        else:
            onlyfiles = [image_directory]

        results = self.object_detector.predict(onlyfiles, **kwargs)
        if plot:
            for result, image_path in zip(results, onlyfiles):
                classes = result.boxes.cls.cpu().numpy()
                boxes = result.boxes.data.cpu().numpy()
                self.visualize_boxes(image_path, boxes, classes, self.class_names)
        return results

# Initialize VisionPipeline
pipeline = vision_pipeline('/Users/declanbracken/Development/UofT_Projects/Meng_Project/code_base/yolo_training/yolo_v8_models/finetune_v5/best.pt')

# Streamlit app
st.title("Information Extraction - Bounding Box Display")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file:
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name

    # Predict bounding boxes using the vision pipeline
    results = pipeline.predict(temp_file_path, plot=False, iou = 0.3, conf = 0.5, agnostic_nms = True)

    # Extract bounding boxes and class names
    result = results[0]
    print(result)
    boxes = result.boxes.data.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()

    # Visualize and display the image with bounding boxes
    img_with_boxes_path = pipeline.visualize_boxes(temp_file_path, boxes, classes, pipeline.class_names)
    
    # Load the image with bounding boxes into a PIL image
    img_with_boxes = Image.open(img_with_boxes_path)
    
    # Display the image with bounding boxes
    st.image(img_with_boxes, caption="Image with Bounding Boxes")#, use_column_width=True)
