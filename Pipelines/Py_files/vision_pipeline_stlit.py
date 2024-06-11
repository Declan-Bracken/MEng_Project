import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO
import os
import tempfile

class VisionPipeline:
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
            ax.text(x1, y1, f'{label} {confidence:.2f}', color='white', fontsize=5, bbox=dict(facecolor=self.class_colors[class_id], alpha=0.5))

        # ax.legend(labels = self.class_names.values)
        plt.axis('off')
        
        # Save the figure to a temporary file
        temp_img_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
        plt.savefig(temp_img_path, bbox_inches='tight', pad_inches=0.0)
        plt.close(fig)
        
        return temp_img_path

    def predict(self, image_directory, **kwargs):
        if os.path.isdir(image_directory):
            onlyfiles = [image_directory + '/' + file for file in os.listdir(image_directory)]
        else:
            onlyfiles = [image_directory]

        results = self.object_detector.predict(onlyfiles, **kwargs)
        return results