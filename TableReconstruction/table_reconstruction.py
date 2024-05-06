from image_processor import ImageProcessor
from text_classifier import TextClassifier
from column_clusterer import ColumnClusterer
from dataframe_processor import DataFrameProcessor
from visualizer import Visualizer

class TableReconstructor:
    def __init__(self, image_path, results):
        self.boxes = results[0].boxes.data.cpu().numpy()
        self.classes = results[0].boxes.cls.cpu().numpy()

        self.image_processor = ImageProcessor(image_path, self.boxes)
        self.text_classifier = TextClassifier(self.image_processor.cropped_images, self.classes, self.boxes)
        self.clusterer = ColumnClusterer()
        self.df_processor = DataFrameProcessor()
        self.visualizer = Visualizer()

    def process_and_visualize(self):
        self.text_classifier.classify_text()
        tables_data = self.text_classifier.get_tables()
        all_tables_df = self.df_processor.process_tables_to_dataframe(tables_data, self.clusterer)

        print(all_tables_df)

if __name__ == "__main__":
    import sys
    import os

    # Add the parent directory of "vision_pipeline.py" to sys.path
    current_file_dir = os.path.dirname(__file__)  # The current directory where this script is located
    parent_dir = os.path.abspath(os.path.join(current_file_dir, "..", "Pipelines"))  # Adjust based on your directory structure
    sys.path.append(parent_dir)

    # Now, you can import the required class from vision_pipeline.py
    from vision_pipeline import vision_pipeline

    image_directory = '/Users/declanbracken/Development/UofT_Projects/Meng_Project/Transcripts/Web_Scraped_Transcripts/'
    image_name = '2015-queens-university-transcript-1-2048.webp'
    image_path = image_directory + image_name

    model_path = '/Users/declanbracken/Development/UofT_Projects/Meng_Project/code_base/yolo_training/yolo_v8_models/finetune_v5/best.pt'
    pipeline = vision_pipeline(model_path)
    results = pipeline.predict(image_path, plot = True, iou = 0.3, conf = 0.5, agnostic_nms = True)

    table_reconstructor = TableReconstructor(image_path, results)
    table_reconstructor.process_and_visualize()