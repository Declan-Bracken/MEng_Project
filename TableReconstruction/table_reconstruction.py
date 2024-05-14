import sys
from image_processor import ImageProcessor
from text_classifier import TextClassifier
from row_clusterer import RowClassifier
from clustering_gui import TableReconstructorGUI
from column_clusterer import ColumnClusterer
from dataframe_processor import DataFrameProcessor
from PyQt5 import QtWidgets

# Manually specify the path to the parent directory
parent_dir = '/Users/declanbracken/Development/UofT_Projects/Meng_Project/code_base'  # Replace this with the path where 'Pipelines' is located
sys.path.append(parent_dir)
from Pipelines.Py_files.vision_pipeline import vision_pipeline


class TableReconstructor:
    def __init__(self, model_path):
        # Initialize the vision pipeline with the model path
        self.vision_pipeline = vision_pipeline(model_path)
    
    def run_pipeline(self, image_path, iou=0.3, conf=0.5, plot=True, linkage='average'):
        """
        Runs the entire table reconstruction pipeline.

        Args:
        - image_path: Path to the image to analyze.
        - iou: IOU threshold for the YOLO model predictions.
        - conf: Confidence threshold for the YOLO model predictions.
        - plot: Whether or not to plot the initial YOLO predictions.
        - similarity_threshold: Threshold for merging clusters in row clustering.
        - linkage: The linkage strategy for hierarchical clustering.
        """
        # Step 1: Predict using the YOLO model
        results = self.vision_pipeline.predict(image_path, plot=plot, iou=iou, conf=conf, agnostic_nms=True)
        boxes = results[0].boxes.data.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        
        # Step 2: Process the image
        image_processor = ImageProcessor(image_path, boxes)
        image = image_processor.image
        cropped_images = image_processor.cropped_images
        print(f"Number of cropped images: {len(cropped_images)}")

        # Step 3: Classify the text
        text_classifier = TextClassifier(cropped_images, classes, boxes)
        
        # Step 4: Cluster the rows
        row_classifier = RowClassifier(text_classifier.headers, text_classifier.single_row, text_classifier.tables)
        all_rows = row_classifier.collect_all_rows()
        print("Number of rows for clustering:", len(all_rows))
        
        # best_threshold = row_classifier.optimize_distance_threshold(row_features, distance_matrix, linkage=linkage)
        best_threshold = 0.37

        # GUI
        app = QtWidgets.QApplication(sys.argv)
        window = TableReconstructorGUI(row_classifier, all_rows, image, best_threshold = best_threshold)
        window.showFullScreen() # Show GUI in full screen
        app.exec_()

        # Get Regrouped Data
        regrouped_data = window.regrouped_data
        
        # Step 5: Cluster columns
        column_clusterer = ColumnClusterer(regrouped_data)
        all_tables_data = column_clusterer.all_tables_data
        
        # Step 6: Process tables into dataframes
        df_processor = DataFrameProcessor()
        final_dfs = df_processor.process_tables_to_dataframe(all_tables_data, column_clusterer)

        # Display results
        for idx, df in enumerate(final_dfs):
            print(f"\nTable {idx + 1} Preview:\n", df.head(50))

        return final_dfs

if __name__ == "__main__":

    image_path = '/Users/declanbracken/Development/UofT_Projects/Meng_Project/Transcripts/Web_Scraped_Transcripts/2015-queens-university-transcript-1-2048.webp'
    image_path = '/Users/declanbracken/Development/UofT_Projects/Meng_Project/Transcripts/Web_Scraped_Transcripts/nyu-official-transcript-2-2048.webp'
    image_path = '/Users/declanbracken/Development/UofT_Projects/Meng_Project/Transcripts/Web_Scraped_Transcripts/completedtranscript8889298-2-2048.webp'
    # image_path = '/Users/declanbracken/Development/UofT_Projects/Meng_Project/Transcripts/Web_Scraped_Transcripts/bachelors-degree-transcript-2-2048-1.webp'
    # image_path = '/Users/declanbracken/Development/UofT_Projects/Meng_Project/Transcripts/Web_Scraped_Transcripts/ba-transcript-3-2048.webp'
    model_path = '/Users/declanbracken/Development/UofT_Projects/Meng_Project/code_base/yolo_training/yolo_v8_models/finetune_v5/best.pt'

    table_reconstructor = TableReconstructor(model_path)
    final_dfs = table_reconstructor.run_pipeline(image_path)