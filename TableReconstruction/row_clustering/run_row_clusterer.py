import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import os
# Ensure the parent directory is added to the system path
parent_dir = r'./'  # This represents the root of the repository
sys.path.append(parent_dir)

from Pipelines.Py_files.vision_pipeline import VisionPipeline
from TableReconstruction.image_processor import ImageProcessor
from TableReconstruction.text_classifier import TextClassifier
from TableReconstruction.row_clustering.row_clusterer_v2 import RowClassifier

# Given That the Yolo Model Has Been Run, Run Row Clustering.

class RunRowClustering:
    def __init__(_self, image_path, results):
        _self.results = results
        _self.image_path = image_path
        _self.row_classifier = _self.preprocess_results()
        _self.distance_matrix = _self.get_distance_matrix()
        _self.best_threshold = _self.get_best_clustering_threshold()
        _self.grouped_data = _self.cluster()

    # Function to crop images
    def process_image(_self, image_path, boxes):
        image_processor = ImageProcessor(image_path, boxes)
        image = image_processor.image
        cropped_images = image_processor.cropped_images
        return image, cropped_images
    
    def preprocess_results(_self):
        # Get Boxes and Classes from YOLO output
        result = _self.results[0]
        boxes = result.boxes.data.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        # img_with_boxes = display_image_with_boxes(image_path, boxes, classes)
        # Get Cropped Images
        _self.image, cropped_images = _self.process_image(_self.image_path, boxes)
        # Reformat
        text_classifier = TextClassifier(cropped_images, classes, boxes)
        # Grab Seperate Items
        headers = text_classifier.headers
        single_row = text_classifier.single_row
        tables = text_classifier.tables

        return RowClassifier(headers, single_row, tables)

    def get_distance_matrix(_self):
        _self.all_rows = _self.row_classifier.collect_all_rows()
        _self.row_features = _self.row_classifier.create_binary_heatmap_features(_self.all_rows)
        distance_matrix = _self.row_classifier.calculate_jaccard_distance_matrix(_self.row_features)
        return distance_matrix

    def get_best_clustering_threshold(_self):
        best_threshold = _self.row_classifier.optimize_by_histogram(_self.all_rows)
        return best_threshold
    
    def cluster(_self, linkage = 'average'):
        labels = _self.row_classifier.cluster_lines_with_agglomerative_jaccard(_self.distance_matrix, distance_threshold=_self.best_threshold, linkage=linkage)
        return _self.row_classifier.group_rows_by_labels(_self.all_rows, labels)

def plot_separated_rows(image, all_rows):
    """
    Plots separated rows after pre-processing on the original image.
    
    Args:
    - image: The original image.
    - all_rows: List of separated rows with their bounding boxes.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    
    for row in all_rows:
        text_box = row['box']
        rect = patches.Rectangle(
            (text_box[0], text_box[1]),  # Starting point
            text_box[2] - text_box[0],   # Width
            text_box[3] - text_box[1],   # Height
            linewidth=1,
            edgecolor='r',
            facecolor='none'
        )
        ax.add_patch(rect)
    
    plt.title("Separated Rows with Bounding Boxes")
    plt.axis('off')
    plt.show()


def plot_binary_heatmap(row_features):
    """
    Plots binary heatmap features for each row.
    
    Args:
    - row_features: 2D numpy array of binary vectors for each row.
    """
    plt.figure(figsize=(9, 8))
    sns.heatmap(row_features, cmap='YlGnBu', cbar=True)
    # plt.title("Binary Heatmap Features for Rows")
    plt.xlim(0,650)
    plt.xlabel("Feature Bins")
    plt.ylabel("Rows")
    plt.savefig("/Users/declanbracken/Development/UofT_Projects/Meng_Project/code_base/TableReconstruction/row_clustering/feature_map.png",dpi = 200)
    plt.show()
    

def plot_jaccard_distance_matrix(distance_matrix):
    """
    Plots the Jaccard distance matrix as a heatmap.
    
    Args:
    - distance_matrix: A square Jaccard distance matrix (2D numpy array).
    """
    plt.figure(figsize=(9, 8))
    sns.heatmap(distance_matrix, cmap='coolwarm', cbar=True)
    # plt.title("Jaccard Distance Matrix Heatmap")
    plt.xlabel("Rows")
    plt.ylabel("Rows")
    plt.savefig("/Users/declanbracken/Development/UofT_Projects/Meng_Project/code_base/TableReconstruction/row_clustering/distance_matrix.png",dpi = 200)
    plt.show()

def plot_clustered_rows(image, grouped_data):
    """
    Plots the bounding boxes of clustered rows with different colors for each cluster.
    
    Args:
    - image: The original image.
    - grouped_data: List of grouped rows with cluster labels.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)
    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    
    for i, group in enumerate(grouped_data):
        for (x1, x2), (y1, y2) in zip(group['global_positions'], group['y_global_positions']):
            color = colors[i % len(colors)]
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2,
                edgecolor=color,
                facecolor='none'
            )
            ax.add_patch(rect)
    
    # plt.title("Clustered Rows with Bounding Boxes")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Set global font size
    plt.rcParams.update({
        'font.size': 14,           # Default font size
        'axes.titlesize': 16,      # Title font size
        'axes.labelsize': 14,      # X and Y label font size
        'xtick.labelsize': 12,     # X tick label font size
        'ytick.labelsize': 12,     # Y tick label font size
        'legend.fontsize': 12,     # Legend font size
        'figure.titlesize': 18     # Figure title font size
    })

    model_path = 'yolo_training/yolo_v8_models/finetune_v5/best.pt'
    image_path = '/Users/declanbracken/Development/UofT_Projects/Meng_Project/Transcripts/Web_Scraped_Transcripts/2015-queens-university-transcript-1-2048.webp'
  
    pipeline = VisionPipeline(model_path)
    results = pipeline.predict(image_path, plot = True, iou = 0.3, conf = 0.5, agnostic_nms = True)
    run_row_clustering = RunRowClustering(image_path,results)

    # plot_separated_rows(run_row_clustering.image, run_row_clustering.row_classifier.collect_all_rows())
    plot_binary_heatmap(run_row_clustering.row_features)
    plot_jaccard_distance_matrix(run_row_clustering.distance_matrix)
