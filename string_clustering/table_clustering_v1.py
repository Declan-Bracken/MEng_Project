import numpy as np
import pandas as pd
import cv2
import seaborn as sns
from PIL import Image, ImageDraw
import pytesseract
from pytesseract import Output
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score

class TextClusterer:
    def __init__(self, image_path, results):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            print(f"Failed to load image from {image_path}.")
            return
        
        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.boxes = results[0].boxes.data.cpu().numpy()
        self.classes = results[0].boxes.cls.cpu().numpy()
        self.crop_images()
    
    # Crop images by YOLO BBoxes
    def crop_images(self):
        # Lists for cropped images and ocr output
        self.cropped_images = []
        # Loop through vision model's bboxes and crop + OCR
        for idx, box in enumerate(self.boxes):
            x1, y1, x2, y2 = map(int, box[:4])
            cropped_image = Image.fromarray(self.image_rgb[y1:y2, x1:x2])
            self.cropped_images.append(cropped_image)

    # Apply OCR and organize by class
    def classify_text(self):
        self.headers = []
        self.tables = []
        self.single_row = []

        # Loop through cropped images
        for idx, box in enumerate(self.boxes):
            class_type = self.classes[idx]
            text = pytesseract.image_to_data(self.cropped_images[idx], output_type=Output.DICT, config='--psm 6')
            if class_type == 0:
                self.headers.append({'box': box, 'text': text})
            elif class_type == 1:
                self.tables.append({'box': box, 'text': text})
            else:
                self.single_row.append({'box': box, 'text': text})
    
    # FUNCTION HERE TO FILTER BY ROW AND REGROUP


    def extract_horizontal_positions(self, tables):
        all_tables_data = []
        
        # Loop through tables
        for table in tables:
            positions = []
            texts = []
            y_positions = []
            line_numbers = []
            
            for i, text in enumerate(table['text']['text']):
                if text.strip():  # Ensure the text is not empty
                    # Get Bounding Box Positions
                    x1 = table['text']['left'][i]
                    width = table['text']['width'][i]
                    y1 = table['text']['top'][i]
                    x2 = x1 + width
                    height = table['text']['height'][i]
                    line_num = table['text']['line_num'][i]

                    positions.append([x1, x2])
                    texts.append(text)
                    y_positions.append((y1, height))
                    line_numbers.append(line_num)

            # Store data for the current table in a dictionary
            table_data = {
                'positions': np.array(positions),
                'texts': texts,
                'y_positions': y_positions,
                'line_numbers': line_numbers
            }
            all_tables_data.append(table_data)

        return all_tables_data

    # Custom Distance Metric for word grouping based on minimum distances
    def custom_distance(self, u, v):
        return min(abs(v[0] - u[1]), abs(v[0] - u[0]), abs(v[1] - u[1]), abs(v[1] - u[0]))

    def calculate_clustering_metrics(self, positions, labels):

        # Assuming 'X' is the dataset and 'labels' are the labels from DBSCAN
        sil_score = silhouette_score(positions, labels)

        db_index = davies_bouldin_score(positions, labels)

        ch_index = calinski_harabasz_score(positions, labels)

        print("Silhouette Score: ", sil_score)
        print("Davies-Bouldin Index: ", db_index)
        print("Calinski-Harabasz Index: ", ch_index)

    # Cluster Using DBSCAN
    def perform_clustering(self, positions, eps = 15):
        self.distance_matrix = pairwise_distances(positions, metric=self.custom_distance)
        dbscan = DBSCAN(eps=eps, min_samples=1, metric="precomputed")
        labels = dbscan.fit_predict(self.distance_matrix)
        return labels

    def binary_search_optimize_eps(self, positions, eps_min=5, eps_max=50, tolerance=1):
        """
        Optimize the `eps` parameter using binary search to minimize the Davies-Bouldin Index.

        Args:
        - positions (np.ndarray): Array of positions to cluster.
        - eps_min (float): Minimum value of the search interval.
        - eps_max (float): Maximum value of the search interval.
        - tolerance (float): Stopping criterion for the search interval.

        Returns:
        - optimal_eps (float): The `eps` value that yields the best Davies-Bouldin Index.
        """
        best_eps = None
        best_dbi = float('inf')

        while eps_max - eps_min > tolerance:
            eps_mid = (eps_min + eps_max) / 2
            labels = self.perform_clustering(positions, eps=eps_mid)

            # Ensure the clustering is valid (more than one cluster and no noise-only groups)
            if len(set(labels)) > 1 and -1 not in set(labels):
                dbi = davies_bouldin_score(positions, labels)
                if dbi < best_dbi:
                    best_dbi = dbi
                    best_eps = eps_mid

            # Adjust search range
            labels_left = self.perform_clustering(positions, eps=(eps_min + eps_mid) / 2)
            labels_right = self.perform_clustering(positions, eps=(eps_mid + eps_max) / 2)

            dbi_left = davies_bouldin_score(positions, labels_left) if len(set(labels_left)) > 1 and -1 not in set(labels_left) else float('inf')
            dbi_right = davies_bouldin_score(positions, labels_right) if len(set(labels_right)) > 1 and -1 not in set(labels_right) else float('inf')

            if dbi_left < dbi_right:
                eps_max = eps_mid
            else:
                eps_min = eps_mid

        print(f"Optimal `eps`: {best_eps}, with Davies-Bouldin Index: {best_dbi}")
        return best_eps

    # Assign text to columns based on labels
    def assign_to_columns(self, texts, labels):
        columns = {}
        for label, text in zip(labels, texts):
            if label not in columns:
                columns[label] = []
            columns[label].append(text)
        return columns

    def process_tables_to_dataframe(self, tables_data):
        """
        Process all tables data to extract and cluster texts, then format into a DataFrame.
        
        Args:
        - tables_data: List of dictionaries containing 'positions', 'texts', and 'y_positions' for each table.
        - clusterer: An instance of the TextClusterer class.
        
        Returns:
        - DataFrame where each column represents clustered texts from all tables.
        """
        # List to hold all tables dataframes
        all_tables_dfs = []
        
        for table_data in tables_data:
            positions = table_data['positions']
            texts = table_data['texts']
            line_numbers = table_data['line_numbers']
            
            # Optimize Clusters
            optimal_eps = self.binary_search_optimize_eps(positions)
            # Perform clustering with optimal eps
            labels = self.perform_clustering(positions, optimal_eps)
            # Get Number of collumns and rows
            num_cols = max(labels) + 1
            num_lines = max(line_numbers) + 1

            # Initialize an empty DataFrame with given dimensions
            df = pd.DataFrame(np.nan, index=range(num_lines), columns=range(num_cols))
            
            # Map texts to the correct positions in the DataFrame
            for text, label, line in zip(texts, labels, line_numbers):
                if 0 <= label < num_cols and 0 <= line < num_lines:
                    if pd.notna(df.at[line, label]):
                        # Concatenate texts if a cell is already occupied
                        df.at[line, label] = f"{df.at[line, label]} {text}"
                    else:
                        df.at[line, label] = text
            
            # Remove rows that are entirely NaN
            df.dropna(how='all', inplace=True)
            df.reset_index(drop=True, inplace=True)
            
            all_tables_dfs.append(df)

        all_tables_combined_df = pd.concat(all_tables_dfs, axis=0, ignore_index=True)
            
        return all_tables_combined_df

    # For plotting
    def plot_results(self, img, positions, texts, labels, y_positions):
        draw = ImageDraw.Draw(img)
        colors = list(mcolors.TABLEAU_COLORS.values())
        
        for (pos, text, label, y_pos) in zip(positions, texts, labels, y_positions):
            x1, x2 = pos
            y1, height = y_pos
            color = colors[label % len(colors)]
            draw.rectangle([x1, y1, x2, y1 + height], outline=color, width=2)
            draw.text((x1, y1), text, fill=color)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    
    def plot_distance_matrix(self):
        # Plotting the distance matrix as a heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.distance_matrix, annot=False, cmap='coolwarm', fmt=".2f", square=True)
        plt.title("Pairwise Distance Matrix (Custom Metric)")
        plt.xlabel("Word Index")
        plt.ylabel("Word Index")
        plt.show()

if __name__ == "__main__":
    import sys
    import os

    # Add the parent directory to sys.path to access modules in sibling directories
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

    # Vision Pipeline to get bboxes and results
    from Pipelines.vision_pipeline import vision_pipeline
    image_directory = '/Users/declanbracken/Development/UofT_Projects/Meng_Project/Transcripts/Web_Scraped_Transcripts/'
    image_name = '2015-queens-university-transcript-1-2048.webp'
    image_path = image_directory + image_name

    model_path = '/Users/declanbracken/Development/UofT_Projects/Meng_Project/code_base/yolo_training/yolo_v8_models/finetune_v5/best.pt'
    pipeline = vision_pipeline(model_path)
    results = pipeline.predict(image_path, plot = True, iou = 0.3, conf = 0.5, agnostic_nms = True)

    # Assuming 'results' is the output from some object detection model
    text_clusterer = TextClusterer(image_path, results)
    text_clusterer.classify_text()
    tables_data = text_clusterer.extract_horizontal_positions(text_clusterer.tables)
    # Create Dataframe
    all_tables_df = text_clusterer.process_tables_to_dataframe(tables_data)
    print(all_tables_df)
    # Access and unpack the first table
    # first_table_data = tables_data[0]  # Access the first table's data

    # # Unpack the dictionary into variables
    # positions = first_table_data['positions']
    # texts = first_table_data['texts']
    # y_positions = first_table_data['y_positions']

    # labels = text_clusterer.perform_clustering(positions)
    # columns = text_clusterer.assign_to_columns(texts, labels)
    # text_clusterer.plot_results(positions, texts, labels, y_positions)
    # text_clusterer.plot_distance_matrix()