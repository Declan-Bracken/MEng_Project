import matplotlib.pyplot as plt
import sys
import numpy as np
import seaborn as sns
from scipy.spatial.distance import squareform
from sklearn.metrics import pairwise_distances
from hdbscan import HDBSCAN
# Ensure the parent directory is added to the system path
parent_dir = r'./'  # This represents the root of the repository
sys.path.append(parent_dir)
from Pipelines.Py_files.vision_pipeline import VisionPipeline
from TableReconstruction.column_clustering.column_clustererV2 import ColumnClusterer
from TableReconstruction.row_clustering.run_row_clusterer import RunRowClustering

def custom_distance(u, v):
        """
        Calculates a custom distance metric for clustering columns.
        """
        return min(abs(v[0] - u[1]), abs(v[0] - u[0]), abs(v[1] - u[1]), abs(v[1] - u[0]))

def plot_custom_distance_heatmap(positions, custom_distance = custom_distance):
    """
    Plots a heatmap of the custom distance matrix for clustering columns.
    """
    # Calculate the custom distance matrix
    distance_matrix = pairwise_distances(positions, metric=custom_distance)

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(distance_matrix, cmap='viridis')
    # plt.title('Custom Distance Matrix Heatmap')
    plt.xlabel('Text Elements')
    plt.ylabel('Text Elements')
    plt.savefig("TableReconstruction/column_clustering/Distance_Matrix.png",dpi = 200)
    plt.show()
    

def plot_dendrogram(clusterer, positions):
    """
    Plots a dendrogram for HDBSCAN clustering.
    """
    clusterer.fit(positions)
    plt.figure(figsize=(10, 8))
    
    clusterer.minimum_spanning_tree_.plot(edge_cmap='viridis', 
                                        edge_alpha=0.6, 
                                        node_size=50, 
                                        edge_linewidth=2)
    # plt.title('HDBSCAN Minimum Spanning Tree')
    plt.xlabel('Cluster Size')
    plt.ylabel('Distance')
    plt.savefig("TableReconstruction/column_clustering/Dendrogram.png",dpi = 200)
    plt.show()

if __name__ == "__main__":
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
    image_path = '/Users/declanbracken/Development/UofT_Projects/Meng_Project/Transcripts/Real Transcripts/2.JPG'
  
    pipeline = VisionPipeline(model_path)
    results = pipeline.predict(image_path, plot = True, iou = 0.3, conf = 0.5, agnostic_nms = True)
    # Row Clustering
    run_row_clustering = RunRowClustering(image_path,results)
    grouped_data = run_row_clustering.grouped_data
    # Run Column Clustering
    run_column_clustering = ColumnClusterer(grouped_data)
    min_samples_list = [4] * len(grouped_data)
    final_dfs = run_column_clustering.process_tables_to_dataframe(grouped_data, min_samples_list, cluster_selection_epsilon = 0.01, alpha = 1)
    for i in range(len(final_dfs)):
        print(f"Dataframe {i}:")
        print(final_dfs[i])
        print("\n")
    
    # PLOTTING
    # for idx, table_data in enumerate(grouped_data):
    #     if idx ==0:
    #         continue
    #     min_samples = min_samples_list[idx]  # Get the min_samples for the current table

    #     positions = table_data['positions']
    #     texts = table_data['texts']
    #     line_numbers = table_data['line_numbers']
    #     print(texts)

    #     num_lines = len(set(line_numbers))

    #     # Check for more than one word per row:
    #     has_columns = len(line_numbers) != num_lines

    #     if has_columns and num_lines > 1:
    #         # Optimize cluster_selection_epsilon
    #         optimal_cluster_selection_epsilon = run_column_clustering.optimize_cluster_selection_epsilon(
    #             positions, min_cluster_size=num_lines, min_samples=min_samples, alpha=1.0)
    #         labels = run_column_clustering.perform_hdbscan_clustering(positions, min_cluster_size=num_lines,
    #                                                     min_samples=min_samples,
    #                                                     cluster_selection_epsilon=optimal_cluster_selection_epsilon,
    #                                                     alpha=1.0)
    #         labels, centroids, label_mapping = run_column_clustering.reorder_columns(labels, positions)
    #         num_cols = len(set(labels))

        # Plot custom distance heatmap
        # plot_custom_distance_heatmap(positions)
    
        # # Plot HDBSCAN dendrogram
        # clusterer = run_column_clustering.clusterer
        # # clusterer.fit(positions)
        # plot_dendrogram(clusterer, positions)




