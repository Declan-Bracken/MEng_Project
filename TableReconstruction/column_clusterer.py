import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.cluster import DBSCAN
from sklearn.metrics import davies_bouldin_score
from hdbscan import HDBSCAN
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import pdist, squareform


class ColumnClusterer:
    def __init__(self, tables):
        self.all_tables_data = tables
    
    def custom_distance(self, u, v):
        return min(abs(v[0] - u[1]), abs(v[0] - u[0]), abs(v[1] - u[1]), abs(v[1] - u[0]))

    def perform_clustering(self, positions, eps=15):
        distance_matrix = pairwise_distances(positions, metric=self.custom_distance)
        dbscan = DBSCAN(eps=eps, min_samples=1, metric='precomputed')
        labels = dbscan.fit_predict(distance_matrix)
        return labels
    
    def perform_hdbscan_clustering(self, positions, min_cluster_size = 5, min_samples = 10):
        clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples = min_samples, metric='euclidean') #, cluster_selection_method='eom'
        labels = clusterer.fit_predict(positions)
        return labels
    
    def reorder_columns(self, labels, positions):
        if len(set(labels)) > 1:
            # Calculate centroids of each cluster
            centroids = {i: np.mean([pos[0] for j, pos in enumerate(positions) if labels[j] == i]) for i in set(labels)}
            # Sort labels based on the centroids
            sorted_labels = sorted(centroids, key=centroids.get)
            label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_labels)}
            new_labels = [label_mapping[label] for label in labels]
            return new_labels
        return labels

    def binary_search_optimize_eps(self, positions, eps_min=5, eps_max=50, tolerance=1):
        best_eps = None
        best_dbi = float('inf')

        while eps_max - eps_min > tolerance:
            eps_mid = (eps_min + eps_max) / 2
            labels = self.perform_clustering(positions, eps=eps_mid)

            if len(set(labels)) > 1 and -1 not in set(labels):
                dbi = davies_bouldin_score(positions, labels)
                if dbi < best_dbi:
                    best_dbi = dbi
                    best_eps = eps_mid

            labels_left = self.perform_clustering(positions, eps=(eps_min + eps_mid) / 2)
            labels_right = self.perform_clustering(positions, eps=(eps_mid + eps_max) / 2)

            dbi_left = davies_bouldin_score(positions, labels_left) if len(set(labels_left)) > 1 and -1 not in set(labels_left) else float('inf')
            dbi_right = davies_bouldin_score(positions, labels_right) if len(set(labels_right)) > 1 and -1 not in set(labels_right) else float('inf')

            if dbi_left < dbi_right:
                eps_max = eps_mid
            else:
                eps_min = eps_mid

        if best_eps == None:
            return eps_mid
        
        return best_eps
    
    def process_tables_to_dataframe(self, tables_data, min_samples_list):
        all_tables_dfs = []

        for idx, table_data in enumerate(tables_data):
            min_samples = min_samples_list[idx]  # Get the min_samples for the current table

            positions = table_data['positions']
            texts = table_data['texts']
            line_numbers = table_data['line_numbers']

            num_lines = len(set(line_numbers))

            # Check for more than one word per row:
            has_columns = len(line_numbers) != num_lines

            if has_columns and num_lines > 1:
                labels = self.perform_hdbscan_clustering(positions, min_cluster_size=num_lines, min_samples=min_samples)
                labels = self.reorder_columns(labels, positions)
                num_cols = len(set(labels))
            elif num_lines > 1:  # More than one line, but one column
                labels = [0 for _ in range(num_lines)]
                num_cols = len(set(labels))
            else:  # More than one column, but one line
                optimal_eps = 27.5
                labels = self.perform_clustering(positions, optimal_eps)
                num_cols = len(set(labels))

            df = pd.DataFrame(np.nan, index=range(num_lines), columns=range(num_cols))

            for text, label, line in zip(texts, labels, line_numbers):
                if 0 <= label < num_cols and 0 <= line < num_lines:
                    if pd.notna(df.at[line, label]):
                        df.at[line, label] = f"{df.at[line, label]} {text}"
                    else:
                        df.at[line, label] = text

            df.dropna(how='all', inplace=True)
            df.reset_index(drop=True, inplace=True)
            all_tables_dfs.append(df)

        return all_tables_dfs


class ColumnClustererV2:
    def __init__(self, tables):
        self.all_tables_data = tables

    def create_binary_heatmap(self, positions, num_bins=1000):
        """ Create a binary heatmap for the given positions. """
        scaler = MinMaxScaler()
        positions_scaled = scaler.fit_transform(positions[:, :2])  # Normalize positions
        heatmap = np.zeros((len(positions), num_bins))

        for idx, (start, end) in enumerate(positions_scaled):
            start_idx = int(start * num_bins)
            end_idx = int(end * num_bins)
            heatmap[idx, start_idx:end_idx] = 1

        return heatmap

    def perform_clustering(self, positions, num_bins=1000, eps=0.3):
        """ Cluster rows based on binary heatmap using Jaccard distance. """
        # Create binary heatmap
        heatmap = self.create_binary_heatmap(positions, num_bins)
        
        # Calculate Jaccard distance matrix
        jaccard_distances = pdist(heatmap, metric='jaccard')
        jaccard_distance_matrix = squareform(jaccard_distances)

        # Cluster using DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=1, metric="precomputed")
        labels = dbscan.fit_predict(jaccard_distance_matrix)
        return labels
    
    def perform_hdbscan_clustering(self, positions, num_bins = 100, min_cluster_size = 5, min_samples = 1):
        # Create binary heatmap
        heatmap = self.create_binary_heatmap(positions, num_bins)
        
        # Calculate Jaccard distance matrix
        jaccard_distances = pdist(heatmap, metric='jaccard')
        jaccard_distance_matrix = squareform(jaccard_distances)

        clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric='precomputed') #, cluster_selection_method='eom'
        labels = clusterer.fit_predict(jaccard_distance_matrix)
        return labels

    def reorder_columns(self, labels, positions):
        if len(set(labels)) > 1:
            # Calculate centroids of each cluster
            centroids = {i: np.mean([pos[0] for j, pos in enumerate(positions) if labels[j] == i]) for i in set(labels)}
            # Sort labels based on the centroids
            sorted_labels = sorted(centroids, key=centroids.get)
            label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_labels)}
            new_labels = [label_mapping[label] for label in labels]
            return new_labels
        return labels



