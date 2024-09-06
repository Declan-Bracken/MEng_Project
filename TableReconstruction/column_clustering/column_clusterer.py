import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import pairwise_distances
from sklearn.cluster import DBSCAN
from sklearn.metrics import davies_bouldin_score
from hdbscan import HDBSCAN
from scipy.spatial.distance import cdist

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
    
    def perform_hdbscan_clustering(self, positions, min_cluster_size = 5, min_samples = 10, cluster_selection_epsilon = 0, alpha = 1):
        clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples = min_samples, metric='euclidean', cluster_selection_epsilon = cluster_selection_epsilon, gen_min_span_tree = True) #, cluster_selection_method='eom'
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
            return new_labels, centroids, label_mapping
        return labels, {}, {}
    
    def classify_new_headers(self, new_positions, centroids, label_mapping):
        distances = cdist(new_positions, np.array(list(centroids.values())).reshape(-1, 1), metric='euclidean')
        new_labels = np.argmin(distances, axis=1)
        new_mapped_labels = [label_mapping[label] for label in new_labels]
        return new_mapped_labels

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
    
    @st.cache_data(show_spinner=False)
    def process_tables_to_dataframe(_self, tables_data, min_samples_list, cluster_selection_epsilon = 0, alpha = 1):
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
                labels = _self.perform_hdbscan_clustering(positions, min_cluster_size=num_lines, min_samples=min_samples, cluster_selection_epsilon = cluster_selection_epsilon, alpha = alpha)
                labels, centroids, label_mapping = _self.reorder_columns(labels, positions)
                num_cols = len(set(labels))
            elif num_lines > 1:  # More than one line, but one column
                labels = [0 for _ in range(num_lines)]
                num_cols = len(set(labels))
            else:  # More than one column, but one line
                optimal_eps = 27.5
                labels = _self.perform_clustering(positions, optimal_eps)
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
    
    def custom_distance(self, u, v):
        return min(abs(v[0] - u[1]), abs(v[0] - u[0]), abs(v[1] - u[1]), abs(v[1] - u[0]))

    def perform_clustering(self, positions, eps=15):
        distance_matrix = pairwise_distances(positions, metric=self.custom_distance)
        dbscan = DBSCAN(eps=eps, min_samples=1, metric='precomputed')
        labels = dbscan.fit_predict(distance_matrix)
        return labels
    
    def perform_hdbscan_clustering(self, positions, min_cluster_size=5, min_samples=10, cluster_selection_epsilon=0, alpha=1):
        clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric='euclidean', cluster_selection_epsilon=cluster_selection_epsilon, alpha=alpha, gen_min_span_tree=False)
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
    
    def compute_relative_features(self, positions):
        relative_features = []
        num_positions = len(positions)

        for i, (x1, x2) in enumerate(positions):
            width = x2 - x1

            prev_x1, prev_x2 = positions[i - 1] if i > 0 else (x1, x1)
            next_x1, next_x2 = positions[i + 1] if i < num_positions - 1 else (x2, x2)

            prev_distance = x1 - prev_x2
            next_distance = next_x1 - x2

            relative_features.append([x1, x2, width, prev_distance, next_distance])

        return relative_features
    
    def process_tables_to_dataframe(self, tables_data, min_samples_list, cluster_selection_epsilon=0, alpha=1):
        all_tables_dfs = []

        for idx, table_data in enumerate(tables_data):
            min_samples = min_samples_list[idx]

            positions = table_data['positions']
            texts = table_data['texts']
            line_numbers = table_data['line_numbers']

            # Compute relative features
            relative_positions = self.compute_relative_features(positions)

            num_lines = len(set(line_numbers))
            has_columns = len(line_numbers) != num_lines

            if has_columns and num_lines > 1:
                labels = self.perform_hdbscan_clustering(relative_positions, min_cluster_size=num_lines, min_samples=min_samples, cluster_selection_epsilon=cluster_selection_epsilon, alpha=alpha)
                labels = self.reorder_columns(labels, relative_positions)
                num_cols = len(set(labels))
            elif num_lines > 1:
                labels = [0 for _ in range(num_lines)]
                num_cols = len(set(labels))
            else:
                optimal_eps = 27.5
                labels = self.perform_clustering(relative_positions, optimal_eps)
                num_cols = len(set(labels))

            df = pd.DataFrame(np.nan, index=range(num_lines), columns=range(num_cols))
            df = df.astype(object)

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

        

