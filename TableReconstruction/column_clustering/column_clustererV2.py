from scipy.spatial.distance import cdist
from sklearn.metrics import davies_bouldin_score
from hdbscan import HDBSCAN
from scipy.optimize import differential_evolution
from sklearn.metrics import pairwise_distances
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ColumnClusterer:
    def __init__(self, tables):
        self.all_tables_data = tables

    def custom_distance(self, u, v):
        """
        Calculates a custom distance metric for clustering columns.
        """
        return min(abs(v[0] - u[1]), abs(v[0] - u[0]), abs(v[1] - u[1]), abs(v[1] - u[0]))

    def perform_clustering(self, positions, eps=15):
        """
        Performs DBSCAN clustering on the positions using the custom distance metric.
        """
        distance_matrix = pairwise_distances(positions, metric=self.custom_distance)
        dbscan = DBSCAN(eps=eps, min_samples=1, metric='precomputed')
        labels = dbscan.fit_predict(distance_matrix)
        return labels

    def perform_hdbscan_clustering(self, positions, min_cluster_size=5, min_samples=4, cluster_selection_epsilon=1.0, alpha=1.0):
        """
        Performs HDBSCAN clustering on the positions.
        """
        self.clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                            metric='euclidean', cluster_selection_epsilon=float(cluster_selection_epsilon),
                            alpha=float(alpha), gen_min_span_tree=True)
        labels = self.clusterer.fit_predict(positions)
        return labels

    def reorder_columns(self, labels, positions):
        """
        Reorders the columns based on the centroids of the clusters.
        """
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
        """
        Classifies new headers based on their distance to existing cluster centroids.
        """
        distances = cdist(new_positions, np.array(list(centroids.values())).reshape(-1, 1), metric='euclidean')
        new_labels = np.argmin(distances, axis=1)
        new_mapped_labels = [label_mapping[label] for label in new_labels]
        return new_mapped_labels

    def optimize_cluster_selection_epsilon(self, positions, min_cluster_size=5, min_samples=10, alpha=1):
        """
        Optimizes the cluster_selection_epsilon parameter using differential evolution.
        """
        def objective_function(cluster_selection_epsilon):
            labels = self.perform_hdbscan_clustering(positions,
                                                     min_cluster_size=min_cluster_size,
                                                     min_samples=min_samples,
                                                     cluster_selection_epsilon=cluster_selection_epsilon[0],
                                                     alpha=alpha)
            # Ensure more than one cluster and no noise-only group
            if len(set(labels)) > 1 and -1 not in set(labels):
                return davies_bouldin_score(positions, labels)
            return float('inf')  # Penalize for a single cluster or noisy clustering

        # Set up bounds for the cluster_selection_epsilon parameter
        bounds = [(0.01, 200.0)]

        # Use differential evolution to minimize the objective function
        result = differential_evolution(objective_function, bounds, strategy='best1bin', maxiter=50, disp=True)

        # Extract the best cluster_selection_epsilon
        best_cluster_selection_epsilon = result.x[0]

        return best_cluster_selection_epsilon

    def process_tables_to_dataframe(self, tables_data, min_samples_list, cluster_selection_epsilon=0.01, alpha=1):
        """
        Processes each table to convert it into a DataFrame format after column clustering.
        """
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
                # Optimize cluster_selection_epsilon
                optimal_cluster_selection_epsilon = self.optimize_cluster_selection_epsilon(
                    positions, min_cluster_size=num_lines, min_samples=min_samples, alpha=alpha)
                labels = self.perform_hdbscan_clustering(positions, min_cluster_size=num_lines,
                                                         min_samples=min_samples,
                                                         cluster_selection_epsilon=optimal_cluster_selection_epsilon,
                                                         alpha=alpha)
                labels, centroids, label_mapping = self.reorder_columns(labels, positions)
                num_cols = len(set(labels))
            elif num_lines > 1:  # More than one line, but one column
                labels = [0 for _ in range(num_lines)]
                num_cols = len(set(labels))
            else:  # More than one column, but one line
                optimal_eps = 27.5
                labels = self.perform_clustering(positions, optimal_eps)
                num_cols = len(set(labels))

            df = pd.DataFrame("", index=range(num_lines), columns=range(num_cols), dtype=object)

            for text, label, line in zip(texts, labels, line_numbers):
                if 0 <= label < num_cols and 0 <= line < num_lines:
                    if pd.notna(df.at[line, label]):
                        df.at[line, label] = f"{str(df.at[line, label])} {str(text)}"
                    else:
                        df.at[line, label] = str(text)

            df.dropna(how='all', inplace=True)
            df.reset_index(drop=True, inplace=True)
            all_tables_dfs.append(df)

        return all_tables_dfs

