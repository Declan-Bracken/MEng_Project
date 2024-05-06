import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.cluster import DBSCAN
from sklearn.metrics import davies_bouldin_score

class ColumnClusterer:
    def custom_distance(self, u, v):
        return min(abs(v[0] - u[1]), abs(v[0] - u[0]), abs(v[1] - u[1]), abs(v[1] - u[0]))

    def perform_clustering(self, positions, eps=15):
        distance_matrix = pairwise_distances(positions, metric=self.custom_distance)
        dbscan = DBSCAN(eps=eps, min_samples=1, metric='precomputed')
        labels = dbscan.fit_predict(distance_matrix)
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

        return best_eps
