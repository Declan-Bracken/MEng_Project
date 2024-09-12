from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import calinski_harabasz_score
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import differential_evolution
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D


class RowClassifier:
    def __init__(self, headers, single_row, tables):
        self.headers = headers
        self.single_row = single_row
        self.tables = tables
        self.all_rows = []

    def _separate_rows(self, table_data):
        """
        Separates rows from a single bounding box into individual dictionaries.
        Returns a list of rows as individual dictionaries.
        """
        rows = {}
        text_data = table_data['text']
        current_line = 0

        for i, word in enumerate(text_data['text']):
            if word == '':  # Detect new line/paragraph start based on empty strings
                current_line += 1
                continue

            if current_line not in rows:
                rows[current_line] = {
                    'box': table_data['box'],
                    'text': {
                        'left': [],
                        'top': [],
                        'width': [],
                        'height': [],
                        'text': []
                    }
                }

            rows[current_line]['text']['left'].append(text_data['left'][i])
            rows[current_line]['text']['top'].append(text_data['top'][i])
            rows[current_line]['text']['width'].append(text_data['width'][i])
            rows[current_line]['text']['height'].append(text_data['height'][i])
            rows[current_line]['text']['text'].append(word)

        return list(rows.values())

    def collect_all_rows(self):
        """ Excludes headers for now. """
        all_rows = []
        #all_headers = []
        for row in self.single_row:
            all_rows.extend(self._separate_rows(row))
        for table in self.tables:
            all_rows.extend(self._separate_rows(table))
        # for header in self.headers:
        #     all_headers.extend(self._separate_rows(header))

        # Clean Rows
        all_rows = self._clean_rows(all_rows)
        #all_headers = self._clean_rows(all_headers)

        return all_rows#, all_headers

    def _clean_rows(self, row_list):
        """
        Cleans the input rows by removing empty text entries and discarding empty rows.
        
        Args:
        - row_list: List of dictionaries representing each row.

        Returns:
        - A cleaned list of rows without empty text entries or completely empty rows.
        """
        cleaned_rows = []

        for row in row_list:
            # Initialize cleaned text information
            cleaned_text = {
                'left': [],
                'top': [],
                'width': [],
                'height': [],
                'text': []
            }

            # Filter out empty text entries and adjust global positions
            for i, word in enumerate(row['text']['text']):
                if word.strip():  # Only keep non-empty words
                    cleaned_text['left'].append(row['text']['left'][i])
                    cleaned_text['top'].append(row['text']['top'][i])
                    cleaned_text['width'].append(row['text']['width'][i])
                    cleaned_text['height'].append(row['text']['height'][i])
                    cleaned_text['text'].append(word)

            # If the cleaned text data is not empty, add this row to the cleaned rows
            if cleaned_text['text']:
                cleaned_rows.append({
                    'box': row['box'],
                    'text': cleaned_text
                })

        return cleaned_rows
    
    def calculate_global_width(self, row_list):
        """
        Calculate the global width by finding the widest bounding box across all rows.

        Args:
        - row_list: List of dictionaries representing each row.

        Returns:
        - global_width: The maximum width across all rows.
        """
        global_min = float('inf')
        global_max = float('-inf')

        for item in row_list:
            box = item['box']  # Bounding box of the cropped row
            global_min = min(global_min, box[0])  # Leftmost point
            global_max = max(global_max, box[2])  # Rightmost point

        return global_max - global_min

    def create_binary_heatmap_features(self, row_list, num_bins=1000):
        """
        Generates binary heatmap features for each row, indicating word presence/absence.

        Args:
        - row_list: List of dictionaries representing each row.
        - num_bins: The number of bins (fine grid) for each row.

        Returns:
        - An array with a fixed-length binary vector for each row.
        """
        # Calculate the global width
        self.global_width = self.calculate_global_width(row_list)

        features = []

        for item in row_list:
            box = item['box']  # Bounding box of the cropped row
            text = item['text']
            left = text['left']
            width = text['width']

            # Calculate the global positions of the words
            global_lefts = [l for l in left]
            global_rights = [l + w for l, w in zip(left, width)]

            # Initialize a binary vector with zeros
            binary_vector = np.zeros(num_bins)

            # Mark bins where a word is present based on global width
            for l, r in zip(global_lefts, global_rights):
                left_idx = int((l / self.global_width) * num_bins)
                right_idx = int((r / self.global_width) * num_bins)
                binary_vector[left_idx:right_idx] = 1

            features.append(binary_vector)

        return np.array(features)
    
    def calculate_jaccard_distance_matrix(self, binary_features):
        """
        Calculates the Jaccard distance matrix for a set of binary feature vectors.
        
        Args:
        - binary_features: 2D numpy array (rows x columns) representing binary vectors.

        Returns:
        - distance_matrix: A square Jaccard distance matrix (2D numpy array).
        """
        # pdist computes pairwise distances; 'jaccard' calculates Jaccard distances
        jaccard_distances = pdist(binary_features, metric='jaccard')
        # Convert to a square distance matrix
        distance_matrix = squareform(jaccard_distances)
        return distance_matrix
    
    def optimize_distance_threshold(self, binary_features, distance_matrix, linkage='complete'):
        """
        Optimizes the distance threshold to minimize the Calinski-Harabasz Index using differential evolution.
        
        Args:
        - binary_features: 2D array-like binary feature vectors representing rows.
        - linkage: Linkage criterion ('ward', 'complete', 'average', 'single').

        Returns:
        - best_threshold: Optimal distance threshold.
        """

        def objective_function(threshold):
            labels = self.cluster_lines_with_agglomerative_jaccard(
                distance_matrix=distance_matrix,
                distance_threshold=threshold[0],
                linkage=linkage
            )
            # Ensure more than one cluster and no noise-only group
            if len(set(labels)) > 1:
                return -davies_bouldin_score(binary_features, labels)
            return float('inf')  # Penalize for a single cluster or noisy clustering

        # Set up bounds for the distance threshold
        bounds = [(0.01, 1.0)]

        # Use differential evolution to minimize the objective function
        result = differential_evolution(objective_function, bounds, strategy='best1bin', maxiter=50, disp=True)

        # Extract the best threshold
        best_threshold = result.x[0]
        # print(f"Optimal distance threshold: {best_threshold}")

        return best_threshold
    
    def optimize_by_histogram(self, rows):
        word_counts = self.create_wordcount_histogram(rows)
        best_threshold = self.determine_threshold_based_on_vmr(word_counts)
        # print(f"Optimal distance threshold: {best_threshold}")
        return best_threshold
    
    def create_wordcount_histogram(self, row_list):
        word_counts = np.zeros(len(row_list))
        for i, item in enumerate(row_list):
            word_counts[i] = len(item['text']['text'])
        return word_counts

    def calculate_vmr(self, word_counts):
        mean_count = np.mean(word_counts)
        variance = np.var(word_counts)
        vmr = variance / mean_count if mean_count != 0 else 0
        # print(f"VMR: {vmr}")
        return vmr

    def determine_threshold_based_on_vmr(self, word_counts):

        vmr = self.calculate_vmr(word_counts)

        # Calculate slope (m) and intercept (c) for linear transformation
        m = 0.15
        c = 0.55

        # Calculate the threshold using the linear equation
        threshold = c - m * vmr

        return threshold


    def cluster_lines_with_agglomerative_jaccard(self, distance_matrix, distance_threshold=0.4, linkage='average'):
        """
        Clusters lines using Agglomerative Clustering with Jaccard distances.

        Args:
        - distance_matrix: Precomputed Jaccard distance matrix (square 2D numpy array).
        - distance_threshold: The threshold for merging clusters.
        - linkage: Linkage criterion ('ward', 'complete', 'average', 'single').

        Returns:
        - labels: Cluster labels for each row.
        """
        # Initialize Agglomerative Clustering with the desired parameters
        agglom = AgglomerativeClustering(
            n_clusters=None,
            metric='precomputed',
            distance_threshold=distance_threshold,
            linkage=linkage
        )

        # Fit the model using the precomputed Jaccard distance matrix
        labels = agglom.fit_predict(distance_matrix)

        return labels
    
    def calculate_clustering_metrics(self, features, labels):
        """Calculate clustering metrics given feature vectors and cluster labels."""
        n_labels = len(set(labels))
        if n_labels > 1:
            sil_score = silhouette_score(features, labels)
            db_score = davies_bouldin_score(features, labels)
            ch_score = calinski_harabasz_score(features, labels)

            print(f"Silhouette Score: {sil_score}")
            print(f"Davies-Bouldin Index: {db_score}")
            print(f"Calinski-Harabasz Index: {ch_score}")

        else:
            print(f"Clustering metrics cannot be calculated with {n_labels} label(s). "
                "Ensure there are at least two clusters with more than one sample each.")
    
    def check_clustering_importance(self, distance_matrix, labels):
        """Compares the WSS of single vs. multiple clusters."""
        def calculate_wss(X, labels):
            """Calculate within-cluster sum of squares."""
            wss = 0
            for label in np.unique(labels):
                cluster_points = X[labels == label]
                centroid = cluster_points.mean(axis=0)
                wss += ((cluster_points - centroid) ** 2).sum()
            return wss

        # Assuming all data belongs to a single cluster initially
        single_cluster_labels = np.zeros(len(labels), dtype=int)
        single_cluster_wss = calculate_wss(distance_matrix, single_cluster_labels)

        # Calculate WSS after clustering (with optimized distance threshold)
        multi_cluster_wss = calculate_wss(distance_matrix, labels)

        print(f"Single Cluster WSS: {single_cluster_wss}")
        print(f"Multiple Cluster WSS: {multi_cluster_wss}")
    
    def find_largest_cluster_width(self, clusters):
        max_width = 0
        for cluster in clusters:
            # Determine the starting (leftmost) position of the cluster
            cluster_left = min(pos[0] for pos in cluster['positions'])
            cluster_right = max(pos[1] for pos in cluster['positions'])
            cluster_width = abs(cluster_right - cluster_left)
            if cluster_width > max_width:
                max_width = cluster_width
        return max_width

    def recombine_clusters_with_reference(self, grouped_data, num_bins=1000, similarity_threshold=0.8, plot_heatmaps=False):
        def create_fixed_width_heatmap(cluster, num_bins, ref_width):
            """Generate a density heatmap for a single cluster using a fixed reference width."""
            density_vector = np.zeros(num_bins)

            if cluster['positions'].size == 0:
                # If the cluster is empty, return an empty feature array with zeros
                row_level_features = np.zeros((num_bins, 0))
                return density_vector, row_level_features

            row_level_features = np.zeros((num_bins, len(cluster['positions'])))

            # Determine the starting (leftmost) position of the cluster
            cluster_left = min(pos[0] for pos in cluster['positions'])

            for sample, x_positions in enumerate(cluster['positions']):
                # Map the positions onto the density vector with a fixed width
                left_idx = int(((x_positions[0] - cluster_left) / ref_width) * num_bins)
                right_idx = int(((x_positions[1] - cluster_left) / ref_width) * num_bins)

                density_vector[left_idx:right_idx] += 1  # Increase density
                row_level_features[left_idx:right_idx, sample] = 1

            cluster_feature = density_vector / max(density_vector) if max(density_vector) != 0 else density_vector

            return cluster_feature, row_level_features

        def relative_error(u, v):
            """Calculate mean squared error between two vectors normalized by the sum of both areas."""
            mse = np.mean((u - v) ** 2)
            area_sum = np.sum(u) + np.sum(v)
            return mse / area_sum if area_sum != 0 else 0

        ref_width = self.find_largest_cluster_width(grouped_data)

        # Create heatmaps and row-level features for each cluster using the fixed width
        results = [create_fixed_width_heatmap(cluster, num_bins, ref_width) for cluster in grouped_data]
        cluster_heatmaps, cluster_features = zip(*results)

        # Flatten the list of arrays into a 2D array where each row corresponds to an individual row sample
        all_row_level_features = np.hstack(cluster_features).T

        # Create corresponding labels for the row-level features
        row_labels = []
        for cluster_idx, features in enumerate(cluster_features):
            row_labels.extend([cluster_idx] * features.shape[1])

        row_labels = np.array(row_labels)

        if plot_heatmaps:
            num_clusters = len(cluster_heatmaps)
            fig, axs = plt.subplots(num_clusters, 1, figsize=(12, num_clusters * 2), sharex=True)

            for i, heatmap in enumerate(cluster_heatmaps):
                axs[i].bar(range(num_bins), heatmap, width=1)
                axs[i].set_title(f'Cluster {i}')
                axs[i].set_ylabel('Word Density')
                axs[i].grid(False)

            axs[-1].set_xlabel('Bins (Position)')
            plt.tight_layout()
            plt.show()

        # Initialize recombined labels
        recombined_labels = np.arange(len(grouped_data))

        # Calculate clustering metrics before recombination
        print("\nINITIAL CLUSTERING METRICS\n")
        self.calculate_clustering_metrics(all_row_level_features, row_labels)

        # Calculate the pairwise relative error distance matrix between cluster heatmaps
        relative_error_distances = pdist(cluster_heatmaps, metric=relative_error)
        distance_matrix = squareform(relative_error_distances)

        # Recombine clusters based on the relative error similarity threshold
        for i in range(len(distance_matrix)):
            for j in range(i + 1, len(distance_matrix)):
                if distance_matrix[i, j] <= similarity_threshold:
                    recombined_labels[j] = recombined_labels[i]

        # Perform ordinal label remapping
        unique_labels = sorted(set(recombined_labels))
        lbl_idx = 0
        for label in unique_labels:
            recombined_labels[recombined_labels == label] = lbl_idx
            lbl_idx += 1

        # Merged Clusters
        merged_clusters = self.merge_clusters(grouped_data, recombined_labels)
    
        # Create new heatmaps and features for the merged clusters to compare clustering scores
        merged_results = [create_fixed_width_heatmap(cluster, num_bins, ref_width) for cluster in merged_clusters]
        _, merged_cluster_features = zip(*merged_results)
        # Flatten the list of arrays into a 2D array where each row corresponds to an individual row sample
        all_merged_row_features = np.hstack(merged_cluster_features).T

        # Create corresponding labels for the merged row-level features
        merged_row_labels = []
        for cluster_idx, features in enumerate(merged_cluster_features):
            merged_row_labels.extend([cluster_idx] * features.shape[1])

        merged_row_labels = np.array(merged_row_labels)

        # Calculate clustering metrics after recombination
        print("\nCLUSTERING METRICS AFTER REGROUPING\n")
        self.calculate_clustering_metrics(all_merged_row_features, merged_row_labels)

        return merged_clusters
    
    def merge_clusters(self, grouped_data, reassigned_labels):
        """
        Merges clusters in `grouped_data` based on the reassigned labels.
        
        Args:
        - grouped_data: Original grouped data containing clusters.
        - reassigned_labels: New cluster assignments for each group.
        
        Returns:
        - merged_data: New grouped data after merging.
        """
        # Initialize a dictionary to hold the new merged clusters
        merged_clusters = {}
        num_labels = len(set(reassigned_labels))

        # Create empty cluster structures for each new label
        for label in range(num_labels):
            merged_clusters[label] = {
                'positions': [],
                'global_positions': [],
                'texts': [],
                'y_positions': [],
                'y_global_positions': [],
                'line_numbers': []
            }

        # Traverse the original grouped data and append to the new structures
        for old_label, new_label in enumerate(reassigned_labels):

            group = grouped_data[old_label]
            merged_clusters[new_label]['positions'].extend(group['positions'])
            merged_clusters[new_label]['global_positions'].extend(group['global_positions'])
            merged_clusters[new_label]['texts'].extend(group['texts'])
            merged_clusters[new_label]['y_positions'].extend(group['y_positions'])
            merged_clusters[new_label]['y_global_positions'].extend(group['y_global_positions'])

            # Keep track of new line number by taking max of previous line number
            if len(merged_clusters[new_label]['line_numbers']) > 0:
                reference_line_number = max(merged_clusters[new_label]['line_numbers']) + 1 # +1 to account for 0 index
                merged_clusters[new_label]['line_numbers'].extend(group['line_numbers'] + reference_line_number)
            else:
                merged_clusters[new_label]['line_numbers'].extend(group['line_numbers'])

        # Convert the merged clusters back to a list
        merged_data = [
            {
                'positions': np.array(cluster['positions']),
                'global_positions': np.array(cluster['global_positions']),
                'texts': cluster['texts'],
                'y_positions': np.array(cluster['y_positions']),
                'y_global_positions': np.array(cluster['y_global_positions']),
                'line_numbers': cluster['line_numbers']
            }
            for cluster in merged_clusters.values()
        ]

        return merged_data
        
    def group_rows_by_labels(self, all_rows, labels):
        """
        Group rows based on cluster labels and organize them into the required format for further text classification.
        
        Args:
        - all_rows: List of dictionaries containing the row-level data.
        - labels: List of labels assigning each row to a cluster.

        Returns:
        - grouped_data: A list of grouped tables, each containing row text and position data.
        """
        # Initialize a dictionary to hold grouped tables by their labels
        grouped_tables = {}
        lines_count = np.zeros(len(labels))

        for row, label in zip(all_rows, labels):
            if label not in grouped_tables:
                grouped_tables[label] = {
                    'positions': [],
                    'global_positions': [],
                    'texts': [],
                    'y_positions': [],
                    'y_global_positions': [],
                    'line_numbers': []
                }

            # Extract text and positional data from the row
            text_data = row['text']
            left = text_data['left']
            top = text_data['top']
            width = text_data['width']
            height = text_data['height']
            texts = text_data['text']

            # Extract box position
            box = row['box']

            # Append data to the corresponding group
            for i, text in enumerate(texts):
                if text.strip():  # Ensure the text is not empty
                    x1 = left[i]
                    x2 = x1 + width[i]
                    y1 = top[i]

                    x1_global = box[0] + left[i]
                    x2_global = x1_global + width[i]
                    y1_global = box[1] + top[i]
                    y2_global = y1_global + height[i]

                    grouped_tables[label]['positions'].append([x1, x2])
                    grouped_tables[label]['global_positions'].append([x1_global, x2_global])
                    grouped_tables[label]['texts'].append(text)
                    grouped_tables[label]['y_positions'].append((y1, height[i]))
                    grouped_tables[label]['y_global_positions'].append((y1_global, y2_global))
                    grouped_tables[label]['line_numbers'].append(lines_count[label])
            
            lines_count[label] += 1

        # Convert the grouped tables to a list of dictionaries
        grouped_data = [
            {
                'positions': np.array(group['positions']),
                'global_positions': np.array(group['global_positions']),
                'texts': group['texts'],
                'y_positions': np.array(group['y_positions']),
                'y_global_positions': np.array(group['y_global_positions']),
                'line_numbers': group['line_numbers']
            }
            for group in grouped_tables.values()
        ]

        return grouped_data

    def plot_global_bounding_boxes(self, grouped_data, image, ax = None):
        """
        Plots the bounding boxes for each class using global positions on the given image.

        Args:
        - grouped_data: A list of dictionaries containing grouped rows with global bounding box positions.
        - image: The image on which to draw the bounding boxes.
        """
        # Initialize the plot with the image
        if ax == None:
            fig, ax = plt.subplots(figsize=(15, 15))
        ax.imshow(image)
        legend_elements = []

        # Iterate through the grouped data to plot each class's bounding boxes
        for label, group in enumerate(grouped_data):
            global_positions = group['global_positions']
            y_global_positions = group['y_global_positions']

            color = f'C{label}'
            legend_elements.append(Line2D([0], [0], color=color, lw=4, label=f'Table {label}'))

            for (x1, x2), (y1, y2) in zip(global_positions, y_global_positions):
                # Create a rectangle representing the bounding box
                rect = patches.Rectangle(
                    (x1, y1),  # Bottom-left corner (global position)
                    x2 - x1,   # Width
                    y2 - y1,   # Height
                    linewidth=1,
                    edgecolor=color,  # Use a different color per label
                    facecolor=color,
                    alpha = 0.4
                )
                ax.add_patch(rect)

        # Add the legend to the plot
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        # Display the plot
        plt.axis('off')
        plt.show()
