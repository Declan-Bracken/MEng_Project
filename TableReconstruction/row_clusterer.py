import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

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

        for i, line_num in enumerate(text_data['line_num']):
            if line_num not in rows:
                rows[line_num] = {
                    'box': table_data['box'],
                    'text': {
                        'left': [],
                        'top': [],
                        'width': [],
                        'height': [],
                        'text': []
                    }
                }
            rows[line_num]['text']['left'].append(text_data['left'][i])
            rows[line_num]['text']['top'].append(text_data['top'][i])
            rows[line_num]['text']['width'].append(text_data['width'][i])
            rows[line_num]['text']['height'].append(text_data['height'][i])
            rows[line_num]['text']['text'].append(text_data['text'][i])

        return list(rows.values())

    # Example function to collect rows across headers, tables, and single-row tables
    def collect_all_rows(self):
        """ Excluding Headers for Now.
        """
        all_rows = []
        # for header in self.headers:
        #     all_rows.extend(self._separate_rows(header))
        for row in self.single_row:
            all_rows.extend(self._separate_rows(row))
        for table in self.tables:
            all_rows.extend(self._separate_rows(table))

        # Clean Rows
        all_rows = self._clean_rows(all_rows)

        return all_rows
    
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

            # Filter out empty text entries
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

    def create_binary_heatmap_features(self, row_list, num_bins=1000):
        """
        Generates binary heatmap features for each row, indicating word presence/absence.
        
        Args:
        - row_list: List of dictionaries representing each row.
        - num_bins: The number of bins (fine grid) for each row.
        
        Returns:
        - An array with a fixed-length binary vector for each row.
        """
        features = []

        for item in row_list:
            # print(item)
            box = item['box']  # Bounding box of the cropped row
            text = item['text']
            left = text['left']
            width = text['width']
            
            # Calculate the global positions of the words
            global_lefts = [box[0] + l for l in left]
            global_rights = [box[0] + l + w for l, w in zip(left, width)]

            # Normalize positions to the bounding box width
            if global_lefts and global_rights:
                min_left = min(global_lefts)
                max_right = max(global_rights)
                normalized_lefts = [(l - min_left) / (max_right - min_left) for l in global_lefts]
                normalized_rights = [(r - min_left) / (max_right - min_left) for r in global_rights]
            else:
                normalized_lefts = []
                normalized_rights = []

            # Initialize a binary vector with zeros
            binary_vector = np.zeros(num_bins)

            # Mark bins where a word is present
            for left, right in zip(normalized_lefts, normalized_rights):
                left_idx = int(left * num_bins)
                right_idx = int(right * num_bins)
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
    
    def calculate_clustering_metrics(self, features, labels):
        # Assume 'features' contains the input data and 'labels' contains the predicted cluster labels
        sil_score = silhouette_score(features, labels)
        db_score = davies_bouldin_score(features, labels)
        ch_score = calinski_harabasz_score(features, labels)

        print(f"Silhouette Score: {sil_score}")
        print(f"Davies-Bouldin Index: {db_score}")
        print(f"Calinski-Harabasz Index: {ch_score}")

    def cluster_lines_with_agglomerative_jaccard(self, binary_features, distance_threshold=0.3, linkage='average'):
        """
        Clusters lines using Agglomerative Clustering with Jaccard distances.

        Args:
        - binary_features: 2D array-like binary feature vectors representing rows.
        - distance_threshold: The threshold for merging clusters.
        - linkage: Linkage criterion ('ward', 'complete', 'average', 'single').

        Returns:
        - labels: Cluster labels for each row.
        """
        # Compute the Jaccard distance matrix
        distance_matrix = self.calculate_jaccard_distance_matrix(binary_features)

        # Initialize Agglomerative Clustering with the desired parameters
        agglom = AgglomerativeClustering(
            n_clusters=None,
            affinity='precomputed',
            distance_threshold=distance_threshold,
            linkage=linkage
        )

        # Fit the model using the precomputed Jaccard distance matrix
        labels = agglom.fit_predict(distance_matrix)

        return labels

# Example usage
# row_list = [
#     {'box': np.array([1040.4, 377.89, 1926.3, 566]), 'text': {'left': [0, 100], 'width': [100, 200], 'text': ['APSC', '200']}},
#     {'box': np.array([91.53, 841.74, 974.96, 1000.4]), 'text': {'left': [0, 150], 'width': [80, 120], 'text': ['MECH', '101']}}
# ]
# features = extract_positional_features(row_list)
# labels = cluster_lines_with_positions(features)
# print(labels)
