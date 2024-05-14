import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

class StreamlitClusterTuning:
    def __init__(_self, image, _row_classifier, all_rows, best_threshold=0.5, linkage='average'):
        _self.image = image
        _self.row_classifier = _row_classifier
        _self.all_rows = all_rows
        _self.best_threshold = best_threshold
        _self.linkage = linkage

        _self.distance_matrix = _self.get_distance_matrix()

    def get_distance_matrix(_self):
        row_features = _self.row_classifier.create_binary_heatmap_features(_self.all_rows)
        distance_matrix = _self.row_classifier.calculate_jaccard_distance_matrix(row_features)
        return distance_matrix
    
    def update_clustering(_self, threshold):
        _self.best_threshold = threshold
        labels = _self.row_classifier.cluster_lines_with_agglomerative_jaccard(_self.distance_matrix, distance_threshold=_self.best_threshold, linkage=_self.linkage)
        _self.grouped_data = _self.row_classifier.group_rows_by_labels(_self.all_rows, labels)
        _self.plot_data()

    def plot_data(_self):
        fig, ax = plt.subplots(figsize=(10, 10), dpi=200)
        _self.plot_global_bounding_boxes(ax, _self.grouped_data, _self.image, title="Row Grouping")
        st.pyplot(fig)

    def plot_global_bounding_boxes(_self, ax, grouped_data, image, title=""):
        ax.imshow(image)
        legend_elements = []

        for label, group in enumerate(grouped_data):
            global_positions = group['global_positions']
            y_global_positions = group['y_global_positions']

            color = f'C{label}'
            legend_elements.append(Line2D([0], [0], color=color, lw=4, label=f'Table {label}'))

            for (x1, x2), (y1, y2) in zip(global_positions, y_global_positions):
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor=color, facecolor=color, alpha=0.4)
                ax.add_patch(rect)

        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        ax.set_title(title)
        ax.axis('off')
