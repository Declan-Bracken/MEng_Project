import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

class InteractiveClusterTuning:
    def __init__(self, image, row_classifier, all_rows, best_threshold = 0.23, linkage = 'average'):
        self.image = image
        self.row_classifier = row_classifier
        self.all_rows = all_rows
        self.best_threshold = best_threshold
        self.linkage = linkage

        # Run the pipeline and plot initial results
        self.distance_matrix = self.get_distance_matrix()

        self.best_threshold_slider = widgets.FloatSlider(value=self.best_threshold, min=0.1, max=0.8, step=0.001, description='Lower <-  Segmentation  -> Greater:')
        self.recalculate_button = widgets.Button(description='Recalculate')
        self.recalculate_button.on_click(self.recalculate)
        self.output = widgets.Output()

        display(self.best_threshold_slider, self.recalculate_button, self.output)

    def get_distance_matrix(self):
        """Generate binary heatmap features and calculate Jaccard distance matrix."""
        row_features = self.row_classifier.create_binary_heatmap_features(self.all_rows)
        distance_matrix = self.row_classifier.calculate_jaccard_distance_matrix(row_features)
        return distance_matrix
    
    def recalculate(self, b):
        """Recalculate clustering based on the updated thresholds."""
        with self.output:
            self.output.clear_output(wait=True)
            self.best_threshold = self.best_threshold_slider.value
            self.plot_data()

    def plot_data(self):
        """Perform clustering and plot initial and regrouped data side by side."""
        labels = self.row_classifier.cluster_lines_with_agglomerative_jaccard(self.distance_matrix, distance_threshold=self.best_threshold, linkage=self.linkage)
        self.grouped_data = self.row_classifier.group_rows_by_labels(self.all_rows, labels)
        # self.regrouped_data = self.row_classifier.recombine_clusters_with_reference(self.grouped_data, similarity_threshold=self.similarity_threshold)

        fig, axes = plt.subplots(figsize=(10, 10), dpi=300)

        # Plot initial groupings
        self.plot_global_bounding_boxes(axes, self.grouped_data, self.image, title="Row Grouping")

        plt.show()

    def plot_global_bounding_boxes(self, ax, grouped_data, image, title=""):
        """Plot bounding boxes on the given axis."""
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

# Example usage
# image = plt.imread('path_to_your_image.png')
# row_classifier = RowClassifier(...)  # Initialize with your data
# all_rows = [...]  # Your rows data
# gui = InteractiveClusterTuning(image, row_classifier, all_rows)
