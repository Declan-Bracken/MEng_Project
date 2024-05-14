from PyQt5 import QtWidgets, QtCore
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.lines import Line2D
import matplotlib.patches as patches
import sys

class TableReconstructorGUI(QtWidgets.QWidget):
    def __init__(self, row_classifier, all_rows, image, best_threshold=0.37, similarity_threshold=0.0003, linkage='average'):
        super().__init__()
        self.row_classifier = row_classifier
        self.all_rows = all_rows
        self.image = image
        self.best_threshold = best_threshold
        self.similarity_threshold = similarity_threshold
        self.linkage = linkage
        self.regrouped_data = None

        # Main layout
        layout = QtWidgets.QVBoxLayout(self)

        # Row-Level Separation Threshold
        layout.addWidget(QtWidgets.QLabel("Row-Level Separation Threshold"))

        # Slider (mapped from 10-80 to 0.1-0.8)
        self.best_threshold_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.best_threshold_slider.setRange(10, 80)
        layout.addWidget(self.best_threshold_slider)

        # Spinbox (shows value directly as 0.1-0.8)
        self.best_threshold_spinbox = QtWidgets.QDoubleSpinBox()
        self.best_threshold_spinbox.setRange(0.1, 0.8)
        self.best_threshold_spinbox.setDecimals(2)
        self.best_threshold_spinbox.setSingleStep(0.01)
        layout.addWidget(self.best_threshold_spinbox)

        # Connect best threshold slider and spinbox
        self.best_threshold_spinbox.valueChanged.connect(self.update_best_slider)
        self.best_threshold_slider.valueChanged.connect(self.update_best_spinbox)
        self.best_threshold_slider.sliderReleased.connect(self.recalculate)

        # Initialize slider and spinbox values
        self.best_threshold_slider.setValue(int(self.best_threshold * 100))
        self.best_threshold_spinbox.setValue(self.best_threshold)

        # Cluster-Level Separation Threshold
        layout.addWidget(QtWidgets.QLabel("Cluster-Level Separation Threshold"))

        # Slider (mapped from 1-8 to 0.0001-0.0008)
        self.similarity_threshold_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.similarity_threshold_slider.setRange(10, 80)
        layout.addWidget(self.similarity_threshold_slider)

        # Spinbox (shows value directly as 0.0001-0.0008)
        self.similarity_threshold_spinbox = QtWidgets.QDoubleSpinBox()
        self.similarity_threshold_spinbox.setRange(0.0001, 0.0008)
        self.similarity_threshold_spinbox.setDecimals(5)
        self.similarity_threshold_spinbox.setSingleStep(0.00001)
        layout.addWidget(self.similarity_threshold_spinbox)

        # Connect similarity threshold slider and spinbox
        self.similarity_threshold_spinbox.valueChanged.connect(self.update_similarity_slider)
        self.similarity_threshold_slider.valueChanged.connect(self.update_similarity_spinbox)
        self.similarity_threshold_slider.sliderReleased.connect(self.recalculate)

        # Initialize slider and spinbox values
        self.similarity_threshold_slider.setValue(int(self.similarity_threshold * 10000))
        self.similarity_threshold_spinbox.setValue(self.similarity_threshold)

        # Add buttons
        recalculate_button = QtWidgets.QPushButton("Recalculate")
        recalculate_button.clicked.connect(self.recalculate)
        layout.addWidget(recalculate_button)

        finish_button = QtWidgets.QPushButton("Finish Tuning")
        finish_button.clicked.connect(self.finish_tuning)
        layout.addWidget(finish_button)

        # Set up plots
        fig = Figure(figsize=(20, 10))
        self.ax1, self.ax2 = fig.subplots(1, 2)
        self.canvas = FigureCanvas(fig)
        layout.addWidget(self.canvas)

        self.setLayout(layout)
        self.setWindowTitle("Table Reconstruction")

        # Initial calculations and plotting
        self.distance_matrix = self.get_distance_matrix()
        self.run_regrouping()
        self.plot_global_bounding_boxes(self.ax1, self.grouped_data)
        self.plot_global_bounding_boxes(self.ax2, self.regrouped_data)
        self.canvas.draw()

    def update_best_slider(self, value):
        """Update the best threshold slider based on the spinbox value."""
        self.best_threshold_slider.blockSignals(True)
        self.best_threshold_slider.setValue(int(value * 100))  # Convert to slider range
        self.best_threshold_slider.blockSignals(False)

    def update_best_spinbox(self, value):
        """Update the best threshold spinbox based on the slider value."""
        self.best_threshold_spinbox.blockSignals(True)
        self.best_threshold_spinbox.setValue(value / 100.0)  # Convert to spinbox range
        self.best_threshold_spinbox.blockSignals(False)

    def update_similarity_slider(self, value):
        """Update the similarity threshold slider based on the spinbox value."""
        self.similarity_threshold_slider.blockSignals(True)
        self.similarity_threshold_slider.setValue(int(value * 100000))  # Convert to slider range
        self.similarity_threshold_slider.blockSignals(False)

    def update_similarity_spinbox(self, value):
        """Update the similarity threshold spinbox based on the slider value."""
        self.similarity_threshold_spinbox.blockSignals(True)
        self.similarity_threshold_spinbox.setValue(value / 100000.0)  # Convert to spinbox range
        self.similarity_threshold_spinbox.blockSignals(False)

    def recalculate(self):
        """Recalculate the clusters based on updated thresholds."""
        self.best_threshold = self.best_threshold_spinbox.value()
        self.similarity_threshold = self.similarity_threshold_spinbox.value()

        # Rerun the pipeline and update the plots
        self.run_regrouping()
        self.ax1.clear()
        self.ax2.clear()
        self.plot_global_bounding_boxes(self.ax1, self.grouped_data)
        self.plot_global_bounding_boxes(self.ax2, self.regrouped_data)
        self.canvas.draw()

    def get_distance_matrix(self):
        """Generate the binary heatmap features and calculate the distance matrix."""
        row_features = self.row_classifier.create_binary_heatmap_features(self.all_rows)
        distance_matrix = self.row_classifier.calculate_jaccard_distance_matrix(row_features)
        return distance_matrix

    def run_regrouping(self):
        """Regroup data using updated thresholds."""
        labels = self.row_classifier.cluster_lines_with_agglomerative_jaccard(
            self.distance_matrix, distance_threshold=self.best_threshold, linkage=self.linkage
        )
        self.grouped_data = self.row_classifier.group_rows_by_labels(self.all_rows, labels)
        self.regrouped_data = self.row_classifier.recombine_clusters_with_reference(
            self.grouped_data, similarity_threshold=self.similarity_threshold
        )

    def finish_tuning(self):
        """Close the window."""
        self.close()

    def plot_global_bounding_boxes(self, ax, grouped_data):
        """Plot bounding boxes on the provided axes."""
        ax.imshow(self.image)
        legend_elements = []

        for label, group in enumerate(grouped_data):
            global_positions = group['global_positions']
            y_global_positions = group['y_global_positions']

            color = f'C{label}'
            legend_elements.append(Line2D([0], [0], color=color, lw=4, label=f'Table {label}'))

            for (x1, x2), (y1, y2) in zip(global_positions, y_global_positions):
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor=color, facecolor=color, alpha=0.4)
                ax.add_patch(rect)

        ax.legend(handles=legend_elements, loc='upper right', fontsize='xx-small')
        ax.axis('off')
        # Apply tight layout or similar adjustments to minimize whitespace
        ax.figure.tight_layout()


# Usage
# if __name__ == "__main__":
#     app = QtWidgets.QApplication(sys.argv)
#     window = TableReconstructorGUI(row_classifier, all_rows, image, best_threshold=0.23)
#     window.show()
#     sys.exit(app.exec_())
