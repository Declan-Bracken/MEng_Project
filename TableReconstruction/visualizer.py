import seaborn as sns
import matplotlib.pyplot as plt
from PIL import ImageDraw
import matplotlib.colors as mcolors

class Visualizer:
    def plot_results(self, img, positions, texts, labels, y_positions):
        draw = ImageDraw.Draw(img)
        colors = list(mcolors.TABLEAU_COLORS.values())

        for (pos, text, label, y_pos) in zip(positions, texts, labels, y_positions):
            x1, x2 = pos
            y1, height = y_pos
            color = colors[label % len(colors)]
            draw.rectangle([x1, y1, x2, y1 + height], outline=color, width=2)
            draw.text((x1, y1), text, fill=color)

        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.show()

    def plot_distance_matrix(self, distance_matrix):
        plt.figure(figsize=(10, 8))
        sns.heatmap(distance_matrix, annot=False, cmap='coolwarm', fmt=".2f", square=True)
        plt.title("Pairwise Distance Matrix (Custom Metric)")
        plt.xlabel("Word Index")
        plt.ylabel("Word Index")
        plt.show()
