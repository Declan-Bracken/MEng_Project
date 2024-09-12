import matplotlib.pyplot as plt
import json
import pandas as pd
import numpy as np

class GraphResults:
    def __init__(self, filepath):
        self.filepath = filepath
        self.load_results()
        self.remap_pipeline_names()

        plt.rcParams.update({
            'text.usetex': True,
            'font.size': 14,           # Default font size
            'axes.titlesize': 18,      # Title font size
            'axes.labelsize': 16,      # X and Y label font size
            'xtick.labelsize': 14,     # X tick label font size
            'ytick.labelsize': 14,     # Y tick label font size
            'legend.fontsize': 14,     # Legend font size
            'figure.titlesize': 20     # Figure title font size
        })


    def load_results(self):
        """
        Loads the results from a JSON file, excluding 'pipe2' and 'BLEU Score'.
        """
        try:
            with open(self.filepath, 'r') as f:
                self.results = json.load(f)
                
                # Remove 'pipe2' and 'BLEU Score' from the results
                self.results.pop("Evaluation/Test_Results/pipe2_output.json", None)
                
                for pipeline in self.results.keys():
                    if "Average Results" in self.results[pipeline]:
                        self.results[pipeline]["Average Results"].pop("Average BLEU Score", None)
                    for image in self.results[pipeline]["Image Results"]:
                        self.results[pipeline]["Image Results"][image].pop("BLEU Score", None)
                        
        except Exception as e:
            raise Exception(f"Unable to load dataset as json: {e}")

    def remap_pipeline_names(self):
        """
        Remaps the pipeline names to 'Method 1', 'Method 2', and 'Method 3'.
        """
        pipeline_mapping = {
            "Evaluation/Test_Results/pipe1_output.json": "Method 1", # 0-shot
            "Evaluation/Test_Results/pipe3_output.json": "Method 2", # Prompt Pipeline
            "Evaluation/Test_Results/pipeline_results.json": "Method 3" # Modular Pipeline
        }
        self.results = {pipeline_mapping.get(k, k): v for k, v in self.results.items()}

    def get_average_results(self):
        """
        Extracts the average results for each pipeline.

        Returns:
        - dict: A dictionary where keys are pipeline names and values are average result dictionaries.
        """
        avg_results = {}
        for pipeline, data in self.results.items():
            avg_results[pipeline] = data["Average Results"]
        return avg_results

    def create_results_table(self):
        """
        Creates a pandas DataFrame summarizing the average results for each pipeline.

        Returns:
        - pd.DataFrame: DataFrame summarizing average results.
        """
        avg_results = self.get_average_results()
        df = pd.DataFrame(avg_results).T
        return df
    
    def plot_individual_metrics(self, save_dir):
        # Define metrics and their colors
        metrics = ['Average Levenshtein Distance', 'Average Token F1 Score', 'Average ROUGE-L Score', 'Average Word Error Rate']
        colors = ['skyblue', 'lightgreen', 'salmon', 'orange']
        metric_ylabels = {
            metrics[0]: 'Levenshtein Distance',
            metrics[1]: 'Token F1 Score',
            metrics[2]: 'ROUGE-L Score',
            metrics[3]: 'Word Error Rate'
        }

        # Get average values from results
        averages = self.get_average_results()
        method_names = list(averages.keys())
        n_methods = len(method_names)
        x = np.arange(n_methods)

        for i, metric in enumerate(metrics):
            metric_values = [averages[method][metric] for method in method_names]

            plt.figure(figsize=(10, 6))
            plt.bar(x, metric_values, color=colors[i], width=0.5)
            plt.xticks(x, method_names, fontsize=12)
            plt.xlabel('Methods', fontsize=14)
            plt.ylabel(metric_ylabels[metric], fontsize=14)
            # plt.title(f'Comparison of {metric} Across Methods', fontsize=16)
            # plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add value labels on bars
            for j, val in enumerate(metric_values):
                plt.text(j, val + 0.02, f'{val:.2f}', ha='center', fontsize=12)
            plt.savefig(save_dir + f"Fig_Average_{metric}",dpi = 200)
            plt.show()

    def plot_combined_metrics(self):
        """
        Plots a color-coordinated bar chart with multiple axes for the average results across pipelines.
        """
        avg_results = self.get_average_results()
        df = pd.DataFrame(avg_results).T

        
        # Define colors for each metric
        colors = {
            'Average Levenshtein Distance': 'skyblue',
            'Average Token F1 Score': 'lightgreen',
            'Average ROUGE-L Score': 'salmon',
            'Average Word Error Rate': 'orange'
        }

        # Set up the figure with multiple axes
        fig, ax1 = plt.subplots(figsize=(12, 8))

        # Define positions for bars
        bar_width = 0.2
        index = np.arange(len(df))

        # Plot bars for each metric with different axes
        ax1.bar(index, df['Average Levenshtein Distance'], bar_width, label='Levenshtein Distance', color=colors['Average Levenshtein Distance'])
        ax2 = ax1.twinx()
        ax2.bar(index + bar_width, df['Average Token F1 Score'], bar_width, label='Token F1 Score', color=colors['Average Token F1 Score'])
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        ax3.bar(index + 2 * bar_width, df['Average ROUGE-L Score'], bar_width, label='ROUGE-L Score', color=colors['Average ROUGE-L Score'])
        ax4 = ax1.twinx()
        ax4.spines['right'].set_position(('outward', 120))
        ax4.bar(index + 3 * bar_width, df['Average Word Error Rate'], bar_width, label='Word Error Rate', color=colors['Average Word Error Rate'])

        # Set labels and title
        ax1.set_xlabel('Methods')
        ax1.set_ylabel('Levenshtein Distance')
        ax2.set_ylabel('Token F1 Score')
        ax3.set_ylabel('ROUGE-L Score')
        ax4.set_ylabel('Word Error Rate')
        plt.title('Comparison of Average Results Across Methods')

        # Set ticks and legend
        ax1.set_xticks(index + bar_width)
        ax1.set_xticklabels(df.index)
        fig.legend(loc='upper left')#, bbox_to_anchor=(1, 1)

        plt.tight_layout()
        plt.show()
    
    def plot_image_sample_performance(self, image_sample, save_dir = None):
        """
        Plots a bar chart comparing the performance of each method for a specific image sample.

        Args:
        - image_sample (str): The image sample for which to plot the performance.
        """
        # Data preparation for plotting
        # print(self.results)
        image_data = {}
        for method, method_results in self.results.items():
            if image_sample in method_results['Image Results']:
                print('gotcha')
                image_data[method] = method_results['Image Results'][image_sample]
        
        # Exclude Bleu Score and Pipeline 2
        if 'Evaluation/Test_Results/pipe2_output.json' in image_data:
            del image_data['Evaluation/Test_Results/pipe2_output.json']
        for data in image_data.values():
            if 'BLEU Score' in data:
                del data['BLEU Score']

        # Remap the method names
        method_mapping = {
            'Evaluation/Test_Results/pipe1_output.json': 'Method 1',
            'Evaluation/Test_Results/pipe3_output.json': 'Method 2',
            'Evaluation/Test_Results/pipeline_results.json': 'Method 3'
        }
        image_data = {method_mapping.get(k, k): v for k, v in image_data.items()}
        print(image_data)
        # Convert data to a DataFrame for easy plotting
        df = pd.DataFrame(image_data).T

        # Define colors for each metric
        colors = {
            'Levenshtein Distance': 'skyblue',
            'Token F1 Score': 'lightgreen',
            'ROUGE-L Score': 'salmon',
            'Word Error Rate': 'orange'
        }

        # Plotting
        fig, ax1 = plt.subplots(figsize=(10, 6))

        bar_width = 0.2
        index = np.arange(len(df))

        # Plot bars for each metric with different axes
        ax1.bar(index, df['Levenshtein Distance'], bar_width, label='Levenshtein Distance', color=colors['Levenshtein Distance'])
        ax2 = ax1.twinx()
        ax2.bar(index + bar_width, df['Token F1 Score'], bar_width, label='Token F1 Score', color=colors['Token F1 Score'])
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        ax3.bar(index + 2 * bar_width, df['ROUGE-L Score'], bar_width, label='ROUGE-L Score', color=colors['ROUGE-L Score'])
        ax4 = ax1.twinx()
        ax4.spines['right'].set_position(('outward', 120))
        ax4.bar(index + 3 * bar_width, df['Word Error Rate'], bar_width, label='Word Error Rate', color=colors['Word Error Rate'])

        # Set labels and title
        ax1.set_xlabel('Methods')
        ax1.set_ylabel('Levenshtein Distance')
        ax2.set_ylabel('Token F1 Score')
        ax3.set_ylabel('ROUGE-L Score')
        ax4.set_ylabel('Word Error Rate')
        # plt.title(f'Performance Comparison for Image Sample: {image_sample}')

        # Set ticks and legend
        ax1.set_xticks(index + bar_width)
        ax1.set_xticklabels(df.index)
        fig.legend(loc='upper left')

        plt.tight_layout()
        if save_dir:
            plt.savefig(save_dir + image_sample, dpi=200)
        plt.show()

# Example Usage
if __name__ == "__main__":
    filepath = r'Evaluation\Test_Results\results.json'  # Replace with your JSON file path
    graph_results = GraphResults(filepath)

    # Print average results table
    avg_results_df = graph_results.create_results_table()
    print(avg_results_df)

    # Plot combined metrics across methods
    # graph_results.plot_combined_metrics()
    sample_img = "Synthetic_Image_Annotation/Test_Data/images/1qXufiu8cLDN5CCxRI69T867qZSsxOd2X.jpg"
    save_dir = r"c:\Users\Declan Bracken\Pictures\Saved Pictures\Meng Project Figures\ "
    graph_results.plot_image_sample_performance(sample_img, save_dir=save_dir)

    # graph_results.plot_individual_metrics(save_dir = save_dir)


