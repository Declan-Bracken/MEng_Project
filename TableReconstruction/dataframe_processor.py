import pandas as pd
import numpy as np

class DataFrameProcessor:
    def process_tables_to_dataframe(self, tables_data, clusterer):
        all_tables_dfs = []

        for table_data in tables_data:

            positions = table_data['positions']
            texts = table_data['texts']
            line_numbers = table_data['line_numbers']

            num_lines = len(set(line_numbers))

            # Check for more than one word per row (IE more than one column):
            has_columns = len(line_numbers) != num_lines

            # Only perform clustering if table has more than one column and more than one row (otherwise single sample per label)
            # if has_columns and num_lines > 1:
            #     optimal_eps = clusterer.binary_search_optimize_eps(positions)
            #     labels = clusterer.perform_clustering(positions, optimal_eps)
            #     num_cols = len(set(labels))
            # elif num_lines > 1: # More than one line, but one column
            #     labels = [0]
            #     num_cols = len(set(labels))
            # else: # More than one column, but one line
            #     optimal_eps = 27.5
            #     labels = clusterer.perform_clustering(positions, optimal_eps)
            #     num_cols = len(set(labels))

            if has_columns and num_lines > 1:
                optimal_eps = clusterer.optimize_eps(positions)
                labels = clusterer.perform_clustering(positions, eps = optimal_eps)
                num_cols = len(set(labels))
            elif num_lines > 1: # More than one line, but one column
                labels = [0 for _ in range(num_lines)]
                num_cols = len(set(labels))
            else: # More than one column, but one line
                labels = clusterer.perform_clustering(positions, eps = 27.5)
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
    
class DataFrameProcessorV2():
    def process_tables_to_dataframe(self, tables_data, clusterer, min_samples = 4):
        all_tables_dfs = []

        for table_data in tables_data:

            positions = table_data['positions']
            texts = table_data['texts']
            line_numbers = table_data['line_numbers']

            num_lines = len(set(line_numbers))

            # Check for more than one word per row:
            has_columns = len(line_numbers) != num_lines

            if has_columns and num_lines > 1:
                labels = clusterer.perform_hdbscan_clustering(positions, min_cluster_size = num_lines, min_samples = min_samples)
                labels = clusterer.reorder_columns(labels, positions)
                # labels = column_clusterer.perform_clustering(positions, optimal_eps)
                num_cols = len(set(labels))
            elif num_lines > 1: # More than one line, but one column
                labels = [0 for _ in range(num_lines)]
                num_cols = len(set(labels))
            else: # More than one column, but one line
                optimal_eps = 27.5
                labels = clusterer.perform_clustering(positions, optimal_eps)
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
            

class DataFrameProcessorV3:
    def process_tables_to_dataframe(self, tables_data, clusterer, min_samples_list):
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
                labels = clusterer.perform_hdbscan_clustering(positions, min_cluster_size=num_lines, min_samples=min_samples)
                labels = clusterer.reorder_columns(labels, positions)
                num_cols = len(set(labels))
            elif num_lines > 1:  # More than one line, but one column
                labels = [0 for _ in range(num_lines)]
                num_cols = len(set(labels))
            else:  # More than one column, but one line
                optimal_eps = 27.5
                labels = clusterer.perform_clustering(positions, optimal_eps)
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