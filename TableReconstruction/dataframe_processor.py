import pandas as pd
import numpy as np

class DataFrameProcessor:
    def process_tables_to_dataframe(self, tables_data, clusterer):
        all_tables_dfs = []

        for table_data in tables_data:

            positions = table_data['positions']
            texts = table_data['texts']
            line_numbers = table_data['line_numbers']

            optimal_eps = clusterer.binary_search_optimize_eps(positions)
            labels = clusterer.perform_clustering(positions, optimal_eps)
            num_cols = max(labels) + 1
            num_lines = max(line_numbers) + 1

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

        all_tables_combined_df = pd.concat(all_tables_dfs, axis=0, ignore_index=True)
        return all_tables_combined_df
