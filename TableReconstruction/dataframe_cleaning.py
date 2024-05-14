# dataframe_cleaning.py
import pandas as pd
import re

class DataFrameCleaning:
    def __init__(self):
        self.grade_corrections = {
            'At': 'A+',
            'Bf': 'B+',
            'C_': 'C-',
            # Add more common OCR mistakes and corrections
        }
        self.course_code_pattern = re.compile(r'^[A-Z]{3}\d{3}$')  # Example pattern for course codes
        self.grade_pattern = re.compile(r'^[A-F][+-]?$')  # Example pattern for grades

    def ensure_required_columns(self, df, required_columns):
        missing_columns = [col for col in required_columns if col not in df.columns]
        for col in missing_columns:
            df[col] = None
        return df

    def split_combined_columns(self, df):
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].apply(lambda x: re.split(r'\s+|\|', x) if isinstance(x, str) else x)
                df = df.explode(col).reset_index(drop=True)
        return df

    def clean_data(self, df):
        # Correct OCR mistakes in grades
        if 'Grade' in df.columns:
            df['Grade'] = df['Grade'].replace(self.grade_corrections)
        
        # Ensure course codes follow the pattern
        if 'Course Code' in df.columns:
            df['Course Code'] = df['Course Code'].apply(lambda x: x if self.course_code_pattern.match(x) else None)
        
        # Ensure grades follow the pattern
        if 'Grade' in df.columns:
            df['Grade'] = df['Grade'].apply(lambda x: x if self.grade_pattern.match(x) else None)
        
        # Correct common OCR mistakes
        for col in df.columns:
            df[col] = df[col].replace(self.grade_corrections)
        
        return df

    def process_dataframes(self, dataframes, required_columns):
        cleaned_dfs = []
        for df in dataframes:
            df = self.ensure_required_columns(df, required_columns)
            df = self.split_combined_columns(df)
            df = self.clean_data(df)
            cleaned_dfs.append(df)
        return cleaned_dfs
