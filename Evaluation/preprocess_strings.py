import json
import os
import re

class PreprocessStrings:
    def __init__(self, json_files, dataset_path):
        """
        Initializes the PreprocessStrings class.

        Args:
        - json_files (list): List of file paths to JSON files containing predicted strings.
        - dataset_path (str): Path to the JSON file containing the actual ground truth strings.
        """
        self.json_files = json_files
        self.dataset_path = dataset_path
        self.load_dataset()
        self.predictions = self.load_json_files()
    
    def load_dataset(self):
        """
        Loads the dataset from the provided path.

        Returns:
        - dict: Dictionary containing the dataset loaded from JSON.
        """
        try:
            with open(self.dataset_path, 'r') as f:
                self.dataset_list = json.load(f)
        except FileNotFoundError:
            print(f"Error: The file {self.dataset_path} was not found.")
        except json.JSONDecodeError:
            print(f"Error: The file {self.dataset_path} is not a valid JSON file.")
        
        # Dictionary mapping image locations to their ground truth csv strings
        self.dataset = {}

        for entry in self.dataset_list:
            try:
                image_path = entry["image"]
                csv_string = entry["conversations"][1]["content"]
                self.dataset[image_path] = csv_string
            except KeyError as e:
                print(f"Missing key {e} in entry: {entry}")

    def load_json_files(self):
        """
        Loads predicted strings from the provided JSON files.

        Returns:
        - dict: Dictionary where each key is the file name and value is the dictionary loaded from JSON.
        """
        predictions = {}
        for file_path in self.json_files:
            try:
                with open(file_path, 'r') as f:
                    predictions[file_path] = json.load(f)
            except FileNotFoundError:
                print(f"Error: The file {file_path} was not found.")
            except json.JSONDecodeError:
                print(f"Error: The file {file_path} is not a valid JSON file.")
        return predictions

    def preprocess_string(self, csv_string):
        """
        Cleans and standardizes the input CSV-like string for comparison.
        
        Args:
        - csv_string (str): The raw string from the OCR or model output.
        
        Returns:
        - str: The cleaned and standardized string.
        """
        # Remove non-ASCII characters
        csv_string = re.sub(r'[^\x00-\x7F]+', '', csv_string)
        
        # Split into lines to process each row separately
        lines = csv_string.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Normalize whitespace in each line
            line = re.sub(r'\s+', ' ', line).strip()
            
            # Remove unnecessary quotes
            line = line.replace("\"", "")
            
            # Convert to lowercase for consistent comparison
            line = line.lower()
            
            # Tokenize by commas, strip whitespace from each token, and join back
            tokens = [token.strip() for token in line.split(',')]
            cleaned_line = ','.join(tokens)
            
            # Add the cleaned line to the list
            cleaned_lines.append(cleaned_line)
        
        # Join the cleaned lines with newline characters to maintain row separation
        processed_string = '\n'.join(cleaned_lines)
        
        return processed_string

    def preprocess_data(self):
        """
        Preprocesses both predicted and actual strings.

        Returns:
        - dict: Dictionary containing preprocessed predictions and actuals for comparison.
        """
        # Preprocess predictions from JSON files
        preprocessed_predictions = {}
        for file_name, prediction_dict in self.predictions.items():
            preprocessed_predictions[file_name] = {
                key: self.preprocess_string(value) for key, value in prediction_dict.items()
            }

        # Preprocess actual strings from dataset
        preprocessed_actuals = {}
        for image_path, csv_string in self.dataset.items():
            preprocessed_actuals[image_path] = self.preprocess_string(csv_string)

        return preprocessed_predictions, preprocessed_actuals

# Example usage
if __name__ == "__main__":
    # Define paths to JSON files
    json_files = [
        'Evaluation/Test_Results/pipe1_output.json',
        'Evaluation/Test_Results/pipe2_output.json',
        'Evaluation/Test_Results/pipe3_output.json',
        'Evaluation/Test_Results/pipeline_results.json'
    ]
    dataset_path = 'Synthetic_Image_Annotation/Test_Data/Cleaned_JSON/Test_Responses.json'

    # Instantiate and preprocess strings
    preprocessor = PreprocessStrings(json_files, dataset_path)
    preprocessed_predictions, preprocessed_actuals = preprocessor.preprocess_data()

    # Print results for verification
    for file_name, predictions in preprocessed_predictions.items():
        print(f"Processed Predictions for {file_name}:")
        print(predictions)
    
    print("\nProcessed Actuals:")
    print(preprocessed_actuals)
