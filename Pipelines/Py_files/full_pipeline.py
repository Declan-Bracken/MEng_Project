from Pipelines.Py_files.ocr_processor import OCRProcessor
from Pipelines.Py_files.vision_pipeline import VisionPipeline
from Pipelines.Py_files.mistral_pipeline_v3 import MistralInference
import torch
import os
from glob import glob

class TranscriptPipeline():
    def __init__(self, device = torch.device('cuda'), cnn_path = r'yolo_training\yolo_v8_models\finetune_v5\best.pt',
                 LLM_path = r"d:\llm_models\Mistral-7B-Instruct-v0.3.fp16.gguf"):

        if device == torch.device('cuda') and torch.cuda.is_available():
            print("Using GPU Acceleration.")
            self.device = torch.device("cuda")
        else:
            print('GPU Unavailable, using CPU may take longer when running Mistral.')
            device = torch.device('cpu')

        self.vision_pipeline = VisionPipeline(cnn_path, device = self.device)                           # YOLO Model
        self.ocr_processor = OCRProcessor()                                                             # Tesseract Engine
        self.mistral_pipeline = MistralInference(device = self.device, model_path=LLM_path)             # LLM
    
    def get_image_paths(self, input_path):
        """
        Get a list of image file paths from a single file, list of files, or a directory.

        Args:
        - input_path (str or list): Path to an image file, a list of image file paths, or a directory.

        Returns:
        - list of str: A list of image file paths.
        """
        if isinstance(input_path, str):  # Single path provided
            if os.path.isdir(input_path):  # If it's a directory
                # Get all image files in the directory
                image_files = glob(os.path.join(input_path, "*.jpg")) + glob(os.path.join(input_path, "*.png")) + glob(os.path.join(input_path, "*.jpeg")) + glob(os.path.join(input_path, "*.webp"))
                if not image_files:
                    raise ValueError("No image files found in the provided directory.")
                return image_files
            elif os.path.isfile(input_path):  # If it's a single image file
                return [input_path]
            else:
                raise ValueError(f"The path provided is neither a file nor a directory: {input_path}")
        elif isinstance(input_path, list):  # List of file paths provided
            if all(os.path.isfile(path) for path in input_path):  # Check if all are valid files
                return input_path
            else:
                raise ValueError("One or more paths in the list are not valid files.")
        else:
            raise TypeError("Input should be a string (file or folder path) or a list of file paths.")
    
    
    def process_transcripts(self, input_path, plot_bboxes=False, **kwargs):
        """
        Processes a given input path (file, list of files, or directory) to extract tables from images and generate dataframes.

        Args:
        - input_path (str or list): Path to an image file, a list of image file paths, or a directory.
        - plot_bboxes (bool): Whether to plot bounding boxes.

        Returns:
        - dict: A dictionary where the key is the image path and the value is the Mistral LLM output (DataFrame).
        """
        # Get list of image paths
        image_paths = self.get_image_paths(input_path)
        
        all_tables = {}  # Dictionary to store the Mistral LLM output for each image
        
        # Process each image
        for image_path in image_paths:
            # Step 1: Run the vision pipeline to predict bounding boxes
            results = self.vision_pipeline.predict(image_path, plot=plot_bboxes, **kwargs)
            
            # Step 2: Process OCR output from predicted regions
            processed_results = self.ocr_processor.process_images_with_ocr(results, image_path, use_tesseract=True)
            # Clean and format strings
            formatted_strings = self.ocr_processor.format_strings(processed_results)
            formatted_strings = formatted_strings[next(iter(formatted_strings))]
            
            # Step 3: Extract 'table_data' and 'header_data' directly since there's only one entry
            if formatted_strings:  # Ensure there is some OCR output
                table_data = formatted_strings.get('table_data', '')
                header_data = formatted_strings.get('header_data', '')
                
                print(f"Processing {image_path}...")
                # Query Mistral LLM to generate CSV formatted table
                csv_output = self.mistral_pipeline.query_mistral(table_data, header_data)
                
                # If valid CSV output, store in the dictionary
                if csv_output:
                    all_tables[image_path] = csv_output
        
        return all_tables

if __name__ == "__main__":
    # Set model path
    vision_model_path = r'yolo_training\yolo_v8_models\finetune_v5\best.pt'
    # Set transcript path
    transcript_path = r'\Users\Declan Bracken\Pictures\Saved Pictures\2015-queens-university-transcript-1-2048.webp'

    # Instantiate pipeline
    pipeline = TranscriptPipeline(cnn_path = vision_model_path)
    # Process input (can be a file, list of files, or folder)
    tables = pipeline.process_transcripts(transcript_path, plot_bboxes=False, iou=0.3, conf=0.2, agnostic_nms=True)

    # Print the results
    print(tables)
