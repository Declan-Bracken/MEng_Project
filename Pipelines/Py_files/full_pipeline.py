from Py_files.ocr_processor import OCRProcessor
from Py_files.vision_pipeline import VisionPipeline
from Py_files.mistral_pipeline_v2 import MistralInference
import torch
import pandas as pd 

class TranscriptPipeline():
    def __init__(self, device = torch.device('cuda'), cnn_path = r'yolo_training\yolo_v8_models\finetune_v4 (3_classes)\best (1).pt',
                 LLM_path = r"C:\Users\Declan Bracken\MEng_Project\mistral\models\dolphin-2.1-mistral-7b.Q5_K_M.gguf"):

        if device == torch.device('cuda') and torch.cuda.is_available():
            print("Using GPU Acceleration.")
            self.device = torch.device("cuda")
        else:
            print('GPU Unavailable, using CPU may take longer when running Mistral.')
            device = torch.device('cpu')

        self.vision_pipeline = VisionPipeline(cnn_path, device = self.device)                           # YOLO Model
        self.ocr_processor = OCRProcessor()                                                             # Tesseract Engine
        self.mistral_pipeline = MistralInference(device = self.device, model_path=LLM_path)             # LLM

    def get_table_strings(self, image_path, plot_bboxes = False, **kwargs):
        # Predict bboxes
        results = self.vision_pipeline.predict(image_path, plot = plot_bboxes, **kwargs)
        # Get ocr output
        processed_results = self.ocr_processor.process_images_with_ocr(results, image_path, use_tesseract=True)
        # Clean and format strings
        formatted_strings = self.ocr_processor.format_strings(processed_results)
        return formatted_strings
    
    def get_dataframes(self, table_and_header_dict, output_filename = None):
        #Process Image Data Sequentially
        tables = {} # Dictionary of dataframes
        for image_name in table_and_header_dict:
            text = table_and_header_dict[image_name]['table_data']
            headers = table_and_header_dict[image_name]['header_data']
            table = self.mistral_pipeline.process_transcript(headers, text, output_filename = output_filename)
            tables[image_name] = table
        # Return List of dataframes, one for each image
        return tables

if __name__ == "__main__":
    # Set model path
    vision_model_path = r'yolo_training\yolo_v8_models\finetune_v4 (3_classes)\best (1).pt'
    # Set transcript path
    transcript_path = r'\Users\Declan Bracken\Pictures\Saved Pictures\2015-queens-university-transcript-1-2048.webp'

    # Instantiate pipeline
    pipeline = TranscriptPipeline(cnn_path = vision_model_path)
    table_and_header_dict = pipeline.get_table_strings(transcript_path, plot_bboxes = True,
                                            iou = 0.3, conf = 0.2, agnostic_nms = True)
    tables = pipeline.get_dataframes(table_and_header_dict, output_directory=None)

    pd.set_option('display.max_rows', 50)
    print(tables[0])

    
    




