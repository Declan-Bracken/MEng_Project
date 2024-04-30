from ocr_processor import OCRProcessor
from vision_pipeline import VisionPipeline

class TranscriptPipeline():
    def __init__(self, cnn_path = r'yolo_training\yolo_v8_models\finetune_v4 (3_classes)\best (1).pt'):
        
        self.vision_pipeline = VisionPipeline(cnn_path)
        self.ocr_processor = OCRProcessor()

    def get_table_strings(self, image_path, plot_bboxes = False, **kwargs):
        # Predict bboxes
        results = self.vision_pipeline.predict(image_path, plot = plot_bboxes, **kwargs)
        # Get ocr output
        processed_results = self.ocr_processor.process_images_with_ocr(results, image_path, use_tesseract=True)
        # Clean and format strings
        formatted_strings = self.ocr_processor.format_strings(processed_results)
        return formatted_strings

if __name__ == "__main__":
    # Set model path
    vision_model_path = r'yolo_training\yolo_v8_models\finetune_v4 (3_classes)\best (1).pt'
    # Set transcript path
    transcript_path = r'\Users\Declan Bracken\Pictures\Saved Pictures\2015-queens-university-transcript-1-2048.webp'

    # Instantiate pipeline
    pipeline = TranscriptPipeline(cnn_path = vision_model_path)
    table_and_header_dict = pipeline.get_table_strings(transcript_path, plot_bboxes = True,
                                            iou = 0.3, conf = 0.2, agnostic_nms = True)
    first_image_name = next(iter(table_and_header_dict))
    print(table_and_header_dict[first_image_name]['header_data'])
    print(table_and_header_dict[first_image_name]['table_data'])


