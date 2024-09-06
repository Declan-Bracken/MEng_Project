import os
from Evaluation.text_reconstruction_evaluator import TextReconstructionEvaluator
from Pipelines.Py_files.full_pipeline import TranscriptPipeline
import json

class RunEvaluation():
    def __init__(self, test_set_path, save_file_path, vision_model_path = r'yolo_training\yolo_v8_models\finetune_v5\best.pt', 
                 llm_model_path = r'd:\llm_models\Mistral-7B-Instruct-v0.3.fp16.gguf'):
        self.vision_model_path = vision_model_path
        self.llm_model_path = llm_model_path
        self.test_set_path = test_set_path
        self.save_file_path = save_file_path
        self.load_dataset()
        self.create_image_list()

        # Instantiate pipeline
        self.transcript_pipeline = TranscriptPipeline(cnn_path = self.vision_model_path, LLM_path=self.llm_model_path)

    def load_dataset(self):
        try:
            with open(self.test_set_path, 'r') as f:
                self.dataset = json.load(f)
        except FileNotFoundError:
            print(f"Error: The file {self.test_set_path} was not found.")
        except json.JSONDecodeError:
            print(f"Error: The file {self.test_set_path} is not a valid JSON file.")
    
    def save_results(self, dict):
        # Check if the directory exists, if not, create it
        if not os.path.exists(self.save_file_path):
            os.makedirs(self.save_file_path)

        with open(self.save_file_path, 'w') as f:
            json.dump(dict, f, indent=4)
        print(f"Dataset saved to {self.save_file_path}")

    def create_image_list(self):
        # List of image locations to pass to pipeline
        self.image_list = []
        # Dictionary mapping image locations to their ground truth csv strings
        self.image_label_dict = {}

        for entry in self.dataset:
            try:
                image_path = entry["image"]
                csv_string = entry["conversations"][1]["content"]
                self.image_list.append(image_path)
                self.image_label_dict[image_path] = csv_string
            except KeyError as e:
                print(f"Missing key {e} in entry: {entry}")

    def run_pipeline(self, **kwargs):
        # Process input (can be a file, list of files, or folder)
        predicted_strings_dict = self.transcript_pipeline.process_transcripts(self.image_list, **kwargs)
        self.save_results(predicted_strings_dict)
        return predicted_strings_dict
    
    def evaluate_tables(self, predicted_strings_dict):
        results_dictionary = {}
        for image_path in predicted_strings_dict:
            evaluator = TextReconstructionEvaluator(predicted_strings_dict[image_path], self.image_label_dict[image_path])
            results = evaluator.evaluate()
            print("Evaluation Results for", image_path, ":", results)
            results_dictionary[image_path] = results
        return results_dictionary

if __name__ == "__main__":
    test_set_path = "Synthetic_Image_Annotation\Test_Data\Cleaned_JSON\Test_Responses - Copy.json"
    save_path = "Evaluation\Test_Results"

    run_evaluation = RunEvaluation(test_set_path, save_path)
    print(run_evaluation.image_list)
    print(run_evaluation.image_label_dict)

    run_evaluation.run_pipeline(plot_bboxes=True, iou=0.3, conf=0.2, agnostic_nms=True)
    