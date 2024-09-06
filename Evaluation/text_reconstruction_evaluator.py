import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from preprocess_strings import PreprocessStrings
import json

class TextReconstructionEvaluator:
    def __init__(self, json_files_path, dataset_path):
        """
        Initializes the evaluator with lists of predicted and actual strings.

        Args:
        - predicted (list of str): The predicted outputs from the models.
        - actual (list of str): The ground truth outputs.
        """
        self.json_files_path = json_files_path
        self.dataset_path = dataset_path
        self.preprocessor = PreprocessStrings(json_files_path, dataset_path)
        self.predictions, self.actuals = self.preprocessor.preprocess_data()
        

    def levenshtein_distance(self, s1, s2):
        """
        Computes the Levenshtein Distance between two strings.

        Args:
        - s1 (str): First string.
        - s2 (str): Second string.

        Returns:
        - int: Levenshtein Distance.
        """
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def token_f1_score(self, pred, actual):
        """
        Computes the F1 score based on token overlap between predicted and actual strings.

        Args:
        - pred (str): Predicted string.
        - actual (str): Actual ground truth string.

        Returns:
        - float: F1 score.
        """
        pred_tokens = set(pred.split())
        actual_tokens = set(actual.split())
        
        common_tokens = pred_tokens.intersection(actual_tokens)
        
        precision = len(common_tokens) / len(pred_tokens) if len(pred_tokens) > 0 else 0
        recall = len(common_tokens) / len(actual_tokens) if len(actual_tokens) > 0 else 0
        
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return f1_score
    
    def bleu_score(self, pred, actual):
        pred_tokens = pred.split()
        actual_tokens = actual.split()  # Reference is expected as a list of lists
        return sentence_bleu(actual_tokens, pred_tokens)

    def rouge_score(self, pred, actual):
        rouge = Rouge()
        scores = rouge.get_scores(pred, actual)
        return scores[0]['rouge-l']['f']
    
    def word_error_rate(self, pred, actual):
        pred_words = pred.split()
        actual_words = actual.split()
        return self.levenshtein_distance(pred_words, actual_words) / len(actual_words)

    def evaluate(self):
        """
        Evaluates the reconstruction quality for each prediction set and computes averages.

        Returns:
        - dict: A dictionary containing metrics for each prediction set.
        """
        results = {}

        for pred_set_name, pred_set in self.predictions.items():
            image_results = {}
            levenshtein_distances = []
            f1_scores = []
            bleu_scores = []
            rouge_scores = []
            wer_scores = []

            # Evaluate on an image-by-image basis
            for image_path, actual_text in self.actuals.items():
                if image_path in pred_set:
                    pred_text = pred_set[image_path]
                    lev_dist = self.levenshtein_distance(pred_text, actual_text)
                    f1 = self.token_f1_score(pred_text, actual_text)
                    bleu = self.bleu_score(pred_text, actual_text)
                    rouge = self.rouge_score(pred_text, actual_text)
                    wer = self.word_error_rate(pred_text, actual_text)

                    image_results[image_path] = {
                        'Levenshtein Distance': lev_dist,
                        'Token F1 Score': f1,
                        'BLEU Score': bleu,
                        'ROUGE-L Score': rouge,
                        'Word Error Rate': wer
                    }

                    # Collect metrics for averaging
                    levenshtein_distances.append(lev_dist)
                    f1_scores.append(f1)
                    bleu_scores.append(bleu)
                    rouge_scores.append(rouge)
                    wer_scores.append(wer)
                else:
                    raise("Image not found in Test Set.")

            # Calculate averages for this prediction set
            average_results = {
                'Average Levenshtein Distance': np.mean(levenshtein_distances),
                'Average Token F1 Score': np.mean(f1_scores),
                'Average BLEU Score': np.mean(bleu_scores),
                'Average ROUGE-L Score': np.mean(rouge_scores),
                'Average Word Error Rate': np.mean(wer_scores)
            }

            results[pred_set_name] = {
                'Image Results': image_results,
                'Average Results': average_results
            }

        return results
    
    def save_results(self, results, file_path):
        """
        Saves the evaluation results to a JSON file.

        Args:
        - results (dict): The evaluation results.
        - file_path (str): Path to the output JSON file.
        """
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {file_path}")

# Example usage:
if __name__ == "__main__":
    # Define paths to JSON files
    json_files = [
        'Evaluation/Test_Results/pipe1_output.json',
        'Evaluation/Test_Results/pipe2_output.json',
        'Evaluation/Test_Results/pipe3_output.json',
        'Evaluation/Test_Results/pipeline_results.json'
    ]
    dataset_path = 'Synthetic_Image_Annotation/Test_Data/Cleaned_JSON/Test_Responses.json'

    evaluator = TextReconstructionEvaluator(json_files, dataset_path)
    results = evaluator.evaluate()
    evaluator.save_results(results, "Evaluation/Test_Results/results.json")
    print("Evaluation Results:", results)
