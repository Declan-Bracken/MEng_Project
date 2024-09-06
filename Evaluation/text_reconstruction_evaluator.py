import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from difflib import SequenceMatcher

class TextReconstructionEvaluator:
    def __init__(self, predicted, actual):
        """
        Initializes the evaluator with lists of predicted and actual strings.

        Args:
        - predicted (list of str): The predicted outputs from the models.
        - actual (list of str): The ground truth outputs.
        """
        self.predicted = predicted
        self.actual = actual
        
        if len(predicted) != len(actual):
            raise ValueError("Predicted and actual lists must have the same length.")

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

    def evaluate(self):
        """
        Evaluates the reconstruction quality using Levenshtein Distance and Token-based F1 Score.

        Returns:
        - dict: A dictionary containing the average Levenshtein Distance and F1 Score across all pairs.
        """
        levenshtein_distances = []
        f1_scores = []
        
        for pred, actual in zip(self.predicted, self.actual):
            levenshtein_distances.append(self.levenshtein_distance(pred, actual))
            f1_scores.append(self.token_f1_score(pred, actual))
        
        avg_levenshtein = np.mean(levenshtein_distances)
        avg_f1_score = np.mean(f1_scores)
        
        results = {
            'Average Levenshtein Distance': avg_levenshtein,
            'Average Token F1 Score': avg_f1_score
        }
        
        return results

# Example usage:
if __name__ == "__main__":
    predicted_strings = [
        "MAT 101, Intro to Math, 3.000, A",
        "ENG 201, Advanced English, 4.000, B",
        "PHY 301, Physics, 3.000, C"
    ]
    actual_strings = [
        "MAT 101, Intro to Math, 3.000, A",
        "ENG 201, Advanced English, 4.000, B+",
        "PHY 301, Physics, 3.000, B"
    ]
    
    evaluator = TextReconstructionEvaluator(predicted_strings, actual_strings)
    results = evaluator.evaluate()
    print("Evaluation Results:", results)
