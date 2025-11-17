
import numpy as np
from typing import List, Tuple, Dict

class Metrics:
    """
    A utility class for computing various performance metrics
    for neural network classification tasks.
    """
    
    @staticmethod
    def accuracy(predictions: List[int], actual: List[int]) -> float:
        """
        Calculate accuracy: (correct predictions / total predictions)
        
        Args:
            predictions: List of predicted labels
            actual: List of actual labels
            
        Returns:
            Accuracy as a float between 0 and 1
        """
        if len(predictions) != len(actual):
            raise ValueError("Predictions and actual must have same length")
        
        correct = sum(1 for pred, act in zip(predictions, actual) if pred == act)
        return correct / len(actual) if len(actual) > 0 else 0.0
    
    @staticmethod
    def confusion_matrix(predictions: List[int], actual: List[int], num_classes: int = 10) -> np.ndarray:
        """
        Compute confusion matrix for multi-class classification.
        
        Args:
            predictions: List of predicted labels
            actual: List of actual labels
            num_classes: Number of classes (default 10 for MNIST)
            
        Returns:
            Confusion matrix as numpy array of shape (num_classes, num_classes)
            where element [i,j] represents predictions of class j when actual was class i
        """
        matrix = np.zeros((num_classes, num_classes), dtype=int)
        
        for pred, act in zip(predictions, actual):
            matrix[act][pred] += 1
            
        return matrix
    
    @staticmethod
    def precision_recall_f1(predictions: List[int], actual: List[int], num_classes: int = 10) -> Dict[str, np.ndarray]:
        """
        Calculate precision, recall, and F1 score for each class.
        
        Args:
            predictions: List of predicted labels
            actual: List of actual labels
            num_classes: Number of classes (default 10 for MNIST)
            
        Returns:
            Dictionary containing 'precision', 'recall', and 'f1' arrays for each class
        """
        cm = Metrics.confusion_matrix(predictions, actual, num_classes)
        
        precision = np.zeros(num_classes)
        recall = np.zeros(num_classes)
        f1 = np.zeros(num_classes)
        
        for i in range(num_classes):
            # True Positives: diagonal element
            tp = cm[i][i]
            
            # False Positives: sum of column i minus tp
            fp = np.sum(cm[:, i]) - tp
            
            # False Negatives: sum of row i minus tp
            fn = np.sum(cm[i, :]) - tp
            
            # Calculate precision
            precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            
            # Calculate recall
            recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            # Calculate F1 score
            if precision[i] + recall[i] > 0:
                f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
            else:
                f1[i] = 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    @staticmethod
    def print_metrics_report(predictions: List[int], actual: List[int], num_classes: int = 10):
        """
        Print a comprehensive metrics report.
        
        Args:
            predictions: List of predicted labels
            actual: List of actual labels
            num_classes: Number of classes (default 10 for MNIST)
        """
        accuracy = Metrics.accuracy(predictions, actual)
        metrics = Metrics.precision_recall_f1(predictions, actual, num_classes)
        cm = Metrics.confusion_matrix(predictions, actual, num_classes)
        
        print("\n" + "="*60)
        print("PERFORMANCE METRICS REPORT")
        print("="*60)
        print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Total Samples: {len(actual)}")
        print(f"Correct Predictions: {int(accuracy * len(actual))}")
        print(f"Incorrect Predictions: {len(actual) - int(accuracy * len(actual))}")
        
        print("\n" + "-"*60)
        print("PER-CLASS METRICS")
        print("-"*60)
        print(f"{'Class':<8} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
        print("-"*60)
        
        for i in range(num_classes):
            print(f"{i:<8} {metrics['precision'][i]:<12.4f} {metrics['recall'][i]:<12.4f} {metrics['f1'][i]:<12.4f}")
        
        print("\n" + "-"*60)
        print("AVERAGE METRICS")
        print("-"*60)
        print(f"Average Precision: {np.mean(metrics['precision']):.4f}")
        print(f"Average Recall: {np.mean(metrics['recall']):.4f}")
        print(f"Average F1 Score: {np.mean(metrics['f1']):.4f}")
        
        print("\n" + "-"*60)
        print("CONFUSION MATRIX")
        print("-"*60)
        print("(Rows: Actual, Columns: Predicted)")
        print()
        
        # Print header
        print("     ", end="")
        for i in range(num_classes):
            print(f"{i:>5}", end="")
        print()
        print("     " + "-"*5*num_classes)
        
        # Print matrix
        for i in range(num_classes):
            print(f"{i:<4}|", end="")
            for j in range(num_classes):
                print(f"{cm[i][j]:>5}", end="")
            print()
        
        print("="*60 + "\n")