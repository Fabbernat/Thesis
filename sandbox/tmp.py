from typing import List, Tuple, Optional
from dataclasses import dataclass
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

@dataclass
class EvaluationMetrics:
    """Contains evaluation metrics for the WiC classification task."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    correct_predictions: int
    total_predictions: int


def calculate_metrics(
        predictions: List[str],
        labels: List[str]
) -> EvaluationMetrics:
    """
    Calculate evaluation metrics for WiC classification.

    Args:
        predictions: List of predicted labels ('T' or 'F')
        labels: List of true labels ('T' or 'F')

    Returns:
        EvaluationMetrics object containing accuracy, precision, recall, etc.
    """
    correct_predictions = sum(p == t for p, t in zip(predictions, labels))
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        pos_label='T',
        average='binary'
    )

    return EvaluationMetrics(
        accuracy=correct_predictions / len(labels),
        precision=precision,
        recall=recall,
        f1_score=f1,
        correct_predictions=correct_predictions,
        total_predictions=len(labels)
    )


def evaluate(
        similarities: np.ndarray,
        labels: List[str],
        data: List[Tuple],
        threshold: float = 0.449,
        verbose: bool = False
) -> EvaluationMetrics:
    """
    Evaluates WiC classification performance based on similarity threshold.

    Args:
        similarities: Array of similarity scores between sentence pairs
        labels: List of true labels ('T' or 'F')
        data: Original data tuples containing words and sentences
        threshold: Similarity threshold for classification
        verbose: If True, prints detailed error analysis

    Returns:
        EvaluationMetrics object
    """
    predictions = ['T' if sim > threshold else 'F' for sim in similarities]
    metrics = calculate_metrics(predictions, labels)

    if verbose:
        print("\nDetailed Error Analysis:")
        print_evaluation_details(predictions, labels, similarities, data,
                                 "False Positives", 'T', 'F')
        print_evaluation_details(predictions, labels, similarities, data,
                                 "False Negatives", 'F', 'T')

        print(f"\nEvaluation Metrics:")
        print(f"Accuracy: {metrics.accuracy:.3%}")
        print(f"Precision: {metrics.precision:.3%}")
        print(f"Recall: {metrics.recall:.3%}")
        print(f"F1 Score: {metrics.f1_score:.3%}")
        print(f"Correct predictions: {metrics.correct_predictions}/{metrics.total_predictions}")

    return metrics


def main():
    # Paths to WiC dataset files
    base_path = "C:/WiC_dataset/train"
    data_file = os.path.join(base_path, "train.data.txt")
    gold_file = os.path.join(base_path, "train.gold.txt")

    # Load data and compute similarities
    data, labels = load_wic_data(data_file, gold_file)
    similarities = compute_sentence_similarity(data)

    # Evaluate model with all metrics
    metrics = evaluate(similarities, labels, data, verbose=True)