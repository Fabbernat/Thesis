from typing import List, Tuple
import numpy as np

from wic_tfidf_baseline_test import print_evaluation_details
from metrics import EvaluationMetrics, calculate_metrics


def evaluate(
        similarities: np.ndarray,
        labels: List[str],
        data: List[Tuple],
        threshold: float = 0.449,
        verbose: bool = False
) -> EvaluationMetrics:
    """Evaluates WiC classification performance."""
    predictions = ['T' if sim > threshold else 'F' for sim in similarities]
    metrics = calculate_metrics(predictions, labels)

    if verbose:
        print_evaluation_results(predictions, labels, similarities, data, metrics)

    return metrics


def print_evaluation_results(
        predictions: List[str],
        labels: List[str],
        similarities: np.ndarray,
        data: List[Tuple],
        metrics: EvaluationMetrics
    ) -> None:
    """Prints detailed evaluation results."""
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