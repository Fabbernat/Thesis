from dataclasses import dataclass
from typing import List
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
    """Calculate evaluation metrics for WiC classification."""
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