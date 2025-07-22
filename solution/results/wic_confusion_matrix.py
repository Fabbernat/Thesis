# C:\PycharmProjects\Peternity\solution\confusion_matrix.py
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from solution.implementation import similarity, combined, config
from independent_scripts.y_true.y_true_train import train_y_true


def plot_confusion_matrix(tn, fp, fn, tp, style='seaborn'):
    """
    Plots a labeled confusion matrix using either Seaborn or Matplotlib style.

    Parameters:
        tn (int): True Negative count
        fp (int): False Positive count
        fn (int): False Negative count
        tp (int): True Positive count
        style (str): Plotting style - 'seaborn' (default) or 'matplotlib'
    """
    matrix = np.array([[tn, fp], [fn, tp]])
    labels = [["TN", "FP"], ["FN", "TP"]]

    plt.figure(figsize=(5, 4))

    if style.lower() == 'seaborn':
        # Seaborn style with coolwarm colormap
        ax = sns.heatmap(matrix, annot=True, fmt="d", cmap="coolwarm",
                         xticklabels=["Actual Negative", "Actual Positive"],
                         yticklabels=["Predicted Negative", "Predicted Positive"])
    else:
        # Matplotlib style with Blues colormap
        ax = sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                         xticklabels=["Actual Negative", "Actual Positive"],
                         yticklabels=["Predicted Negative", "Predicted Positive"])

    # Overlay text labels (TN, FP, FN, TP) for clarity
    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.7, labels[i][j],
                    ha="center", va="center", color="black", fontsize=12)

    plt.title("Confusion Matrix")
    plt.xlabel("Actual Label")
    plt.ylabel("Predicted Label")
    plt.show()

if __name__ == '__main__':
    # Paths to WiC dataset files
    base_path = "C:/WiC_dataset/train"
    data_file = os.path.normpath(os.path.join(base_path, "train.data.text_files"))
    gold_file = os.path.normpath(os.path.join(base_path, "train.gold.text_files"))

    # Load data and compute similarities
    data, labels = config.load_wic_data(Path(data_file), Path(gold_file))
    similarities = similarity.compute_sentence_similarity(data)

    EvaluationMetrics = combined.evaluate(similarities, labels, data) # EvaluationMetrics should contain accuracy, correct_answers_count, y_pred


    # Ha az assert nem megfelelő, addig a confusion_matrix function sem fog lefutni
    assert len(train_y_true) == EvaluationMetrics.total_predictions
    cm = confusion_matrix(train_y_true, EvaluationMetrics.correct_predictions, labels=['T', 'F'])

    # Takes the confusion matrix (cm), flattens it into a 1D array using .ravel(), and then unpacks its values into four variables
    tp, fp, fn, tn = cm.ravel()

    print(f"Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")

    plot_confusion_matrix(tn, fp, fn, tp)  # Default seaborn style
    plot_confusion_matrix(tn, fp, fn, tp, style='matplotlib')
