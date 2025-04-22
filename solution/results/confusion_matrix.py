# C:\PycharmProjects\Peternity\solution\confusion_matrix.py
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

import similarity
import combined
from independent_scripts.y_true.y_true_train import y_true_train


def matplotlib_plot_confusion_matrix(tn, fp, fn, tp):
    """Plots a labeled confusion matrix using Matplotlib and Seaborn."""
    matrix = np.array([[tn, fp], [fn, tp]])  # Corrected order
    labels = [["TN", "FP"], ["FN", "TP"]]

    plt.figure(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Actual Negative", "Actual Positive"],
                yticklabels=["Predicted Negative", "Predicted Positive"])

    # Overlay text labels (TN, FP, FN, TP) for clarity
    for i in range(2):
        for j in range(2):
            plt.text(j + 0.5, i + 0.7, labels[i][j], ha="center", va="center", color="black", fontsize=12)

    plt.title("Confusion Matrix")
    plt.xlabel("Actual Label")
    plt.ylabel("Predicted Label")
    plt.show()


def seaborn_plot_confusion_matrix(tn, fp, fn, tp):
    """Plots a labeled confusion matrix using Seaborn."""
    matrix = np.array([[tn, fp], [fn, tp]])  # Corrected order
    labels = np.array([["TN", "FP"], ["FN", "TP"]])

    plt.figure(figsize=(5, 4))
    ax = sns.heatmap(matrix, annot=True, fmt="d", cmap="coolwarm",
                     xticklabels=["Actual Negative", "Actual Positive"],
                     yticklabels=["Predicted Negative", "Predicted Positive"])

    # Overlay text labels (TN, FP, FN, TP) for clarity
    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.7, labels[i][j], ha="center", va="center", color="black", fontsize=12)

    plt.title("Confusion Matrix")
    plt.xlabel("Actual Label")
    plt.ylabel("Predicted Label")
    plt.show()


if __name__ == '__main__':
    # Paths to WiC dataset files
    base_path = "C:/WiC_dataset/train"
    data_file = os.path.normpath(os.path.join(base_path, "train.data.txt"))
    gold_file = os.path.normpath(os.path.join(base_path, "train.gold.txt"))

    # Load data and compute similarities
    data, labels = config.load_wic_data(data_file, gold_file)
    similarities = similarity.compute_sentence_similarity(data)

    EvaluationMetrics = combined.evaluate(similarities, labels, data) # EvaluationMetrics should contain accuracy, correct_answers_count, y_pred


    # Ha az assert nem megfelelő, addig a confusion_matrix function sem fog lefutni
    assert len(y_true_train) == EvaluationMetrics.total_predictions
    cm = confusion_matrix(y_true_train, EvaluationMetrics.correct_predictions, labels=['T', 'F'])

    # Takes the confusion matrix (cm), flattens it into a 1D array using .ravel(), and then unpacks its values into four variables
    tp, fp, fn, tn = cm.ravel()

    print(f"Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")

    matplotlib_plot_confusion_matrix(tn, fp, fn, tp)
    seaborn_plot_confusion_matrix(tn, fp, fn, tp)
