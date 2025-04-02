import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from modules_and_data.wic_data_loader import load_wic_data
from wic_tfidf_baseline_combined import compute_sentence_similarity, evaluate
from y_true_test import test_y_true


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


def main():
    # Paths to WiC dataset files
    base_path = "C:/WiC_dataset/test"
    data_file = os.path.normpath(os.path.join(base_path, "test.data.txt"))
    gold_file = os.path.normpath(os.path.join(base_path, "test.gold.txt"))

    # Load data and compute similarities
    data, labels = load_wic_data(data_file, gold_file)
    similarities = compute_sentence_similarity(data)

    # Evaluate model and get predictions
    accuracy, correct_answers_count, y_pred = evaluate(similarities, labels, data, return_predictions=True)

    # Confusion matrix calculation

    # Ha az assert nem megfelelő, addig a confusion_matrix function sem fog lefutni
    assert len(test_y_true) == len(y_pred)
    cm = confusion_matrix(test_y_true, y_pred, labels=['T', 'F'])

    # Takes the confusion matrix (cm), flattens it into a 1D array using .ravel(), and then unpacks its values into four variables
    tp, fp, fn, tn = cm.ravel()

    print(f"Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")

    matplotlib_plot_confusion_matrix(tn, fp, fn, tp)
    seaborn_plot_confusion_matrix(tn, fp, fn, tp)

if __name__ == "__main__":
    main()