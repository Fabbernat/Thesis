import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

import WiCTfidfBaseline_combined
from y_true_combined import y_true_combined

QUICK_EVALUATE = False

def matplotlib_plot_confusion_matrix(tn, fp, fn, tp):
    """Plots a labeled confusion matrix using Matplotlib and Seaborn."""
    # Confusion matrix data
    matrix = np.array([[tn, fp], [fn, tp]])
    labels = np.array([["TN", "FP"], ["FN", "TP"]])

    # Create a larger figure
    plt.figure(figsize=(6, 5))

    # Use a visually appealing colormap
    ax = sns.heatmap(matrix, annot=True, fmt="d", cmap="coolwarm",
                     xticklabels=["Actual Negative", "Actual Positive"],
                     yticklabels=["Predicted Negative", "Predicted Positive"],
                     linewidths=2, linecolor="black", square=True, cbar=True,
                     annot_kws={"size": 14, "weight": "bold", "color": "white"})

    # Overlay text labels (TN, FP, FN, TP)
    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.7, labels[i][j], ha="center", va="center",
                    fontsize=12, weight="bold", color="black")

    # Improve overall styling
    plt.title("Confusion Matrix", fontsize=16, fontweight="bold")
    plt.xlabel("Actual Label", fontsize=14, fontweight="bold")
    plt.ylabel("Predicted Label", fontsize=14, fontweight="bold")

    # Show the plot
    plt.show()

def main():
    """Main function to compute confusion matrix and plot it."""

    if QUICK_EVALUATE:
        # region normal evaluate

        # Paths to all 6 WiC dataset files
        data_paths = {
            "dev": ("C:/WiC_dataset/dev/dev.data.txt", "C:/WiC_dataset/dev/dev.gold.txt"),
            "test": ("C:/WiC_dataset/test/test.data.txt", "C:/WiC_dataset/test/test.gold.txt"),
            "train": ("C:/WiC_dataset/train/train.data.txt", "C:/WiC_dataset/train/train.gold.txt"),
        }

        all_data = []
        all_labels = []
        all_similarities = []

        for dataset_name, (data_file, gold_file) in data_paths.items():
            data, labels = WiCTfidfBaseline_combined.load_wic_data(data_file, gold_file)
            similarities = WiCTfidfBaseline_combined.compute_sentence_similarity(data)

            all_data.extend(data)
            all_labels.extend(labels)
            all_similarities.extend(similarities)

        # Evaluate model and get predictions
        accuracy, correct_answers_count, y_pred = WiCTfidfBaseline_combined.evaluate(data=all_data,
                                                                                     similarities=all_similarities,
                                                                                     labels=all_labels,
                                                                                     return_predictions=True,
                                                                                     verbose=False)

        # Confusion matrix calculation

        # Ha az assert nem megfelelő, addig a confusion_matrix function sem fog lefutni
        assert len(y_true_combined) == len(y_pred)
        cm = confusion_matrix(y_true_combined, y_pred, labels=['T', 'F'])

        print('shape: ', cm.shape)  # Should be (2,2)
        # Takes the confusion matrix (cm), flattens it into a 1D array using .ravel(), and then unpacks its values into four variables
        tp, fp, fn, tn = cm.ravel()

        print(f"Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")

        matplotlib_plot_confusion_matrix(tn, fp, fn, tp)

        # endregion
    else:
        #region uncertain evaluate
        data_paths = {
            "dev": ("C:/WiC_dataset/dev/dev.data.txt", "C:/WiC_dataset/dev/dev.gold.txt"),
            "test": ("C:/WiC_dataset/test/test.data.txt", "C:/WiC_dataset/test/test.gold.txt"),
            "train": ("C:/WiC_dataset/train/train.data.txt", "C:/WiC_dataset/train/train.gold.txt"),
        }

        all_data = []
        all_labels = []
        all_similarities = []

        for dataset_name, (data_file, gold_file) in data_paths.items():
            data, labels = WiCTfidfBaseline_combined.load_wic_data(data_file, gold_file)
            similarities = WiCTfidfBaseline_combined.compute_sentence_similarity(data)

            all_data.extend(data)
            all_labels.extend(labels)
            all_similarities.extend(similarities)

        # Evaluate model and get predictions
        accuracy, correct_answers_count, y_pred = WiCTfidfBaseline_combined.evaluate_with_uncertainty(data=all_data,
                                                                                     similarities=all_similarities,
                                                                                     labels=all_labels,
                                                                                     verbose=False)

        # Confusion matrix calculation

        # Ha az assert nem megfelelő, addig a confusion_matrix function sem fog lefutni
        assert len(y_true_combined) == len(y_pred)
        cm = confusion_matrix(y_true_combined, y_pred, labels=['T', 'F'])

        print('shape: ', cm.shape)  # Should be (2,2)
        # Takes the confusion matrix (cm), flattens it into a 1D array using .ravel(), and then unpacks its values into four variables
        tp, fp, fn, tn = cm.ravel()

        print(f"Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")

        matplotlib_plot_confusion_matrix(tn, fp, fn, tp)
        #endregion



if __name__ == "__main__":
    main()