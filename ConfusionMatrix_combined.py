import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

import WiCTfidfBaseline_combined
from y_true_combined import y_true_combined


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

    # Paths to all 6 WiC dataset files
    data_paths = {
        "dev": ("C:/WiC_dataset/dev/dev.data.txt", "C:/WiC_dataset/dev/dev.gold.txt"),
        "test": ("C:/WiC_dataset/test/test.data.txt", "C:/WiC_dataset/test/test.gold.txt"),
        "train": ("C:/WiC_dataset/train/train.data.txt", "C:/WiC_dataset/train/train.gold.txt"),
    }

    all_data = []
    all_labels = []
    all_similarities = []

    # Load data and compute similarities
    for dataset_name, (data_file, gold_file) in data_paths.items():
        data, labels = WiCTfidfBaseline_combined.load_wic_data(data_file, gold_file)
        similarities = WiCTfidfBaseline_combined.compute_sentence_similarity(data)

        all_data.extend(data)
        all_labels.extend(labels)
        all_similarities.extend(similarities)

    # Convert similarities to numpy array
    all_similarities = np.array(all_similarities)

    # Get the best threshold
    best_threshold = WiCTfidfBaseline_combined.optimize_threshold(all_similarities, all_labels)

    # Get predictions
    _, _, y_pred_combined = WiCTfidfBaseline_combined.evaluate_with_uncertainty(
        all_similarities, all_labels, all_data, threshold=best_threshold
    )

    # Compute confusion matrix
    cm = confusion_matrix(y_true_combined, y_pred_combined, labels=['T', 'F'])

    # Extract values
    tn, fp, fn, tp = cm.ravel()

    print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    # Plot the confusion matrix
    matplotlib_plot_confusion_matrix(tn, fp, fn, tp)


if __name__ == "__main__":
    main()