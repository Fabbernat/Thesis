import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

import WiCTfidfBaseline
from dev_y_true import dev_y_true


def plot_confusion_matrix(tn, fp, fn, tp):
    """
        Plots a confusion matrix.
        tn = True Negative
        tp = True Positive
        fn = False Negative
        fp = False Positive
    """
    matrix = np.array([[tp, fp], [fn, tn]])
    labels = ["TP", "FP", "FN", "TN"]

    plt.figure(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Actual 1", "Actual 0"],
                yticklabels=["Pred 1", "Pred 0"])
    plt.title("Confusion Matrix")
    plt.xlabel("Actual Label")
    plt.ylabel("Predicted Label")
    plt.show()

def seaborn_plot_confusion_matrix():
    sns.heatmap(cm,
                annot=True,
                fmt='g',
                xticklabels=['Dog', 'Not Dog'],
                yticklabels=['Dog', 'Not Dog'])
    plt.ylabel('Actual', fontsize=13)
    plt.title('Confusion Matrix', fontsize=17, pad=20)
    plt.gca().xaxis.set_label_position('top')
    plt.xlabel('Prediction', fontsize=13)
    plt.gca().xaxis.tick_top()

    plt.gca().figure.subplots_adjust(bottom=0.2)
    plt.gca().figure.text(0.5, 0.05, 'Prediction', ha='center', fontsize=13)
    plt.show()

if __name__ == "__main__":
    # Paths to WiC dataset files
    base_path = "C:/WiC_dataset/dev"
    data_file = os.path.normpath(os.path.join(base_path, "dev.data.txt"))
    gold_file = os.path.normpath(os.path.join(base_path, "dev.gold.txt"))

    # Load data and compute similarities
    data, labels = WiCTfidfBaseline.load_wic_data(data_file, gold_file)
    similarities = WiCTfidfBaseline.compute_similarity(data)

    # Evaluate model and get predictions
    accuracy, correct_answers_count, y_pred = WiCTfidfBaseline.evaluate(similarities, labels, return_predictions=True)

    # Confusion matrix calculation

    # Ha az assert nem megfelelő, addig a confusion_matrix function sem fog lefutni
    assert len(dev_y_true) == len(y_pred)
    cm = confusion_matrix(dev_y_true, y_pred, labels=['T', 'F'])
    tp, fp, fn, tn = cm.ravel()

    print(f"Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")


    plot_confusion_matrix(tn, fp, fn, tp)
