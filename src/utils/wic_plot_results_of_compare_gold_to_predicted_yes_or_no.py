# C:\PycharmProjects\Peternity\src\utils\wic_plot_results_of_compare_gold_to_predicted_yes_or_no.py
import matplotlib.pyplot as plt
import seaborn as sns
from wic_compare_gold_to_predicted_yes_or_no import get_results

def plot_wic_results(tp, fp, fn, tn):
    # Data for plotting
    categories = ['True Positive (TP)', 'False Positive (FP)', 'False Negative (FN)', 'True Negative (TN)']
    values = [tp, fp, fn, tn]

    # Seaborn style setup
    sns.set_theme(style="whitegrid")

    # Create the bar plot
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x=categories, y=values, palette='viridis')

    # Add value labels
    for i, v in enumerate(values):
        ax.text(i, v + 0.1, str(v), ha='center')

    # Plot styling
    plt.title('WiC Model Results')
    plt.ylabel('Count')
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    # Show the plot
    plt.show()

# Example usage
if __name__ == '__main__':
    results = get_results()
    tp, fp, fn, tn = results[0], results[1], results[2], results[3]
    plot_wic_results(tp, fp, fn, tn)


