# C:\PycharmProjects\Peternity\src\utils\wic_compare_gold_to_predicted_yes_or_no.py
import os

import src.utils.wic_data_loader
from src.PATH import BASE_PATH


def wic_compare_gold_to_predicted_yes_or_no(gold_str, predicted_str):
    # Normalize inputs and split by lines
    gold = [word.strip().replace('T', 'Yes').replace('F', 'No') for word in gold_str.strip().split('\n')]
    predicted = [word.strip().replace('T', 'Yes').replace('F', 'No') for word in predicted_str.strip().split('\n')]

    # Ensure both lists have the same length
    if len(gold) != len(predicted):
        raise ValueError("Input strings must have the same number of entries.")

    # Initialize counts
    tp = fp = fn = tn = 0

    # Calculate metrics
    for g, p in zip(gold, predicted):
        if g == 'Yes' and p == 'Yes':
            tp += 1
        elif g == 'No' and p == 'Yes':
            fp += 1
        elif g == 'Yes' and p == 'No':
            fn += 1
        elif g == 'No' and p == 'No':
            tn += 1

    return tp, fp, fn, tn

def get_results():
    # TODO replace with data from {actual_working_file}.gold.txt and the {actual_model}_output.txt

    base_path = BASE_PATH
    data_path = os.path.normpath(os.path.join(base_path, "test/test.data.txt"))
    print(data_path)
    gold_path = os.path.normpath(os.path.join(base_path, "test/test.gold.txt"))
    print(gold_path)
    gold_data, predicted_data = src.utils.wic_data_loader.load_wic_data(data_path, gold_path)

    return wic_compare_gold_to_predicted_yes_or_no(gold_data, predicted_data)

# Example usage
if __name__ == '__main__':
    results = get_results
    print(f"TP: {results.__str__()}, FP: {results}, FN: {results}, TN: {results}")
