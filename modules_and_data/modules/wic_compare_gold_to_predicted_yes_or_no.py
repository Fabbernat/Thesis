# C:\PycharmProjects\Peternity\modules_and_data\modules\wic_compare_gold_to_predicted_yes_or_no.py
import os

import modules_and_data.modules.wic_data_loader

# Define which dataset you want to work with
actual_working_dataset = 'test'

def wic_compare_gold_to_predicted_yes_or_no(gold_list, predicted_list):
    # Normalize inputs and split by lines
    gold = [label.strip().replace('T', 'Yes').replace('F', 'No') for label in gold_list]
    predicted = [label.strip().replace('T', 'Yes').replace('F', 'No') for label in predicted_list]

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

    base_path = '../../WiC_dataset/'
    data_path = os.path.normpath(os.path.join(base_path, "test/test.data.txt"))
    gold_path = os.path.normpath(os.path.join(base_path, "test/test.gold.txt"))

    print(data_path)
    print(gold_path)

    data, gold = modules_and_data.modules.wic_data_loader.load_wic_data(data_path, gold_path)

    predicted_path = os.path.normpath(os.path.join(base_path, f"{actual_working_dataset}/{actual_working_dataset}.gold.txt"))
    with open(predicted_path, 'r', encoding='utf-8') as f:
        predicted = [line.strip() for line in f.readlines()]

    return wic_compare_gold_to_predicted_yes_or_no(gold, predicted)

# Example usage
if __name__ == '__main__':
    results = get_results()
    print(f"TP: {results.__str__()}, FP: {results}, FN: {results}, TN: {results}")
