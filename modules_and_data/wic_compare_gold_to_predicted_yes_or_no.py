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


# Example usage
if __name__ == "__main__":
    gold_data = """Yes
                   No
                   T
                   F
                   Yes"""
    predicted_data = """Yes
                        Yes
                        No
                        F
                        No"""

    results = wic_compare_gold_to_predicted_yes_or_no(gold_data, predicted_data)
    print(f"TP: {results[0]}, FP: {results[1]}, FN: {results[2]}, TN: {results[3]}")
