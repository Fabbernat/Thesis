import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_wic_data(data_path, gold_path):
    """
        Loads the WiC dataset and its gold into a structured format.
        extracts index1 and index2 from the index field.
     """

    data = []
    gold = []

    with open(data_path, 'r', encoding='utf-8') as f_data, open(gold_path, 'r', encoding='utf-8') as f_gold:
        for line, label in zip(f_data, f_gold):
            parts = line.strip().split('\t')  # Word, POS, index, sentence1, sentence2
            word, pos, index, sentence_a, sentence_b = parts
            try:
                index1, index2 = map(int, index.split('-'))  # Extract integer indices
            except ValueError:
                continue  # Skip lines with incorrect index format

            '''
                make a python parser program that gets a variable and returns 2 integers: index1 and index2
                each sentence can be as long as 99 words.
                Their format:
                
                1-1
                6-7
                0-5
                14-8
                etc...
            '''
            gold.append(label.strip())
            data.append((word, pos, index1, index2, sentence_a, sentence_b))

    return data, gold


def compute_similarity(data):
    """Computes cosine similarity between sentence pairs using TF-IDF."""
    vectorizer = TfidfVectorizer(lowercase=True)
    similarities = []

    for word, pos, index1, index2, sentence1, sentence2 in data:
        vectors = vectorizer.fit_transform([sentence1, sentence2])
        sim = cosine_similarity(vectors[0], vectors[1])[0][0]
        similarities.append(sim)

    return np.array(similarities)


def evaluate(similarities, labels, threshold=0.5):
    """Evaluates accuracy based on a threshold for similarity."""
    predictions = ['T' if sim > threshold else 'F' for sim in similarities]
    correct_answers_count = sum(pred == true_label for pred, true_label in zip(predictions, labels))
    accuracy = correct_answers_count / len(labels)
    return accuracy, correct_answers_count


if __name__ == "__main__":
    # Paths to WiC dataset files
    base_path = "C:/WiC_dataset/train"
    data_file = os.path.normpath(os.path.join(base_path, "train.data.txt"))
    gold_file = os.path.normpath(os.path.join(base_path, "train.gold.txt"))

    # Load data and compute similarities
    data, labels = load_wic_data(data_file, gold_file)
    similarities = compute_similarity(data)

    # Evaluate model
    accuracy, correct_answers_count = evaluate(similarities, labels)
    print(f"Baseline accuracy: {accuracy:.3%}")
    print(f"{correct_answers_count} correct answer(s) out of {len(labels)} answers.")