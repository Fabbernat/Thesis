import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_wic_data(data_path, labels_path):
    """Loads the WiC dataset and its labels into a structured format."""
    data = []
    labels = []

    with open(data_path, 'r', encoding='utf-8') as f_data, open(labels_path, 'r', encoding='utf-8') as f_labels:
        for line, label in zip(f_data, f_labels):
            parts = line.strip().split('\t')  # Word, POS, index, sentence1, sentence2
            word, pos, index, sentence1, sentence2 = parts
            labels.append(label.strip())
            data.append((word, sentence1, sentence2))

    return data, labels


def compute_similarity(data):
    """Computes cosine similarity between sentence pairs using TF-IDF."""
    vectorizer = TfidfVectorizer()
    similarities = []

    for word, sentence1, sentence2 in data:
        vectors = vectorizer.fit_transform([sentence1, sentence2])
        sim = cosine_similarity(vectors[0], vectors[1])[0][0]
        similarities.append(sim)

    return np.array(similarities)


def evaluate(similarities, labels, threshold=0.5):
    """Evaluates accuracy based on a threshold for similarity."""
    predictions = ['YES' if sim > threshold else 'NO' for sim in similarities]
    accuracy = np.mean([pred == true_label for pred, true_label in zip(predictions, labels)])
    return accuracy


if __name__ == "__main__":
    # Paths to WiC dataset files
    base_path = "C:/Users/Bernát/Downloads/WiC_dataset/train"
    data_file = os.path.join(base_path, "train.data.txt")
    labels_file = os.path.join(base_path, "train.gold.txt")

    # Load data and compute similarities
    data, labels = load_wic_data(data_file, labels_file)
    similarities = compute_similarity(data)

    # Evaluate model
    accuracy = evaluate(similarities, labels)
    print(f"Baseline accuracy: {accuracy:.2%}")
