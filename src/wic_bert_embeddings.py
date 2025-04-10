import os

import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()  # Set model to evaluation mode (disables dropout)


def compute_wic_similarity(sentence1, sentence2, word_index1, word_index2, target_word):
    """
    Computes the similarity between contextualized embeddings of a target word in two different sentences.

    Args:
        sentence1 (str): The first sentence containing the target word.
        sentence2 (str): The second sentence containing the target word.
        word_index1 (int): Index (by word position) of the target word in sentence1.
        word_index2 (int): Index (by word position) of the target word in sentence2.
        target_word (str): The target word whose meaning is being compared.

    Returns:
        float: Cosine similarity score between the two word embeddings.
               If the target word cannot be located properly, returns None.
    """

    # Tokenize both sentences and retrieve token-level character spans
    tokenized_sentence1 = tokenizer(sentence1, return_offsets_mapping=True, add_special_tokens=True,
                                    return_tensors="pt")
    tokenized_sentence2 = tokenizer(sentence2, return_offsets_mapping=True, add_special_tokens=True,
                                    return_tensors="pt")

    # Identify token positions corresponding to the target word
    target_token_indices1 = [i for i, offset in enumerate(tokenized_sentence1['offset_mapping'][0]) if
                             i - 1 == word_index1]
    target_token_indices2 = [i for i, offset in enumerate(tokenized_sentence2['offset_mapping'][0]) if
                             i - 1 == word_index2]

    if not target_token_indices1 or not target_token_indices2:
        print(f"Warning: Could not find token indices for target word '{target_word}'. Skipping instance.")
        return None

    # Extract contextualized embeddings using BERT
    with torch.no_grad():
        output_sentence1 = model(**tokenized_sentence1)
        output_sentence2 = model(**tokenized_sentence2)
        embeddings_sentence1 = output_sentence1.last_hidden_state  # Shape: (batch_size, seq_len, hidden_dim)
        embeddings_sentence2 = output_sentence2.last_hidden_state

    # Compute average embedding for the target word (handles multi-token cases)
    target_embedding1 = torch.mean(embeddings_sentence1[0, target_token_indices1, :], dim=0).unsqueeze(0)
    target_embedding2 = torch.mean(embeddings_sentence2[0, target_token_indices2, :], dim=0).unsqueeze(0)

    # Compute cosine similarity between embeddings
    similarity_score = cosine_similarity(target_embedding1, target_embedding2)[0][0]

    return similarity_score


def load_wic_data(wic_data_file, gold_labels_file):
    """
    Loads WiC dataset and corresponding gold labels from files.

    Args:
        wic_data_file (str): Path to the dataset file containing sentences and indices.
        gold_labels_file (str): Path to the file containing gold-standard labels (T/F).

    Returns:
        list: List of tuples containing (word, pos, index1, index2, sentence1, sentence2).
        list: Corresponding gold-standard labels.
    """

    dataset = []
    gold_labels = []

    with open(wic_data_file, 'r', encoding='utf-8') as data_file, open(gold_labels_file, 'r',
                                                                       encoding='utf-8') as label_file:
        for data_line, label in zip(data_file, label_file):
            parts = data_line.strip().split('\t')  # Format: word, POS, indices, sentence1, sentence2

            if len(parts) != 5:
                print(f"Skipping malformed line: {data_line}")
                continue  # Skip lines that don't have the expected format

            word, pos, indices, sentence1, sentence2 = parts

            try:
                word_index1, word_index2 = map(int, indices.split('-'))  # Convert indices to integers
            except ValueError:
                print(f"Skipping line due to index format error: {data_line}")
                continue

            dataset.append((word, pos, word_index1, word_index2, sentence1, sentence2))
            gold_labels.append(label.strip())

    return dataset, gold_labels


def evaluate_wic_model(wic_dataset, gold_labels, similarity_threshold=0.7):
    """
    Evaluates the WiC model by computing accuracy against gold-standard labels.

    Args:
        wic_dataset (list): List of WiC instances (word, pos, index1, index2, sentence1, sentence2).
        gold_labels (list): List of ground-truth labels ("T" for True, "F" for False).
        similarity_threshold (float, optional): Threshold for similarity score classification.

    Returns:
        dict: Evaluation results including accuracy and average similarity score.
    """

    correct_predictions = 0
    similarity_scores = []

    for (word, pos, index1, index2, sentence1, sentence2), gold_label in zip(wic_dataset, gold_labels):
        similarity_score = compute_wic_similarity(sentence1, sentence2, index1, index2, word)

        if similarity_score is None:
            continue  # Skip if similarity could not be computed

        similarity_scores.append(similarity_score)

        # Convert similarity score to binary classification (same meaning: True/False)
        predicted_label = "T" if similarity_score > similarity_threshold else "F"

        if predicted_label == gold_label:
            correct_predictions += 1

    accuracy = correct_predictions / len(gold_labels) if gold_labels else 0
    avg_similarity = np.mean(similarity_scores) if similarity_scores else 0

    return {"accuracy": accuracy, "average_similarity": avg_similarity}


if __name__ == '__main__':

    # Define which dataset you want to work with
    actual_working_dataset = 'test'
    # Load dataset and gold labels
    base_path = f"C:/WiC_dataset/{actual_working_dataset}"
    data_file = os.path.normpath(os.path.join(base_path, f"{actual_working_dataset}.data.txt"))
    gold_file = os.path.normpath(os.path.join(base_path, f"{actual_working_dataset}.gold.txt"))

    wic_dataset, gold_labels = load_wic_data(data_file, gold_file)

    # Evaluate model performance
    evaluation_results = evaluate_wic_model(wic_dataset, gold_labels)

    print("Evaluation Results:")
    print(evaluation_results)

    # Attempt to evaluate on a development set if available
    try:
        dev_data_file = 'dev.txt'
        dev_dataset, dev_gold_labels = load_wic_data(dev_data_file, gold_file)

        print(f"Loaded {len(dev_dataset)} instances from {dev_data_file}")
        dev_results = evaluate_wic_model(dev_dataset, dev_gold_labels)

        print("Evaluation Results on Dev Set:")
        print(dev_results)

    except FileNotFoundError:
        print("Warning: Development dataset not found. Please ensure 'dev.txt' is available in the working directory.")
        print("You can download the WiC dataset from: https://pilehvar.github.io/wic/")
