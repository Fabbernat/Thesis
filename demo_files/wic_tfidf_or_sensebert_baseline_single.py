# C:\PycharmProjects\Peternity\demo_files\wic_tfidf_or_sensebert_baseline_single.py

import collections
import os
import time
from typing import Any, LiteralString, Sized

import numpy as np
from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from torch import Tensor

from modules_and_data.modules.PATH import BASE_PATH

# Download necessary NLTK resources (uncomment if needed)
# import nltk
# nltk.download("wordnet")
# nltk.download("omw-1.4")
# nltk.download("punkt")

# Load sentence embedding model
SENTENCE_EMBEDDING_MODEL = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def get_best_sense(word, sentence):
    """Uses sentence embeddings to disambiguate word senses using both definition and example sentences."""
    synsets: object = wn.synsets(word)
    if not synsets:
        return None

    sentence_embedding: Tensor = SENTENCE_EMBEDDING_MODEL.encode(sentence, convert_to_tensor=True)
    best_sense = max(synsets, key=lambda sense:
    util.pytorch_cos_sim(sentence_embedding,
                         SENTENCE_EMBEDDING_MODEL.encode(sense.definition(), convert_to_tensor=True)).item())

    return best_sense


def get_disambiguated_synonyms(word: object, sentence: object) -> set[Any]:
    """
        Gets synonyms only for the most relevant sense of the word.
        Uses advanced Word Sense Disambiguation to get only relevant synonyms for a word in context.
        :rtype: object
    """
    sense: object | None | Any = get_best_sense(word, sentence)
    if sense:
        return {lemma.name().replace("_", " ") for lemma in sense.lemmas()}  # Return synonyms for that sense only
    return set()


def optimize_threshold(similarities: object, labels: Sized) -> float:
    """Finds the best similarity threshold for classification."""
    from sklearn.metrics import precision_recall_curve

    precisions, recalls, thresholds = precision_recall_curve(
        [1 if label == 'T' else 0 for label in labels],
        similarities
    )

    # Find threshold that maximizes F1 score
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx]


def expand_sentence_with_wsd(sentence, target_word: object) -> LiteralString:
    """Expands a sentence by adding only contextually relevant synonyms."""
    words: collections.Iterable = sentence.split()
    expanded_words: list[Any] = []

    for word in words:
        if word == target_word:  # Only expand the target word
            synonyms: set[Any] = get_disambiguated_synonyms(word, sentence)
            expanded_words.append(word + " " + " ".join(synonyms) if synonyms else word)
        else:
            expanded_words.append(word)

    return " ".join(expanded_words)


def load_wic_data(data_path: object, gold_path: object) -> tuple[list[tuple[str, str, int, int, str, str]], list[str]]:
    """
        Loads the WiC dataset and its gold into a structured format.
        extracts index1 and index2 from the index field.
        :rtype: object
        :param data_path:
        :param gold_path:
        :return:
     """

    data: list[tuple[str, str, int, int, str, str]] = []
    gold: list[str] = []

    with open(data_path, 'r', encoding='utf-8') as f_data, open(gold_path, 'r', encoding='utf-8') as f_gold:
        for line, label in zip(f_data, f_gold):
            parts: list[str] = line.strip().split('\t')  # Word, POS, index, sentence1, sentence2
            word, pos, index, sentence_a, sentence_b = parts

            '''
                This parser try-catch tries to convert  index1 and index2, the two, likely string variables to integers
                each sentence can be as long as 99 words.
                Their format:

                1-1
                6-7
                0-5
                14-8
                etc...
            '''
            try:
                index1, index2 = map(int, index.split('-'))  # Extract integer indices
            except ValueError:
                continue  # Skip lines with incorrect index format

            # Expand sentences with synonyms
            # sentence_a = NltkHandler.expand_with_synonyms(sentence_a)
            # sentence_b = NltkHandler.expand_with_synonyms(sentence_b)

            # Highlight target word for better feature extraction
            # sentence_a = sentence_a.replace(word, word + " " + word)
            # sentence_b = sentence_b.replace(word, word + " " + word)

            # sentence_a = expand_sentence_with_wsd(sentence_a, word)
            # sentence_b = expand_sentence_with_wsd(sentence_b, word)

            # sentence_a = normalize_sentence(sentence_a)
            # sentence_b = normalize_sentence(sentence_b)

            gold.append(label.strip())
            data.append((word, pos, index1, index2, sentence_a, sentence_b))

    return data, gold


def compute_sentence_similarity(data, mode="hybrid"):
    """Hybrid approach combining TF-IDF and BERT"""
    if mode == "hybrid":
        # TF-IDF features
        tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),  # Include bigrams
            max_features=10000,
            sublinear_tf=True
        )

        # Process all sentences at once for better efficiency
        sentence_pairs = [(s1, s2) for _, _, _, _, s1, s2 in data]
        all_sentences = [s for pair in sentence_pairs for s in pair]
        tfidf_vectorizer.fit(all_sentences)

        similarities = []
        for sentence1, sentence2 in sentence_pairs:
            # TF-IDF similarity
            tfidf_vecs = tfidf_vectorizer.transform([sentence1, sentence2])
            tfidf_sim = cosine_similarity(tfidf_vecs[0], tfidf_vecs[1])[0][0]

            # BERT similarity
            bert_embs = SENTENCE_EMBEDDING_MODEL.encode([sentence1, sentence2], convert_to_tensor=True)
            bert_sim = util.pytorch_cos_sim(bert_embs[0], bert_embs[1]).item()

            # Weighted combination
            similarities.append(0.4 * tfidf_sim + 0.6 * bert_sim)

        return np.array(similarities)

    elif mode == "bert":
        similarities = []
        for _, _, _, _, sentence1, sentence2 in data:
            bert_embs = SENTENCE_EMBEDDING_MODEL.encode([sentence1, sentence2], convert_to_tensor=True)
            sim = util.pytorch_cos_sim(bert_embs[0], bert_embs[1]).item()
            similarities.append(sim)
        return np.array(similarities)

    return None


def preprocess_sentences(data: list) -> list:
    """Expand sentences with WSD-based synonyms for the target word."""
    processed_data = []
    for word, pos, idx1, idx2, sent1, sent2 in data:
        # Expand only the target word with its contextually relevant synonyms
        expanded_sent1 = expand_sentence_with_wsd(sent1, word)
        expanded_sent2 = expand_sentence_with_wsd(sent2, word)
        processed_data.append((word, pos, idx1, idx2, expanded_sent1, expanded_sent2))
    return processed_data


def evaluate_with_uncertainty(similarities, labels, data, threshold=0.449, gray_zone=(0.40, 0.50), verbose=False) -> tuple[float, int, list[Any]]:
    """Evaluates accuracy, adding a gray zone for uncertain cases."""
    predictions = []
    uncertain_cases = 0

    for sim in similarities:
        if sim > threshold:
            predictions.append('T')
        elif gray_zone[0] <= sim <= gray_zone[1]:
            predictions.append('U')  # Uncertain
            uncertain_cases += 1
        else:
            predictions.append('F')

    correct_predictions = sum(pred == true_label or pred == 'U' for pred, true_label in zip(predictions, labels))
    accuracy = correct_predictions / len(labels)

    if verbose:
        # Print False Positives (predicted True when actually False)
        print_evaluation_details(predictions, labels, similarities, data, "False Positives", 'T', 'F')
        # Print False Negatives (predicted False when actually True)
        print_evaluation_details(predictions, labels, similarities, data, "False Negatives", 'F', 'T')

    print(f"Uncertain cases: {uncertain_cases}/{len(labels)} ({(uncertain_cases / len(labels)):.2%})")
    return accuracy, correct_predictions, predictions


def print_evaluation_details(predictions, labels, similarities, data, title, predicted_label, true_label):
    """Helper function to print evaluation details."""
    print(f"\n{title}:")
    for i, (pred, true, sim, (word, _, _, _, sentence_a, sentence_b)) in enumerate(
            zip(predictions, labels, similarities, data)):
        if pred == predicted_label and true == true_label:
            print(f"Index {i}: Similarity = {sim:.3f}")
            print(f"Word: {word}")
            print(f"Sentence A: {sentence_a}")
            print(f"Sentence B: {sentence_b}")
            print("-" * 80)


def evaluate(similarities, labels, data, threshold=0.449, return_predictions=False, verbose=False) -> tuple[float, int, list[str]] | tuple[float, int]:
    """
            Evaluates accuracy based on `similarity threshold`.
            :param similarities:
            :param labels:
            :param data:
            :param threshold:
            :param return_predictions:
            :param verbose: If verbose=True, prints false positives, true negatives, and relevant sentences.
            :return:
        """
    print("predicting...")
    predictions = ['T' if sim > threshold else 'F' for sim in similarities]
    print("counting correct predictions...")
    correct_predictions_count = sum(pred == true_label for pred, true_label in zip(predictions, labels))
    accuracy = correct_predictions_count / len(labels)

    if verbose:
        # Print False Positives (predicted True when actually False)
        print_evaluation_details(predictions, labels, similarities, data, "False Positives", 'T', 'F')
        # Print False Negatives (predicted False when actually True)
        print_evaluation_details(predictions, labels, similarities, data, "False Negatives", 'F', 'T')

    if return_predictions:
        return accuracy, correct_predictions_count, predictions
    return accuracy, correct_predictions_count


def main():
    start_time = time.time()

    # === CONFIGURATION SECTION ===

    # Define which dataset you want to work with
    actual_working_dataset = 'train'
    use_processed_data = True
    use_bert = True
    use_best_threshold = False # This does not work yet, leave on False
    use_evaluate_with_uncertainty = False
    verbose = True
    # =============================

    # Paths to WiC dataset files
    data_file = os.path.normpath(BASE_PATH + rf'\{actual_working_dataset}\{actual_working_dataset}.data.txt')
    gold_file = os.path.normpath(BASE_PATH + rf'\{actual_working_dataset}\{actual_working_dataset}.gold.txt')

    # Load data and labels
    data, labels = load_wic_data(data_file, gold_file)

    # Optionally preprocess data
    processed_data = preprocess_sentences(data) if use_processed_data else data

    # Compute similarities
    mode = "bert" if use_bert else "tfidf"
    similarities = compute_sentence_similarity(processed_data, mode=mode)

    # Threshold selection
    best_threshold_value = optimize_threshold(similarities, labels)
    threshold = best_threshold_value if use_best_threshold else 0.449
    print(f"Using threshold: {threshold:.3f}")

    # Evaluate model
    eval_fn = evaluate_with_uncertainty if use_evaluate_with_uncertainty else evaluate
    if use_evaluate_with_uncertainty:
        accuracy, correct_answers_count, predictions = eval_fn(similarities, labels, data, verbose=verbose,
                                                               threshold=threshold)
    else:
        accuracy, correct_answers_count = eval_fn(similarities, labels, data, verbose=verbose, threshold=threshold)


    print(f"Baseline accuracy: {accuracy:.3%}")
    print(f"{correct_answers_count} correct answer(s) out of {len(labels)} answers.")

    runtime = time.time() - start_time
    print(f"\nTotal runtime: {runtime:.2f} seconds")


if __name__ == '__main__':
    main()
