import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer, util

from WiCTfidfBaseline_combined import compute_sentence_similarity

# Download necessary NLTK resources (uncomment if needed)
# import nltk
# nltk.download("wordnet")
# nltk.download("omw-1.4")
# nltk.download("punkt")

# Load sentence embedding model
SENTENCE_EMBEDDING_MODEL = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Load a sentence embedding model (e.g., all-MiniLM)


def get_best_sense(word, sentence):
    """Uses sentence embeddings to disambiguate word senses."""
    synsets = wn.synsets(word)
    if not synsets:
        return None

    sentence_embedding = SENTENCE_EMBEDDING_MODEL.encode(sentence, convert_to_tensor=True)
    best_sense = max(synsets, key=lambda sense:
        util.pytorch_cos_sim(sentence_embedding, SENTENCE_EMBEDDING_MODEL.encode(sense.definition(), convert_to_tensor=True)).item())

    return best_sense


def get_disambiguated_synonyms(word, sentence):
    """
        Gets synonyms only for the most relevant sense of the word.
        Uses advanced Word Sense Disambiguation to get only relevant synonyms for a word in context.
    """
    sense = get_best_sense(word, sentence)
    if sense:
        return {lemma.name().replace("_", " ") for lemma in sense.lemmas()}  # Return synonyms for that sense only
    return set()


def optimize_threshold(similarities, labels):
    """Finds the best similarity threshold for classification."""
    best_accuracy = 0
    best_threshold = 0.0

    for threshold in np.arange(0.3, 0.6, 0.01):  # Kipróbál értékeket 0.3 és 0.6 között
        predictions = ['T' if sim > threshold else 'F' for sim in similarities]
        accuracy = sum(pred == true_label for pred, true_label in zip(predictions, labels)) / len(labels)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    return best_threshold


def expand_sentence_with_wsd(sentence, target_word):
    """Expands a sentence by adding only contextually relevant synonyms."""
    words = sentence.split()
    expanded_words = []

    for word in words:
        if word == target_word:  # Only expand the target word
            synonyms = get_disambiguated_synonyms(word, sentence)
            expanded_words.append(word + " " + " ".join(synonyms) if synonyms else word)
        else:
            expanded_words.append(word)

    return " ".join(expanded_words)


def normalize_sentence(sentence):
    """Replaces contractions for better word sense disambiguation."""
    return sentence.replace(" 's", "'s")  # Example normalization


def normalize_negations(sentence):
    """Replaces contractions like n't with 'not' for better word sense disambiguation."""
    sentence = sentence.replace("n't", "not")
    return sentence

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


def compute_sentence_similarity(data):
    """Computes cosine similarity between sentence pairs using TF-IDF."""

    # max_df 0.1-0.9 does not change much
    vectorizer = TfidfVectorizer(lowercase=True,
                                 ngram_range=(0, 1),
                                 max_df=0.85,
                                 min_df=2,
                                 sublinear_tf=True,
                                 norm='l2')

    # Precompute vocabulary using all sentences
    all_sentences = [sentence for _, _, _, _, sentence_a, sentence_b in data for sentence in (sentence_a, sentence_b)]
    vectorizer.fit(all_sentences)

    similarities = []
    for _, _, _, _, sentence1, sentence2 in data:
        vectors = vectorizer.transform([sentence1, sentence2])
        sim = cosine_similarity(vectors[0], vectors[1])[0][0]
        similarities.append(sim)

    return np.array(similarities)


def evaluate_with_uncertainty(similarities, labels, data, threshold=0.449, gray_zone=(0.40, 0.50), verbose=False):
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
        print_evaluation_details(predictions, labels, similarities, data, "False Positives", 'T', 'F')
        print_evaluation_details(predictions, labels, similarities, data, "True Negatives", 'F', 'F')

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


def evaluate(similarities, labels, data, threshold=0.449, return_predictions=False, verbose=False):
        """
            Evaluates accuracy based on similarity threshold.
            :param similarities:
            :param labels:
            :param data:
            :param threshold:
            :param return_predictions:
            :param verbose: If verbose=True, prints false positives, true negatives, and relevant sentences.
            :return:
        """
        predictions = ['T' if sim > threshold else 'F' for sim in similarities]
        correct_predictions_count = sum(pred == true_label for pred, true_label in zip(predictions, labels))
        accuracy = correct_predictions_count / len(labels)

        if verbose:
            print_evaluation_details(predictions, labels, similarities, data, "False Positives", 'T', 'F')
            print_evaluation_details(predictions, labels, similarities, data, "True Negatives", 'F', 'F')

        if return_predictions:
            return accuracy, correct_predictions_count, predictions
        return accuracy, correct_predictions_count


def main():
    # Paths to WiC dataset files
    base_path = "C:/WiC_dataset/train"
    data_file = os.path.normpath(os.path.join(base_path, "train.data.txt"))
    gold_file = os.path.normpath(os.path.join(base_path, "train.gold.txt"))


    # Load data and compute similarities
    data, labels = load_wic_data(data_file, gold_file)
    similarities = compute_sentence_similarity(data)

    # Evaluate model
    accuracy, correct_answers_count = evaluate(similarities, labels, data, verbose=True)
    print(f"Baseline accuracy: {accuracy:.3%}")
    print(f"{correct_answers_count} correct answer(s) out of {len(labels)} answers.")

if __name__ == "__main__":
    main()