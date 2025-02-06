from nltk.wsd import lesk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
import nltk

# nltk.download("wordnet")
# nltk.download("omw-1.4")
# nltk.download("punkt")


def get_disambiguated_synonyms(word, sentence):
    """Uses Word Sense Disambiguation (WSD) to get only relevant synonyms for a word in context."""
    sense = lesk(word_tokenize(sentence), word)  # Get the best sense for the word in this sentence
    if sense:
        return {lemma.name().replace("_", " ") for lemma in sense.lemmas()}  # Return synonyms for that sense only
    return set()


def expand_sentence_with_wsd(sentence, target_word):
    """Expands a sentence by adding only contextually relevant synonyms."""
    words = sentence.split()
    expanded_words = []

    for word in words:
        if word == target_word:  # Only expand the target word
            synonyms = get_disambiguated_synonyms(word, sentence)
            if synonyms:
                expanded_words.append(word + " " + " ".join(synonyms))
            else:
                expanded_words.append(word)
        else:
            expanded_words.append(word)

    return " ".join(expanded_words)


import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# import NltkHandler

def normalize_negations(sentence):
    """Replaces contractions like n't with 'not' for better word sense disambiguation."""
    sentence = sentence.replace("n't", " not")  # Convert "didn't" to "did not"
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

            # sentence_a = normalize_negations(sentence_a)
            # sentence_b = normalize_negations(sentence_b)

            gold.append(label.strip())
            data.append((word, pos, index1, index2, sentence_a, sentence_b))

    return data, gold


def compute_similarity(data):
    """Computes cosine similarity between sentence pairs using TF-IDF."""

    # vectorizer: optional configs: stop_words="english"
    # max_df 0.1-0.9 does not change much
    vectorizer = TfidfVectorizer(lowercase=True,
                                 ngram_range=(1, 3),
                                 max_df=0.9, min_df=2,
                                 sublinear_tf=True, norm='l2')

    # Precompute vocabulary using all sentences
    all_sentences = [sentence for _, _, _, _, sentence_a, sentence_b in data for sentence in (sentence_a, sentence_b)]
    vectorizer.fit(all_sentences)

    similarities = []
    for word, pos, index1, index2, sentence1, sentence2 in data:
        vectors = vectorizer.transform([sentence1, sentence2])
        sim = cosine_similarity(vectors[0], vectors[1])[0][0]
        similarities.append(sim)

    return np.array(similarities)


def evaluate(similarities, labels, data, threshold=0.450, return_predictions=False, verbose=False):
    """
        Evaluates accuracy based on a threshold for similarity.

        If verbose=True, prints false positives, true negatives, and relevant sentences.
    """
    predictions = ['T' if sim > threshold else 'F' for sim in similarities]
    correct_answers_count = sum(pred == true_label for pred, true_label in zip(predictions, labels))
    accuracy = correct_answers_count / len(labels)

    if verbose:
        print("\nFalse Positives (Predicted T but should be F):")
        for i, (pred, true_label, sim, (word, pos, index1, index2, sentence_a, sentence_b)) in enumerate(zip(predictions, labels, similarities, data)):
            if pred == 'T' and true_label == 'F':
                print(f"Index {i}: Similarity = {sim:.3f}")
                print(f"Word: {word}")
                print(f"Sentence A: {sentence_a}")
                print(f"Sentence B: {sentence_b}")
                print("-" * 80)

        print("\nTrue Negatives (Predicted F and was correct):")
        for i, (pred, true_label, sim, (word, pos, index1, index2, sentence_a, sentence_b)) in enumerate(zip(predictions, labels, similarities, data)):
            if pred == 'F' and true_label == 'F':
                print(f"Index {i}: Similarity = {sim:.3f}")
                print(f"Word: {word}")
                print(f"Sentence A: {sentence_a}")
                print(f"Sentence B: {sentence_b}")
                print("-" * 80)

    if return_predictions:
        return accuracy, correct_answers_count, predictions
    return accuracy, correct_answers_count

if __name__ == "__main__":
    # Paths to WiC dataset files
    base_path = "C:/WiC_dataset/dev"
    data_file = os.path.normpath(os.path.join(base_path, "dev.data.txt"))
    gold_file = os.path.normpath(os.path.join(base_path, "dev.gold.txt"))

    # Load data and compute similarities
    data, labels = load_wic_data(data_file, gold_file)
    similarities = compute_similarity(data)

    # Evaluate model
    accuracy, correct_answers_count = evaluate(similarities, labels, data, verbose=True)
    print(f"Baseline accuracy: {accuracy:.3%}")
    print(f"{correct_answers_count} correct answer(s) out of {len(labels)} answers.")
