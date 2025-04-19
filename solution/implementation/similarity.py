# C:\PycharmProjects\Peternity\solution\similarity.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Tuple


def compute_sentence_similarity(data: List[Tuple]) -> np.ndarray:
    """
    Computes cosine similarity between sentence pairs using TF-IDF.

    Args:
        data: List of tuples containing (word, pos, index1, index2, sentence_a, sentence_b)

    Returns:
        Array of similarity scores
    """
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(0, 1),
        max_df=0.85,
        min_df=2,
        sublinear_tf=True,
        norm='l2'
    )

    all_sentences = [
        sentence for _, _, _, _, sentence_a, sentence_b in data
        for sentence in (sentence_a, sentence_b)
    ]
    vectorizer.fit(all_sentences)

    similarities = []
    for _, _, _, _, sentence1, sentence2 in data:
        vectors = vectorizer.transform([sentence1, sentence2])
        sim = cosine_similarity(vectors[0], vectors[1])[0][0]
        similarities.append(sim)

    return np.array(similarities)