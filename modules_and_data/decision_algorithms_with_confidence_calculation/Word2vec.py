import nltk
import gensim.downloader as api
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Download NLTK tokenizer
nltk.download('punkt')

# Load pretrained Word2Vec model
print("Loading Word2Vec model (this might take a bit)...")
w2v_model = api.load("word2vec-google-news-300")
print("Model loaded.")


def get_context_vector(sentence, word, model, window=2):
    tokens = word_tokenize(sentence.lower())
    word = word.lower()

    if word not in tokens:
        return None  # word not found in sentence

    idx = tokens.index(word)
    start = max(0, idx - window)
    end = min(len(tokens), idx + window + 1)
    context_tokens = [t for t in tokens[start:end] if t != word and t in model]

    if not context_tokens:
        return None

    # Average the vectors of the context words
    vectors = [model[t] for t in context_tokens]
    return np.mean(vectors, axis=0)


def word2vec_wic_decision(word, sentence1, sentence2):
    vec1 = get_context_vector(sentence1, word, w2v_model)
    vec2 = get_context_vector(sentence2, word, w2v_model)

    if vec1 is None or vec2 is None:
        return "No", 0.0

    # Cosine similarity
    similarity = cosine_similarity([vec1], [vec2])[0][0]
    confidence_yes = round(similarity * 100, 2)

    # Decision
    threshold = 70.0
    decision = "Yes" if confidence_yes >= threshold else "No"

    return decision, confidence_yes


# Example usage
word = "bank"
sentence1 = "She went to the bank to withdraw money."
sentence2 = "The river overflowed its bank after the storm."

result, confidence = word2vec_wic_decision(word, sentence1, sentence2)
print(f"Same meaning? {result} (Confidence in YES: {confidence}%)")

# sample output:
# Same meaning? No (Confidence in YES: 34.12%)
