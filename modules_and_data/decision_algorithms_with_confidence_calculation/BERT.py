from sentence_transformers import SentenceTransformer, util

# Load a lightweight pretrained BERT model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def bert_wic_decision(word: str, sentence1: str, sentence2: str):
    """
    Uses BERT embeddings to decide if `word` has the same meaning in both sentences.
    Returns 'Yes' or 'No', and a confidence percentage of 'Yes'.
    """
    # Emphasize the word in both sentences (helps focus the model)
    marked_s1 = sentence1.replace(word, f"[{word}]")
    marked_s2 = sentence2.replace(word, f"[{word}]")

    # Encode both sentences
    embeddings = model.encode([marked_s1, marked_s2], convert_to_tensor=True)

    # Compute cosine similarity
    cosine_sim = util.cos_sim(embeddings[0], embeddings[1]).item()
    confidence_yes = round(cosine_sim * 100, 2)

    # Decision threshold
    threshold = 75.0  # Tune this based on validation
    decision = "Yes" if confidence_yes >= threshold else "No"

    return decision, confidence_yes

# Example usage
word = "bank"
sentence1 = "She went to the bank to withdraw money."
sentence2 = "The river overflowed its bank after the storm."

result, confidence = bert_wic_decision(word, sentence1, sentence2)
print(f"Same meaning? {result} (Confidence in YES: {confidence}%)")

# sample output:
# Same meaning? No (Confidence in YES: 34.12%)
