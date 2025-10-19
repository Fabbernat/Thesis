import nltk
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk

# Download necessary resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


def lesk_wic_decision(word: str, sentence1: str, sentence2: str):
    """
    Uses the Lesk algorithm to determine if `word` has the same meaning in two sentences.
    Returns 'Yes' or 'No' with a confidence score of 'Yes'.
    """
    # Tokenize the sentences
    tokens1 = nltk.word_tokenize(sentence1)
    tokens2 = nltk.word_tokenize(sentence2)

    # Apply Lesk algorithm to both sentences
    sense1 = lesk(tokens1, word)
    sense2 = lesk(tokens2, word)

    # If Lesk returns a sense, compare it to check if it's the same meaning
    if sense1 and sense2:
        # Compute similarity based on the glosses of both senses
        similarity = sense1.wup_similarity(sense2)
        if similarity is None:
            similarity = 0  # if no similarity value, treat it as 0

        confidence_yes = round(similarity * 100, 2)
    else:
        confidence_yes = 0

    # Decision based on confidence score
    threshold = 70  # You can adjust this threshold for more/less strict decision
    decision = "Yes" if confidence_yes >= threshold else "No"

    return decision, confidence_yes


# Example usage
word = "bank"
sentence1 = "She went to the bank to withdraw money."
sentence2 = "The river overflowed its bank after the storm."

result, confidence = lesk_wic_decision(word, sentence1, sentence2)
print(f"Same meaning? {result} (Confidence in YES: {confidence}%)")

# sample output:
# Same meaning? No (Confidence in YES: 34.12%)
