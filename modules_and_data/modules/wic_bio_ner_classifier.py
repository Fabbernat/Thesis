# C:\PycharmProjects\Peternity\modules_and_data\modules\wic_bio_ner_classifier.py
# First you need to install SpaCy
import spacy


def get_bio_classification(sentences):
    nlp = spacy.load("en_core_web_sm")
    results = []

    for sentence in sentences:
        doc = nlp(sentence)
        sentence_result = []

        for token in doc:
            label = "O"  # Default label
            if token.ent_iob_ == "B":
                if token.ent_type_ == "PERSON":
                    label = "B-PER"
                elif token.ent_type_ == "GPE":  # Geographic location
                    label = "B-LOC"
                elif token.ent_type_ == "ORG":
                    label = "B-ORG"
                else:
                    label = "B-MISC"
            elif token.ent_iob_ == "I":
                if token.ent_type_ == "PERSON":
                    label = "I-PER"
                elif token.ent_type_ == "GPE":
                    label = "I-LOC"
                elif token.ent_type_ == "ORG":
                    label = "I-ORG"
                else:
                    label = "I-MISC"

            sentence_result.append((token.text, label))

        results.append(sentence_result)

    return results


# Example usage:
sentences = [
    "I do it for the fun of it .", "He is fun to have around .",
    "The man must answer to his employer for the money entrusted to his care .", "She must answer for her actions .",
    "A history of France .", "A critical time in the school's history ."
]

bio_output = get_bio_classification(sentences)
for sent in bio_output:
    for word, label in sent:
        print(f"{word}\t{label}")
    print()
