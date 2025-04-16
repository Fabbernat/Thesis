# C:\PycharmProjects\Peternity\src\utils\wic_sentence_normalizer.py
import re

def make_sentence_human_readable(sentence: str) -> str:
    """
    Enhances sentence readability by handling contractions and escaping single quotes correctly.

    - Escapes single quotes after specific characters.
    - Fixes spacing issues before punctuation marks.
    """
    # Escape single quotes when needed, except after a period
    sentence = re.sub(r"(?<!\.)'", r"\\'", sentence)

    # Fix spacing before punctuation
    sentence = sentence.replace(" ,", ",").replace(" .", ".")

    return sentence



def make_sentence_human_readable_old(sentence: str) -> str:
    """Replaces contractions for better readability both for humans and for chatbots."""
    sentence = sentence.replace(" 's", "\\'s")
    sentence = sentence.replace(" 't", "\\'t")
    sentence = sentence.replace(" 've", "\\'ve")
    sentence = sentence.replace(" 'll", "\\'ll")
    sentence = sentence.replace(" 'd", "\\'d")
    sentence = sentence.replace(" 're", "\\'re")
    sentence = sentence.replace(" 's", "\\'s")
    sentence = sentence.replace(" ,", ",")
    sentence = sentence.replace(" .", ".")
    # Uppercase
    sentence = sentence.replace("Do n't", "Don't")
    sentence = sentence.replace("Wo n't", "Won't")
    sentence = sentence.replace("Ca n't", "Can't")

    # Lowercase
    sentence = sentence.replace("do n't", "don't")
    sentence = sentence.replace("wo n't", "won't")
    sentence = sentence.replace("ca n't", "can't")
    sentence = sentence.replace(" n't", "not")
    return sentence

def main():
    print(make_sentence_human_readable(
    r"We 've been swimming for hours just to get to the other side .	A big fish was swimming in the tank . Do n't fire until you see the whites of their eyes .	The gun fired ."
    ))

if __name__ == '__main__':
    main()
