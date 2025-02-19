"""
 Functions to make the datasets human-readable
"""
def normalize_sentence(sentence):
    """Replaces contractions for better word sense disambiguation."""
    return sentence.replace(" 's", "'s")


def normalize_negations(sentence):
    """Replaces n't with 'not' for better word sense disambiguation."""
    sentence = sentence.replace(" n't", "not")
    return sentence