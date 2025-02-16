'''
 Functions to make the datasets human-readable
'''
def normalize_sentence(sentence):
    """Replaces contractions for better word sense disambiguation."""
    return sentence.replace(" 's", "'s")  # Example normalization


def normalize_negations(sentence):
    """Replaces contractions like n't with 'not' for better word sense disambiguation."""
    sentence = sentence.replace("n't", "not")
    return sentence