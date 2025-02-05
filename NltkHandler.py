import nltk
from nltk.corpus import wordnet as wn

def download_wordnet_if_needed():
    """Downloads WordNet if it hasn't been downloaded already."""
    try:
        from nltk.corpus import wordnet  # Try importing WordNet first
        # If the import succeeds, WordNet is likely already downloaded.
        # You can optionally test if a specific WordNet function works:
        wordnet.synsets('car') # A quick test
        print("WordNet is already downloaded.")
        return True # Indicate success

    except LookupError: # If the import fails, it's not downloaded
        print("WordNet is not found. Downloading...")
        try:
            nltk.download('wordnet')
            print("WordNet downloaded successfully.")
            return True #Indicate success
        except Exception as e:
            print(f"Error downloading WordNet: {e}")
            return False # Indicate failure

    except Exception as e: # Catch any other potential errors
        print(f"An unexpected error occurred: {e}")
        return False # Indicate failure


# Ensure you have downloaded the WordNet data
if download_wordnet_if_needed():
    from nltk.corpus import wordnet
else:
    print("WordNet is not available. Please check your internet connection and try again.")

# Example usage: Get synonyms for the word "bank"
'''public'''
word = 'bank'
synonyms = wn.synsets(word)

for syn in synonyms:
    print('Bank synonyms:')
    print(f'name: {syn.name()}, definition: {syn.definition()}')

print()
print('End of bank synonyms. Continuing... \n')

def get_synonyms(word):
    """Returns a set of synonyms for a given word using WordNet."""
    synonyms = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace('_', ' '))  # Convert underscores to spaces
    return synonyms


def expand_with_synonyms(sentence):
    """Expands a sentence by adding synonyms for each word."""
    words = sentence.split()
    expanded_words = []
    for word in words:
        synonyms = get_synonyms(word)
        if synonyms:
            expanded_words.append(word + " " + " ".join(synonyms))
        else:
            expanded_words.append(word)
    return " ".join(expanded_words)
