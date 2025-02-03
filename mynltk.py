import nltk
from nltk.corpus import wordnet as wn

# Ensure you have downloaded the WordNet data
nltk.download('wordnet')

# Example usage: Get synonyms for the word "bank"
'''public'''
word = 'bank'
synonyms = wn.synsets(word)

print(f'Bank definition: {synonyms[0]}')
for syn in synonyms:
    print('Bank synonyms:')
    print(syn.name(), syn.definition())

print()
print('End of bank synonyms. Continuing... \n')

