# C:\PycharmProjects\Peternity\solution\word_sense.py
from sentence_transformers import SentenceTransformer, util
from nltk.corpus import wordnet as wn
from typing import Optional, Set

class WordSenseDisambiguator:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def get_best_sense(self, word: str, sentence: str) -> Optional[wn.synset]:
        """Uses sentence embeddings to disambiguate word senses."""
        synsets = wn.synsets(word)
        if not synsets:
            return None

        sentence_embedding = self.model.encode(sentence, convert_to_tensor=True)
        best_sense = max(synsets, key=lambda sense:
            util.pytorch_cos_sim(
                sentence_embedding,
                self.model.encode(sense.definition(), convert_to_tensor=True)
            ).item())

        return best_sense

    def get_disambiguated_synonyms(self, word: str, sentence: str) -> Set[str]:
        """Gets synonyms only for the most relevant sense of the word."""
        sense = self.get_best_sense(word, sentence)
        if sense:
            return {lemma.name().replace("_", " ") for lemma in sense.lemmas()}
        return set()