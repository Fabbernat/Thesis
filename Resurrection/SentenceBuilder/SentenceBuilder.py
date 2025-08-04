class SentenceBuilder:
    def __init__(self):
        self.sentence_a = None
        self.sentence_b = None

    def setSentences(self, sentence_a, sentence_b):
        self.sentence_a = sentence_a
        self.sentence_b = sentence_b

    def buildStraightSentence(self, word: str, sentence_a: str, sentence_b: str) -> str:
        return f'Does the word "{word}" mean the same thing in sentences "{sentence_a}" and "{sentence_b}"?'

    def buildReversedSentence(self, word: str, sentence_a: str, sentence_b: str) -> str:
        return f'Does the word "{word}" mean the same thing in sentences "{sentence_b}" and "{sentence_a}"?'