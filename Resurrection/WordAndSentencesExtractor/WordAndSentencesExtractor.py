class WordAndSentencesExtractor:
    def __init__(self):
        pass

    def extract(self, rowValues) -> (str, str, str):
        return rowValues[0], rowValues[3], rowValues[4]