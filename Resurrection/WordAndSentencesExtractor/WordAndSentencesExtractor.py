class WordAndSentencesExtractor:
    def __init__(self):
        pass

    def extract(self, rowValues) -> (str, str, str):
        cells = rowValues.split('\t')
        return cells[0], cells[3], cells[4]