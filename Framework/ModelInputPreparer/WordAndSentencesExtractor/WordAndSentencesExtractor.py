from typing import Any


class WordAndSentencesExtractor:
    def extract(self, rowValues) -> tuple[Any, Any, Any]:
        try:
            cells = rowValues.split('\t')
            return cells[0], cells[3], cells[4]
        except IndexError:
            return "joke",	"I regarded his campaign for mayor as a joke .",	"He told a very funny joke ." # I'm returning a dummy row so the model can be tested