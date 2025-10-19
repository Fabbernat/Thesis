import sys

from src.Framework.ModelInputPreparer.LabelAdder.LabelAdder import TestFilesMerger
from src.Framework.ModelInputPreparer.SentenceBuilder.SentenceBuilder import SentenceBuilder
from src.Framework.ModelInputPreparer.SentenceNormalizer.SentenceNormalizer import SentenceNormalizer
from src.Framework.ModelInputPreparer.WordAndSentencesExtractor.WordAndSentencesExtractor import \
    WordAndSentencesExtractor


def run(logPartialResults=False):
    testFilesMerger: TestFilesMerger = TestFilesMerger()
    mergedTestValues = testFilesMerger.mergeTestfiles() # this line assumes that there are "test.data.txt" and "test.gold.txt" in this directory

    if logPartialResults:
        print(mergedTestValues) #eddig okés

    wase: WordAndSentencesExtractor =  WordAndSentencesExtractor()
    sentenceBuilder: SentenceBuilder  = SentenceBuilder()
    sentenceNormalizer: SentenceNormalizer = SentenceNormalizer()
    straightSentences = []
    reversedSentences = []

    for rowValues in mergedTestValues.split('\n'):
        word, sentenceA, sentenceB = wase.extract(rowValues)

        if logPartialResults:
            print('\n--\n', word, sentenceA, sentenceB) # ez is okés

        normalizedSentenceA = sentenceNormalizer.makeSentenceHumanReadable(sentenceA)
        normalizedSentenceB = sentenceNormalizer.makeSentenceHumanReadable(sentenceB)

        straightSentence = sentenceBuilder.buildStraightSentence(word, normalizedSentenceA, normalizedSentenceB)
        reversedSentence = sentenceBuilder.buildReversedSentence(word, normalizedSentenceA, normalizedSentenceB)
        straightSentences.append(straightSentence)
        reversedSentences.append(reversedSentence)

    try:
        with open("data.out", "w", encoding="utf-8") as dataJson:
            print("\n".join(straightSentences), file=dataJson)
            print("\n".join(reversedSentences), file=dataJson)
    except OSError as oe:
            sys.stderr.write(f"File writing error: {oe}\n")

    except ValueError as ve:
        sys.stderr.write(f"Error while writing data.json: {ve}\n")

