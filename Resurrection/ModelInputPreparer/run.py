import json

from Resurrection.ModelInputPreparer.LabelAdder.LabelAdder import TestFilesMerger
from Resurrection.ModelInputPreparer.SentenceBuilder.SentenceBuilder import SentenceBuilder
from Resurrection.ModelInputPreparer.SentenceNormalizer.SentenceNormalizer import SentenceNormalizer
from Resurrection.ModelInputPreparer.WordAndSentencesExtractor.WordAndSentencesExtractor import \
    WordAndSentencesExtractor


def run(logPartialResults=False):
    testFilesMerger: TestFilesMerger = TestFilesMerger()
    mergedTestValues = testFilesMerger.mergeTestfiles() # this line assumes that there are "test.data.txt" and "test.gold.txt" in this directory
    if logPartialResults:
        print(mergedTestValues) #edddig okés
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

    with open('data.json') as dataJson:
        print(json.loads('\n'.join(straightSentences)), file=dataJson)
        print(json.loads('\n'.join(reversedSentences)), file=dataJson)
