from addition.LabelAdder.LabelAdder import TestFilesMerger
from addition.SentenceBuilder.SentenceBuilder import SentenceBuilder

TestFilesMerger testFilesMerger = new TestFilesMerger()
mergedTestValues = testFilesMerger.mergeTestfiles()
WordAndSentencesExtractor wase = new WordAndSentencesExtractor()
SentenceBuilder sb = new SentenceBuilder()

straightSentences = []
reversedSentences = []

for rowValues in mergedTestValues:
    word, sentenceA, sentenceB = wase.extract(rowValues)

    normalizedSentenceA =
    normalizedSentenceB =

    straightSentence = sb.buildStraightSentence(word, sentenceA, sentenceB)
    reversedSentence = sb.buildReversedSentence(word, sentenceA, sentenceB)
    straightSentences.append(straightSentence)
    reversedSentences.append(reversedSentence)
