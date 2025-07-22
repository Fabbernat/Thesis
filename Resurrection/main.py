from Resurrection.LabelAdder.LabelAdder import TestFilesMerger
from Resurrection.SentenceBuilder.SentenceBuilder import SentenceBuilder

TestFilesMerger testFilesMerger = new TestFilesMerger()
mergedTestValues = testFilesMerger.mergeTestfiles()
WordAndSentencesExtractor wase = new WordAndSentencesExtractor()
SentenceBuilder sb = new SentenceBuilder()
SentenceNormalizer sn = new SentenceNormalizer()
straightSentences = []
reversedSentences = []

for rowValues in mergedTestValues:
    word, sentenceA, sentenceB = wase.extract(rowValues)

    normalizedSentenceA = sn.makeSentenceHumanReadable(sentenceA)
    normalizedSentenceB = sn.makeSentenceHumanReadable(sentenceB)

    straightSentence = sb.buildStraightSentence(word, sentenceA, sentenceB)
    reversedSentence = sb.buildReversedSentence(word, sentenceA, sentenceB)
    straightSentences.append(straightSentence)
    reversedSentences.append(reversedSentence)
