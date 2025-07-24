from Resurrection.SentenceNormalizer import SentenceNormalizer
from Resurrection.LabelAdder.LabelAdder import TestFilesMerger
from Resurrection.SentenceBuilder.SentenceBuilder import SentenceBuilder
from Resurrection.WordAndSentencesExtractor.WordAndSentencesExtractor import WordAndSentencesExtractor

testFilesMerger: TestFilesMerger = TestFilesMerger()
mergedTestValues = testFilesMerger.mergeTestfiles()
wase: WordAndSentencesExtractor =  WordAndSentencesExtractor()
sb: SentenceBuilder  = SentenceBuilder()
sn: SentenceNormalizer = SentenceNormalizer()
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
