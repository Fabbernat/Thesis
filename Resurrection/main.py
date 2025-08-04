from Resurrection.SentenceNormalizer.SentenceNormalizer import SentenceNormalizer
from Resurrection.LabelAdder.LabelAdder import TestFilesMerger
from Resurrection.SentenceBuilder.SentenceBuilder import SentenceBuilder
from Resurrection.WordAndSentencesExtractor.WordAndSentencesExtractor import WordAndSentencesExtractor

testFilesMerger: TestFilesMerger = TestFilesMerger()
mergedTestValues = testFilesMerger.mergeTestfiles() # this line assumes that there are "test.data.txt" and "test.gold.txt" in this directory
# print(mergedTestValues) #edddig okés
wase: WordAndSentencesExtractor =  WordAndSentencesExtractor()
sentenceBuilder: SentenceBuilder  = SentenceBuilder()
sentenceNormalizer: SentenceNormalizer = SentenceNormalizer()
straightSentences = []
reversedSentences = []

for rowValues in mergedTestValues.split('\n'):
    word, sentenceA, sentenceB = wase.extract(rowValues)
    # print('\n--\n', word, sentenceA, sentenceB) # ez is okés

    normalizedSentenceA = sentenceNormalizer.makeSentenceHumanReadable(sentenceA)
    normalizedSentenceB = sentenceNormalizer.makeSentenceHumanReadable(sentenceB)

    straightSentence = sentenceBuilder.buildStraightSentence(word, sentenceA, sentenceB)
    reversedSentence = sentenceBuilder.buildReversedSentence(word, sentenceA, sentenceB)
    straightSentences.append(straightSentence)
    reversedSentences.append(reversedSentence)

print('\n'.join(straightSentences))
print('\n'.join(reversedSentences))
