from Resurrection.ModelOutputProcessor.main import TESTFILE_LENGTH


class ResultsExtractor:

    def __init__(self):
        with open('ternaryResults.txt') as ternaryResultsFile:
            self.ternaryResultsLines = ternaryResultsFile.readlines()

    def getResultsFromTernaryJson(self):
        pass

    def countMatches(self):
        matches = 0
        for line in self.ternaryResultsLines:
            if line == 'True':
                matches += 1
        return matches

    def getMatchPercentage(self):
        return (self.countMatches() / int(len(self.ternaryResultsLines) / 2)) * 100

    def countConsistentAnswers(self):
        consistentAnswers = 0
        for i in range(int(len(self.ternaryResultsLines) / 2)):
            reversedIndex = i + TESTFILE_LENGTH if i + TESTFILE_LENGTH < len(self.ternaryResultsLines) else 0
            if self.ternaryResultsLines[i] == self.ternaryResultsLines[reversedIndex]:
                consistentAnswers += 1

        return consistentAnswers

    def getConsistencyPercentage(self):
        return (self.countConsistentAnswers() / int(len(self.ternaryResultsLines) / 2)) * 100
