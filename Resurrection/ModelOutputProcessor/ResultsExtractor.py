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
        return self.countMatches() / len(self.ternaryResultsLines)


    def countConsistentAnswers(self):
        consistentAnswers = 0
        for i in range(int(len(self.ternaryResultsLines) / 2)):
            if self.ternaryResultsLines[i] == self.ternaryResultsLines[i + TESTFILE_LENGTH]:
                consistentAnswers += 1

        return consistentAnswers

    def getConsistency(self):
        return self.countConsistentAnswers() / len(self.ternaryResultsLines)