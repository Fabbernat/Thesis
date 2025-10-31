from src.Framework.ModelOutputProcessor.config import TESTFILE_LENGTH


class TernaryResultsProcessor:

    def __init__(self, filePath):
        with open(filePath) as ternaryResultsFilePath:
            self.ternaryResultsLines = ternaryResultsFilePath.readlines()

    def getResultsFromTernaryJson(self):
        pass

    def countMatches(self):
        matches = 0
        for line in self.ternaryResultsLines:
            line = line.strip()
            if line.__contains__('True'):
                matches += 1
        return matches

    def getMatchPercentage(self):
        return (self.countMatches() / int(len(self.ternaryResultsLines))) * 100

    def countConsistentAnswers(self):
        consistentAnswers = 0
        for i in range(int(len(self.ternaryResultsLines) / 2)):
            reversedIndex = i + TESTFILE_LENGTH if i + TESTFILE_LENGTH < len(self.ternaryResultsLines) else 0
            if self.ternaryResultsLines[i] == self.ternaryResultsLines[reversedIndex]:
                consistentAnswers += 1

        return consistentAnswers

    def getConsistencyPercentage(self):
        return (self.countConsistentAnswers() / int(len(self.ternaryResultsLines) / 2)) * 100