import sys

from src.Framework.ModelOutputProcessor.config import GoldFileLengthInLines


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
        if len(self.ternaryResultsLines) % 2 != 0:
            key = input(
                'Warning: cannot count consistency when odd number of lines, please fix input. The last line  will be dropped. Do you wish to continue? (y/n)')
            if key == 'n':
                exit(0)
        else:
            print("Number of ternaryResultsLines: ", len(self.ternaryResultsLines))

        for i in range(len(self.ternaryResultsLines) // 2):
            reversedIndex = i + GoldFileLengthInLines if i + GoldFileLengthInLines < len(self.ternaryResultsLines) else -i
            if self.ternaryResultsLines[i] == self.ternaryResultsLines[reversedIndex]:
                consistentAnswers += 1

        return consistentAnswers

    def getConsistencyPercentage(self):
        return (self.countConsistentAnswers() / int(len(self.ternaryResultsLines) / 2)) * 100