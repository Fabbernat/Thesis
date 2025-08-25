import sys


TESTFILE_LENGTH = 2800


def main():
    from Resurrection.ModelOutputProcessor.TernaryClassifier import TernaryClassifier
    tf = TernaryClassifier()
    tf.classify()
    from Resurrection.ModelOutputProcessor.ternaryResultsExtractor import ResultsExtractor
    re = ResultsExtractor()
    overallPerformanceReport = open("overallPerformanceReport.txt", "w")
    print(f'MatchPercentage: {re.getMatchPercentage()}%', file=overallPerformanceReport)
    print(f'Consistency: {re.getConsistencyPercentage()}%', file=overallPerformanceReport)


if __name__ == '__main__':
    main()
