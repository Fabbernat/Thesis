import sys



TESTFILE_LENGTH = 1400

AFFIRMATIVE_KEYWORDS = ["Yes"]
NEGATIVE_KEYWORDS = ["No"]

def main():
    from Resurrection.ModelOutputProcessor.TernaryClassifier import TernaryClassifier
    tf = TernaryClassifier.TernaryClassifier()
    tf.classify()
    from Resurrection.ModelOutputProcessor.TernaryResultsExtractor import TernaryResultsExtractor
    ternaryResultsFile = 'ternaryResults.txt'
    tre = TernaryResultsExtractor.TernaryResultsExtractor(ternaryResultsFile)

    overallPerformanceReport = open("overallPerformanceReport.txt", "w")
    print(f'MatchPercentage: {tre.getMatchPercentage()}%', file=overallPerformanceReport)
    print(f'Consistency: {tre.getConsistencyPercentage()}%', file=overallPerformanceReport)


if __name__ == '__main__':
    main()
