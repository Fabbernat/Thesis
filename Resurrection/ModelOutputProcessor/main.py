import sys



TESTFILE_LENGTH = 1400


AFFIRMATIVE_KEYWORDS = ["Yes"]
NEGATIVE_KEYWORDS = ["No"]

def main():
    from Resurrection.ModelOutputProcessor.TernaryClassifier import TernaryClassifier
    tf = TernaryClassifier.TernaryClassifier()
    tf.classify()
    from Resurrection.ModelOutputProcessor.TernaryResultsProcessor import TernaryResultsProcessor
    ternaryResultsFile = 'ternaryResults.txt'
    tre = TernaryResultsProcessor.TernaryResultsProcessor(ternaryResultsFile)

    overallPerformanceReport = open("overallPerformanceReport.txt", "w")
    print(f'MatchPercentage: {tre.getMatchPercentage()}%', file=overallPerformanceReport)
    print(f'Consistency: {tre.getConsistencyPercentage()}%', file=overallPerformanceReport)

    print(f'True positives: {tre.}', file=overallPerformanceReport)
    print(f'False positives: {tre.}', file=overallPerformanceReport)
    print(f'False negatives: {tre.}', file=overallPerformanceReport)
    print(f'True negatives: {tre.}', file=overallPerformanceReport)


if __name__ == '__main__':
    main()
