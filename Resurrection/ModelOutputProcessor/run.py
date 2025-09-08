def run():
    from Resurrection.ModelOutputProcessor.TernaryClassifier import TernaryClassifier
    tc = TernaryClassifier.TernaryClassifier()
    tc.classify()
    from Resurrection.ModelOutputProcessor.TernaryResultsProcessor import TernaryResultsProcessor
    ternaryResultsFile = 'ternaryResults.txt'
    tre = TernaryResultsProcessor.TernaryResultsProcessor(ternaryResultsFile)

    overallPerformanceReport = open("overallPerformanceReport.txt", "w")
    print(f'MatchPercentage: {tre.getMatchPercentage()}%', file=overallPerformanceReport)
    print(f'Consistency: {tre.getConsistencyPercentage()}%', file=overallPerformanceReport)

    print(f'True positives: {tc.getTruePositives()}', file=overallPerformanceReport)
    print(f'False positives: {tc.getFalsePositives()}', file=overallPerformanceReport)
    print(f'False negatives: {tc.getFalseNegatives()}', file=overallPerformanceReport)
    print(f'True negatives: {tc.getTrueNegatives()}', file=overallPerformanceReport)