def run():
    from Resurrection.ModelOutputProcessor.TernaryClassifier import TernaryClassifier
    tc = TernaryClassifier.TernaryClassifier()
    tc.classify()
    from Resurrection.ModelOutputProcessor.TernaryResultsProcessor import TernaryResultsProcessor
    ternaryResultsFile = 'ternaryResults.out'
    tre = TernaryResultsProcessor.TernaryResultsProcessor(ternaryResultsFile)

    overallPerformanceReport = open("overallPerformanceReport.out", "w")
    print(f'MatchPercentage: {tre.getMatchPercentage()}%', file=overallPerformanceReport)
    print(f'Consistency: {tre.getConsistencyPercentage()}%', file=overallPerformanceReport)

    tp = tc.getTruePositives()
    fp = tc.getFalsePositives()
    fn = tc.getFalseNegatives()
    tn = tc.getTrueNegatives()

    from Resurrection.ModelOutputProcessor.main import TESTFILE_LENGTH
    print(f'True positives: {tp}\t\t{tp * 100 / TESTFILE_LENGTH:.2f} %', file=overallPerformanceReport)
    print(f'False positives: {fp}\t\t{fp * 100 / TESTFILE_LENGTH:.2f} %', file=overallPerformanceReport)
    print(f'False negatives: {fn}\t\t{fn * 100 / TESTFILE_LENGTH:.2f} %', file=overallPerformanceReport)
    print(f'True negatives: {tn}\t\t{tn * 100 / TESTFILE_LENGTH:.2f} %', file=overallPerformanceReport)
