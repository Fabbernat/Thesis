def run():
    from src.Framework.ModelOutputProcessor.TernaryClassifier import TernaryClassifier
    tc = TernaryClassifier.TernaryClassifier()
    tc.classify()
    from src.Framework.ModelOutputProcessor.TernaryResultsProcessor import TernaryResultsProcessor
    ternaryResultsPath = 'data/ternaryResults.out'
    tre = TernaryResultsProcessor.TernaryResultsProcessor(ternaryResultsPath)

    overallPerformanceReport = open("data/overallPerformanceReport.out", "w")
    print(f'MatchPercentage: {tre.getMatchPercentage()}%', file=overallPerformanceReport)
    print(f'Consistency: {tre.getConsistencyPercentage()}%', file=overallPerformanceReport)

    tp = tc.getTruePositives()
    fp = tc.getFalsePositives()
    fn = tc.getFalseNegatives()
    tn = tc.getTrueNegatives()

    length = tc.NUMBER_OF_LINES
    print(f'True positives: {tp}\t\t{tp * 100 / length:.2f} %', file=overallPerformanceReport)
    print(f'False positives: {fp}\t\t{fp * 100 / length:.2f} %', file=overallPerformanceReport)
    print(f'False negatives: {fn}\t\t{fn * 100 / length:.2f} %', file=overallPerformanceReport)
    print(f'True negatives: {tn}\t\t{tn * 100 / length:.2f} %', file=overallPerformanceReport)