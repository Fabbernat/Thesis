from datetime import datetime
from pathlib import Path


def run():
    from src.Framework.ModelOutputProcessor.TernaryClassifier import TernaryClassifier

    tc = TernaryClassifier.TernaryClassifier()
    # Opens files on paths specified in config.py 
    tc.classify0()


    from src.Framework.ModelOutputProcessor.TernaryResultsProcessor import TernaryResultsProcessor

    basePath = Path(__file__).parent
    ternaryResultsPath = Path(str(basePath) + r'\data\ternaryResults.out')
    tre = TernaryResultsProcessor.TernaryResultsProcessor(ternaryResultsPath)

    files = []
    overallPerformanceReport = open(Path(str(basePath) + r'\data\overallPerformanceReport.out'), 'w')
    logFile = open(Path(str(basePath) + r'\data\logFile.out'), 'a')

    now = datetime.now()
    formattedDate = now.strftime('%Y. %m. %d. %H:%M')

    from src.Framework.ModelOutputProcessor.config import USERNAME
    print(f'{USERNAME} ran an unknown model at {formattedDate} with results', file=logFile)
    files.append(overallPerformanceReport)
    files.append(logFile)

    tp = tc.getTruePositives()
    fp = tc.getFalsePositives()
    fn = tc.getFalseNegatives()
    tn = tc.getTrueNegatives()


    length = tc.modelAnswersLengthInLines

    for file in files:
        print(f'MatchPercentage: {tre.getMatchPercentage()}%', file=file)
        print(f'Consistency: {tre.getConsistencyPercentage()}%', file=file)

        print(f'True positives: {tp}\t\t{tp * 100 / length:.2f} %', file=file)
        print(f'False positives: {fp}\t\t{fp * 100 / length:.2f} %', file=file)
        print(f'False negatives: {fn}\t\t{fn * 100 / length:.2f} %', file=file)
        print(f'True negatives: {tn}\t\t{tn * 100 / length:.2f} %', file=file)

    print('--------\n\n', file=logFile)