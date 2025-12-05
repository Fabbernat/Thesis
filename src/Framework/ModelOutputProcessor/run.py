# run.py
from datetime import datetime
from pathlib import Path


try:
    from src.Framework.ModelOutputProcessor.TernaryClassifier import TernaryClassifier
    from src.Framework.ModelOutputProcessor.TernaryResultsProcessor import BalancedAccuracyCalculator

except Exception:
    from TernaryClassifier import TernaryClassifier
    from TernaryResultsProcessor import BalancedAccuracyCalculator


def run():

    tc = TernaryClassifier.TernaryClassifier()
    # Opens files on paths specified in config.py 
    modelAnswerLineTsOrFs, goldAnswerLineTsOrFs = tc.classify0()

    try:
        from src.Framework.ModelOutputProcessor.TernaryResultsProcessor import TernaryResultsProcessor
    except Exception:
        from TernaryResultsProcessor import TernaryResultsProcessor

    basePath = Path(__file__).parent
    ternaryResultsPath = Path(str(basePath) + r'\data\ternaryResults.out')
    trp = TernaryResultsProcessor.TernaryResultsProcessor(ternaryResultsPath)


    overallPerformanceReport = open(Path(str(basePath) + r'\data\overallPerformanceReport.out'), 'w')
    logFile = open(Path(str(basePath) + r'\data\logFile.out'), 'a')

    now = datetime.now()
    formattedDate = now.strftime('%Y. %m. %d. %H:%M')

    thisModelName = 'an unknown model'
    try:
        from src.Framework.ModelOutputProcessor.config import USERNAME
        from src.Framework.HuggingFaceModelInferencer.modelname import MODEL_NAME
        thisModelName = MODEL_NAME
    except Exception as e:
        print('Exception:', e)
        from config import USERNAME
        try:
            from ..HuggingFaceModelInferencer.modelname import MODEL_NAME
            thisModelName = MODEL_NAME
        except Exception as e:
            print('Exception:', e)

    print(f'{USERNAME} ran {thisModelName} at {formattedDate} with results', file=logFile)

    files = [overallPerformanceReport, logFile]

    tp = tc.getTruePositives()
    fp = tc.getFalsePositives()
    fn = tc.getFalseNegatives()
    tn = tc.getTrueNegatives()
    matchPercentage = trp.getMatchPercentage()
    consistencyPercentage = trp.getConsistencyPercentage()
    balancedAccuracyPercentage = BalancedAccuracyCalculator.calculateBalancedAccuracy(modelAnswerLineTsOrFs, goldAnswerLineTsOrFs, tp, fp, fn, tn)
    length = tc.modelAnswersLengthInLines

    for file in files:
        print(f'MatchPercentage: {matchPercentage:.2f} %', file=file)
        print(f'Consistency: {consistencyPercentage:.2f} %', file=file)
        print(f'Consistently accurate: {(matchPercentage / 100 * consistencyPercentage / 100) * 100:.2f} % ', file=file) # soronként szorozzam össze
        print(f'Balanced accuracy: {balancedAccuracyPercentage:.2f} %', file=file)
        print(f'True positives: {tp}\t\t{tp * 100 / length:.2f} %', file=file)
        print(f'False positives: {fp}\t\t{fp * 100 / length:.2f} %', file=file)
        print(f'False negatives: {fn}\t\t{fn * 100 / length:.2f} %', file=file)
        print(f'True negatives: {tn}\t\t{tn * 100 / length:.2f} %', file=file)

    print('--------\n\n', file=logFile)