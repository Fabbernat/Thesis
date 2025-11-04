from datetime import datetime



def run():
    from src.Framework.ModelOutputProcessor.TernaryClassifier import TernaryClassifier
    tc = TernaryClassifier.TernaryClassifier()
    tc.classify()
    from src.Framework.ModelOutputProcessor.TernaryResultsProcessor import TernaryResultsProcessor
    ternaryResultsPath = 'data/ternaryResults.out'
    tre = TernaryResultsProcessor.TernaryResultsProcessor(ternaryResultsPath)

    files = []
    overallPerformanceReport = open("data/overallPerformanceReport.out", "w")
    logFile = open("data/logFile.out", "a")

    now = datetime.now()
    formattedDate = now.strftime("%Y. %m. %d. %H:%M")

    from Framework.HuggingFaceModelInferencer.config import MODEL_NAME
    from Framework.ModelOutputProcessor.config import USERNAME
    print(f'{USERNAME} ran {MODEL_NAME} at {formattedDate} with results', file=logFile)
    files.append(overallPerformanceReport)
    files.append(logFile)

    tp = tc.getTruePositives()
    fp = tc.getFalsePositives()
    fn = tc.getFalseNegatives()
    tn = tc.getTrueNegatives()


    length = tc.NUMBER_OF_LINES
    for file in files:
        print(f'MatchPercentage: {tre.getMatchPercentage()}%', file=file)
        print(f'Consistency: {tre.getConsistencyPercentage()}%', file=file)

        print(f'True positives: {tp}\t\t{tp * 100 / length:.2f} %', file=file)
        print(f'False positives: {fp}\t\t{fp * 100 / length:.2f} %', file=file)
        print(f'False negatives: {fn}\t\t{fn * 100 / length:.2f} %', file=file)
        print(f'True negatives: {tn}\t\t{tn * 100 / length:.2f} %', file=file)

    print('--------\n\n', file=logFile)