from itertools import count

# try:
#     from src.Framework.ModelOutputProcessor.config import ModelAnswersLengthInLines
# except Exception:
#     from config import ModelAnswersLengthInLines


def calculateBalancedAccuracy(modelAnswerLineYesOrNos, groundTruth, tp, fp, fn, tn):
    

    print(modelAnswerLineYesOrNos)
    print(groundTruth)

    print(f"\nConfusion Matrix:")
    print(f"True Positives (tp): {tp}")
    print(f"True Negatives (tn): {tn}")
    print(f"False Positives (fp): {fp}")
    print(f"False Negatives (fn): {fn}")
 


    return ambiguous,consistentlyAmbiguous # csak az interfész biztosított, az implementacio nem


# ModelAnswersLengthInLines = len(modelAnswerLineYesOrNos)
# print(ModelAnswersLengthInLines)
# yesAnswersModifier = (sum([1 if elem == 'T' else 0 for elem in modelAnswerLineYesOrNos]) / ModelAnswersLengthInLines)
# print(yesAnswersModifier)
# noAnswersModifier = (sum([0 if elem == 'T' else 1 for elem in modelAnswerLineYesOrNos]) / ModelAnswersLengthInLines)
# balancedAccuracyPercentage = yesAnswersModifier * noAnswersModifier
# print(balancedAccuracyPercentage)