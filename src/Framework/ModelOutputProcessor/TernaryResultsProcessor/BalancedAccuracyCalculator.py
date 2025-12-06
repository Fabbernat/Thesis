from itertools import count

# try:
#     from src.Framework.ModelOutputProcessor.config import ModelAnswersLengthInLines
# except Exception:
#     from config import ModelAnswersLengthInLines
try:
    from src.Framework.ModelOutputProcessor.TernaryClassifier.TernaryClassifier import getYesOrNo
except Exception:
    from TernaryClassifier.TernaryClassifier import getYesOrNo

def calculateAmbiguousness(threeAnswersToCompareListOfTuples, modelAnswerLineYesOrNos, groundTruths, tp, fp, fn, tn):
    if len(modelAnswerLineYesOrNos) % 2 != 0:
        keyInput = input('')
        if keyInput == 'n':
            exit(0)


    half = threeAnswersToCompareListOfTuples
    ambiguous: list[bool] = []
    consistentlyAmbiguous: list[bool] = []
    for index, elem in enumerate(
        threeAnswersToCompareListOfTuples):  # a threeAnswersToCompareListOfTuples elérhető kell hogy legyen itt is
        if getYesOrNo(elem[0]) == '?' or getYesOrNo(elem[1]) == '?':
            ambiguous.append(True)
            if elem[2] == '?':
                consistentlyAmbiguous.append(True)
            else:
                consistentlyAmbiguous.append(False)
        else:
            ambiguous.append(False)

        if index + 1 == half:
            break # Ez nem biztos h kell túlindexelni azért itt sem kéne.
    print(modelAnswerLineYesOrNos)
    print(groundTruths)

    print(f"\nConfusion Matrix:")
    print(f"True Positives (tp): {tp}")
    print(f"True Negatives (tn): {tn}")
    print(f"False Positives (fp): {fp}")
    print(f"False Negatives (fn): {fn}")



    return ambiguous,consistentlyAmbiguous, threeAnswersToCompareListOfTuples # csak az interfész biztosított, az implementacio nem


# ModelAnswersLengthInLines = len(modelAnswerLineYesOrNos)
# print(ModelAnswersLengthInLines)
# yesAnswersModifier = (sum([1 if elem == 'T' else 0 for elem in modelAnswerLineYesOrNos]) / ModelAnswersLengthInLines)
# print(yesAnswersModifier)
# noAnswersModifier = (sum([0 if elem == 'T' else 1 for elem in modelAnswerLineYesOrNos]) / ModelAnswersLengthInLines)
# balancedAccuracyPercentage = yesAnswersModifier * noAnswersModifier
# print(balancedAccuracyPercentage)