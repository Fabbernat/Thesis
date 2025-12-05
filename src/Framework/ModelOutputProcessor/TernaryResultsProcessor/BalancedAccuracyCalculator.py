from itertools import count

# try:
#     from src.Framework.ModelOutputProcessor.config import ModelAnswersLengthInLines
# except Exception:
#     from config import ModelAnswersLengthInLines

modelAnswerLineYesOrNos = ['T', 'F', 'F', 'T']
groundTruth = ['T', 'T', 'T', 'F']

def calculateBalancedAccuracy(modelAnswerLineYesOrNos, groundTruth, tp, fp, fn, tn):
    

    print(modelAnswerLineYesOrNos)
    print(groundTruth)

    print(f"\nConfusion Matrix:")
    print(f"True Positives (tp): {tp}")
    print(f"True Negatives (tn): {tn}")
    print(f"False Positives (fp): {fp}")
    print(f"False Negatives (fn): {fn}")

    # Calculate balanced accuracy according to your formula
    if tp + fn > 0:
        sensitivity = tp / (tp + fn)  # True Positive Rate / Recall
    else:
        sensitivity = 0

    if tn + fp > 0:
        specificity = tn / (tn + fp)  # True Negative Rate
    else:
        specificity = 0

    balancedAccuracyPercentage = (1 / 2) * (sensitivity + specificity) * 100

    print(f"\nSensitivity (True Positive Rate): {sensitivity:.2f}")
    print(f"Specificity (True Negative Rate): {specificity:.2f}")
    print(f"Balanced Accuracy: {balancedAccuracyPercentage:.2f}%")

    # For comparison, let's also calculate regular accuracy
    regular_accuracy = (tp + tn) / len(modelAnswerLineYesOrNos) * 100
    print(f"\nRegular Accuracy: {regular_accuracy:.2f}%")

    # Show why balanced accuracy is important
    print(f"\nAnalysis:")
    print(f"- The model predicted '{modelAnswerLineYesOrNos[0]}' for all {len(modelAnswerLineYesOrNos)} examples")
    print(f"- Regular accuracy gives: {regular_accuracy:.2f}%")
    print(f"- Balanced accuracy gives: {balancedAccuracyPercentage:.2f}%")
    if balancedAccuracyPercentage < regular_accuracy:
        print(f"- Balanced accuracy is lower because the model is biased toward '{modelAnswerLineYesOrNos[0]}'")


    return ambiguous,consistentlyAmbiguous


# ModelAnswersLengthInLines = len(modelAnswerLineYesOrNos)
# print(ModelAnswersLengthInLines)
# yesAnswersModifier = (sum([1 if elem == 'T' else 0 for elem in modelAnswerLineYesOrNos]) / ModelAnswersLengthInLines)
# print(yesAnswersModifier)
# noAnswersModifier = (sum([0 if elem == 'T' else 1 for elem in modelAnswerLineYesOrNos]) / ModelAnswersLengthInLines)
# balancedAccuracyPercentage = yesAnswersModifier * noAnswersModifier
# print(balancedAccuracyPercentage)