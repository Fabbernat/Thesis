from itertools import count

# try:
#     from src.Framework.ModelOutputProcessor.config import ModelAnswersLengthInLines
# except Exception:
#     from config import ModelAnswersLengthInLines

modelAnswerLineYesOrNos = ['T', 'F', 'F', 'T']
groundTruth = ['T', 'T', 'T', 'F']

TP = TN = FP = FN = 0


for pred, true in zip(modelAnswerLineYesOrNos, groundTruth):
    if pred == 'T' and true == 'T':
        TP += 1  # True Positive
    elif pred == 'F' and true == 'F':
        TN += 1  # True Negative
    elif pred == 'T' and true == 'F':
        FP += 1  # False Positive
    elif pred == 'F' and true == 'T':
        FN += 1  # False Negative


print(modelAnswerLineYesOrNos)
print(groundTruth)

print(f"\nConfusion Matrix:")
print(f"True Positives (TP): {TP}")
print(f"True Negatives (TN): {TN}")
print(f"False Positives (FP): {FP}")
print(f"False Negatives (FN): {FN}")

# Calculate balanced accuracy according to your formula
if TP + FN > 0:
    sensitivity = TP / (TP + FN)  # True Positive Rate / Recall
else:
    sensitivity = 0

if TN + FP > 0:
    specificity = TN / (TN + FP)  # True Negative Rate
else:
    specificity = 0

balancedAccuracyPercentage = (1 / 2) * (sensitivity + specificity) * 100

print(f"\nSensitivity (True Positive Rate): {sensitivity:.2f}")
print(f"Specificity (True Negative Rate): {specificity:.2f}")
print(f"Balanced Accuracy: {balancedAccuracyPercentage:.2f}%")

# For comparison, let's also calculate regular accuracy
regular_accuracy = (TP + TN) / len(modelAnswerLineYesOrNos) * 100
print(f"\nRegular Accuracy: {regular_accuracy:.2f}%")

# Show why balanced accuracy is important
print(f"\nAnalysis:")
print(f"- The model predicted '{modelAnswerLineYesOrNos[0]}' for all {len(modelAnswerLineYesOrNos)} examples")
print(f"- Regular accuracy gives: {regular_accuracy:.2f}%")
print(f"- Balanced accuracy gives: {balancedAccuracyPercentage:.2f}%")
if balancedAccuracyPercentage < regular_accuracy:
    print(f"- Balanced accuracy is lower because the model is biased toward '{modelAnswerLineYesOrNos[0]}'")


# ModelAnswersLengthInLines = len(modelAnswerLineYesOrNos)
# print(ModelAnswersLengthInLines)
# yesAnswersModifier = (sum([1 if elem == 'T' else 0 for elem in modelAnswerLineYesOrNos]) / ModelAnswersLengthInLines)
# print(yesAnswersModifier)
# noAnswersModifier = (sum([0 if elem == 'T' else 1 for elem in modelAnswerLineYesOrNos]) / ModelAnswersLengthInLines)
# balancedAccuracyPercentage = yesAnswersModifier * noAnswersModifier
# print(balancedAccuracyPercentage)