from itertools import count

try:
    from src.Framework.ModelOutputProcessor.config import ModelAnswersLengthInLines
except Exception:
    from config import ModelAnswersLengthInLines

modelAnswerLineYesOrNos = ['T', 'F', '?']

yesAnswersModifier = (count([elem == 'T' for elem in modelAnswerLineYesOrNos]) / ModelAnswersLengthInLines)
noAnswersModifier = 1
balancedAccuracyPercentage = yesAnswersModifier * noAnswersModifier


