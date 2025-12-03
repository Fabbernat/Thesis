from typing import Any
from pathlib import Path


try:
    from src.Framework.ModelOutputProcessor.TernaryClassifier.AnswerAwareClassificationRule.AnswerAwareClassificationRule import setAnswerAwareClassificationRule
    from src.Framework.ModelOutputProcessor.TernaryClassifier.SentenceClassifier.SentenceClassifier import classifySentence
    from src.Framework.ModelOutputProcessor.config import MODEL_PATH, GOLD_PATH, ADAPTIVE_RUN, GoldFileLengthInLines, ModelAnswersLengthInLines
except Exception:
    from TernaryClassifier.AnswerAwareClassificationRule.AnswerAwareClassificationRule import setAnswerAwareClassificationRule
    from TernaryClassifier.SentenceClassifier.SentenceClassifier import classifySentence
    from config import MODEL_PATH, GOLD_PATH, ADAPTIVE_RUN, GoldFileLengthInLines, ModelAnswersLengthInLines


def getYesOrNo(modelAnswer: str) -> str:
    if ADAPTIVE_RUN:
        return setAnswerAwareClassificationRule(modelAnswer)
    else:
        return classifySentence(modelAnswer)


class TernaryClassifier:
    modelAnswersLengthInLines = ModelAnswersLengthInLines

    def __init__(self):
        self.TruePositives = 0
        self.FalsePositives = 0
        self.FalseNegatives = 0
        self.TrueNegatives = 0


    def classify0(self) -> Any:
        answerCorrectnessValidityFlagsAsBools: list[bool] = [] # TODO global
        modelAnswerLineTsOrFs = []
        goldAnswerLineTsOrFs = []
        confusionMatrixValues: list[str] = []

        basePath = Path(__file__).parent.parent
        with  open(Path(str(basePath) + MODEL_PATH)) as modelFile, open(Path(str(basePath) + GOLD_PATH)) as goldFile:
            modelAnswersLines: list[str] = modelFile.readlines()
            goldAnswersLines: list[str] = goldFile.readlines()

            modelAnswersLengthInLines: int = len(modelAnswersLines)
            self.modelAnswersLengthInLines = modelAnswersLengthInLines
            if modelAnswersLengthInLines % 2 != 0:
                key = input(
                    'Warning: cannot count consistency when odd number of lines, please fix input. The last line  will be dropped. Do you wish to continue? (y/n)')

                if key == 'n':
                    exit(0)
                modelAnswersLengthToProcess = modelAnswersLengthInLines - 1
            else:
                modelAnswersLengthToProcess = modelAnswersLengthInLines


            for i in range(modelAnswersLengthToProcess):
                modelAnswerLine: str = modelAnswersLines[i].strip()

                goldAnswerLineTOrF: str = goldAnswersLines[i % GoldFileLengthInLines].strip() # need to reset at half, because `modelAnswersLengthInLines` is about twice as long as `GoldFileLengthInLines`

                modelAnswerLineTOrF = getYesOrNo(modelAnswerLine) # returns `T`, `F` or `?`
                modelAnswerLineTsOrFs.append(modelAnswerLineTOrF)
                goldAnswerLineTsOrFs.append(goldAnswerLineTOrF)

                print(f'Comparing {i + 1}th line:')
                print(modelAnswerLineTOrF)
                print(goldAnswerLineTOrF)
                isEqual = (modelAnswerLineTOrF == goldAnswerLineTOrF)

                if not isEqual:
                    print(f'MISTAKE IN LINE {i + 1}! Model falsely predicted {modelAnswerLineTOrF} instead of {goldAnswerLineTOrF}')


                if not isinstance(isEqual, bool):
                    raise TypeError(f'Only boolean values can be stored in answerCorrectnessValidityFlagsAsBools!')

                answerCorrectnessValidityFlagsAsBools.append(isEqual)
                confusionMatrixValues.append(self.categorize(modelAnswerLineTOrF, goldAnswerLineTOrF))

        with open(Path(str(basePath) + r'\data\ternaryResults.out'), 'w') as ternaryResultsFile, open(Path(str(basePath) + r'\data\confusionMatrix.out'), 'w') as confusionMatrixFile:
            print(ternaryResultsFile, confusionMatrixFile)
            print('\n'.join((str(answer) for answer in answerCorrectnessValidityFlagsAsBools)), file=ternaryResultsFile)
            print('\n'.join((str(answer) for answer in confusionMatrixValues)), file=confusionMatrixFile)


        return modelAnswerLineTsOrFs, goldAnswerLineTsOrFs


    def categorize(self, modelAnswerLineYesOrNo, goldAnswerLine):
        if modelAnswerLineYesOrNo == 'T' and goldAnswerLine == 'T':
            self.TruePositives += 1
            return 'TP'
        elif modelAnswerLineYesOrNo == 'T' and goldAnswerLine == 'F':
            self.FalsePositives += 1
            return 'FP'
        elif modelAnswerLineYesOrNo == 'F' and goldAnswerLine == 'T':
            self.FalseNegatives += 1
            return 'FN'
        elif modelAnswerLineYesOrNo == 'F' and goldAnswerLine == 'F':
            self.TrueNegatives += 1
            return 'TN'
        else:
            return '?'


    def getTruePositives(self):
        return self.TruePositives

    def getTrueNegatives(self):
        return self.TrueNegatives

    def getFalseNegatives(self):
        return self.FalseNegatives

    def getFalsePositives(self):
        return self.FalsePositives