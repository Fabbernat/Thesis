import json
import re


class TernaryClassifier:
    def getYesOrNo(self, modelAnswer):
        return 'T' if re.match(r'Yes', modelAnswer) else 'F' if re.match(r'No', modelAnswer) else '?'


    def classify(self):
        answerCorrectnessValidityFlagsAsBools: list[bool] = []

        with  open("modelAnswers.txt") as modelFile, open("test.gold.txt") as goldFile:
            modelAnswersLines: list[str] = modelFile.readlines()
            goldAnswersLines: list[str] = goldFile.readlines()

            for i in range(len(modelAnswersLines)):
                modelAnswerLine = modelAnswersLines[i].strip()

                from Resurrection.ModelOutputProcessor.main import TESTFILE_LENGTH

                goldAnswerLine = goldAnswersLines[i % TESTFILE_LENGTH].strip()

                print('1', self.getYesOrNo(modelAnswerLine), '2', goldAnswerLine, sep='\n')
                value = (self.getYesOrNo(modelAnswerLine) == goldAnswerLine)

                if not isinstance(value, bool):
                    raise TypeError(f"Only boolean values can be stored in {answerCorrectnessValidityFlagsAsBools}!")

                answerCorrectnessValidityFlagsAsBools.append(value)

        with open('ternaryResults.txt', 'w') as ternaryResultsFile:
            print('\n'.join((str(answer) for answer in answerCorrectnessValidityFlagsAsBools)), file=ternaryResultsFile)