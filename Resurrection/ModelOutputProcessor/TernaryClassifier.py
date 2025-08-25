import json
import re


class TernaryClassifier:

    def classify(self):
        answerCorrectnessValidityFlagsAsBools: list[bool] = []

        with  open("modelAnswers.txt") as modelFile, open("test.gold.txt") as goldFile:
            modelAnswersLines: list[str] = modelFile.readlines()
            goldAnswersLines: list[str] = goldFile.readlines()

            for i in range(len(modelAnswersLines)):
                modelAnswerLine = modelAnswersLines[i].strip()

                from Resurrection.ModelOutputProcessor.main import TESTFILE_LENGTH

                goldAnswerLine = goldAnswersLines[i % TESTFILE_LENGTH].strip()

                print(f'{i}:{self.getYesOrNo(modelAnswerLine)}\n{i}:{goldAnswerLine}')
                value = (self.getYesOrNo(modelAnswerLine) == goldAnswerLine)

                if not isinstance(value, bool):
                    raise TypeError(f"Only boolean values can be stored in {answerCorrectnessValidityFlagsAsBools}!")

                answerCorrectnessValidityFlagsAsBools.append(value)

        with open('ternaryResults.txt', 'w') as ternaryResultsFile:
            print('\n'.join((str(answer) for answer in answerCorrectnessValidityFlagsAsBools)), file=ternaryResultsFile)

    def getYesOrNo(self, modelAnswer):
        return self.classifySentence(modelAnswer)

    def classifySentence(self, LinebreaklessString: str) -> str:
        """
        function that gets a LinebreaklessString as input and may output 3 different characters based on the sentence: - 'T' if the LinebreaklessString contains the word "Yes" (case-sensitive), or an affirmative message. - 'F' if the LinebreaklessString contains the word "No" (case-sensitive), or a not affirmative message. - '?' in any other cases, where the intent of the sentence is unclear.
        Ez a legjobb ötletem a modell intenciójának az eldöntésére, de ez biztosan nem osztályozza be a szándékokat 100%-os pontossággal
        :return:
        """
        text = LinebreaklessString.strip()
        affirmativeKeywords = {"Yes", "Yes!", "Yes.", "Yes,", "->Yes"}
        negativeKeywords = {"No", "No!", "No.", "No,", "->No", "Never", "Not at all", "Nope", "Nah"}

        if any(word.lower() in text.lower() for word in affirmativeKeywords):
            return 'T'
        if any(word.lower() in text.lower() for word in negativeKeywords):
            return 'F'
        return '?'
