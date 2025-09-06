class ShouldCategorizeException(Exception):
    pass



class TernaryClassifier:
    def __init__(self):
        self.TruePositives = 0
        self.FalsePositives = 0
        self.FalseNegatives = 0
        self.TrueNegatives = 0


    def classify(self) -> None:
        answerCorrectnessValidityFlagsAsBools: list[bool] = []
        confusionMatrixValuse: list[str] = []

        with  open("modelAnswers.txt") as modelFile, open("test.gold.txt") as goldFile:
            modelAnswersLines: list[str] = modelFile.readlines()
            goldAnswersLines: list[str] = goldFile.readlines()

            for i in range(len(modelAnswersLines)):
                modelAnswerLine = modelAnswersLines[i].strip()

                from Resurrection.ModelOutputProcessor.main import TESTFILE_LENGTH

                goldAnswerLine = goldAnswersLines[i % TESTFILE_LENGTH].strip()

                modelAnswerLineYesOrNo = self.getYesOrNo(modelAnswerLine)
                # print(f'{i}:{self.getYesOrNo(modelAnswerLine)}\n{i}:{goldAnswerLine}')
                print("COMPARE2")
                print(modelAnswerLineYesOrNo)
                print(goldAnswerLine)
                value = (modelAnswerLineYesOrNo == goldAnswerLine)




                if not isinstance(value, bool):
                    raise TypeError(f"Only boolean values can be stored in {answerCorrectnessValidityFlagsAsBools}!")

                answerCorrectnessValidityFlagsAsBools.append(value)
                confusionMatrixValuse.append(self.categorize(modelAnswerLineYesOrNo, goldAnswerLine))

        with open('ternaryResults.txt', 'w') as ternaryResultsFile, open('confusionMatrix.txt', 'w') as confusionMatrixFile:
            print('\n'.join((str(answer) for answer in answerCorrectnessValidityFlagsAsBools)), file=ternaryResultsFile)
            print('\n'.join((str(answer) for answer in confusionMatrixValuse)), file=confusionMatrixFile)


    def categorize(self, modelAnswerLineYesOrNo, goldAnswerLine):
        if modelAnswerLineYesOrNo == 'T' and goldAnswerLine == 'T':
            self.TruePositives += 1
            return 'TP'
        if modelAnswerLineYesOrNo == 'T' and goldAnswerLine == 'F':
            self.FalsePositives += 1
            return "FP"
        if modelAnswerLineYesOrNo == 'F' and goldAnswerLine == 'T':
            self.FalseNegatives += 1
            return "FN"
        if modelAnswerLineYesOrNo == 'F' and goldAnswerLine == 'F':
            self.TrueNegatives += 1
            return "TN"
        else:
            raise ShouldCategorizeException("This should never happen")

    def getYesOrNo(self, modelAnswer: str) -> str:
        return self.classifySentence(modelAnswer)

    def classifySentence(self, LinebreaklessString: str) -> str:
        """
        function that gets a LinebreaklessString as input and may output 3 different characters based on the sentence: - 'T' if the LinebreaklessString contains the word "Yes" (case-sensitive), or an affirmative message. - 'F' if the LinebreaklessString contains the word "No" (case-sensitive), or a not affirmative message. - '?' in any other cases, where the intent of the sentence is unclear.
        Ez a legjobb ötletem a modell intenciójának az eldöntésére, de ez biztosan nem osztályozza be a szándékokat 100%-os pontossággal
        :return:
        """
        #TODO mivan ha "Yes and No" a válasz, vagy "eyes"?
        text = LinebreaklessString.strip()

        import main
        affirmativeKeywords = main.AFFIRMATIVE_KEYWORDS
        negativeKeywords = main.NEGATIVE_KEYWORDS

        if any(word.lower() in text.lower() for word in affirmativeKeywords):
            return 'T'
        if any(word.lower() in text.lower() for word in negativeKeywords):
            return 'F'
        return '?'
