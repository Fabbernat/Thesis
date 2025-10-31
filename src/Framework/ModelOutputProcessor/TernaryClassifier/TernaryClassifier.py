from typing import Any

from Framework.ModelOutputProcessor.config import MODEL_PATH, GOLD_PATH


class ShouldCategorizeException(Exception):
    pass



class TernaryClassifier:
    NUMBER_OF_LINES = 1

    def __init__(self):
        self.TruePositives = 0
        self.FalsePositives = 0
        self.FalseNegatives = 0
        self.TrueNegatives = 0


    def classify(self) -> Any:
        answerCorrectnessValidityFlagsAsBools: list[bool] = []
        confusionMatrixValues: list[str] = []

        with  open(MODEL_PATH) as modelFile, open(GOLD_PATH) as goldFile:
            modelAnswersLines: list[str] = modelFile.readlines()
            goldAnswersLines: list[str] = goldFile.readlines()

            self.NUMBER_OF_LINES = len(modelAnswersLines)
            for i in range(self.NUMBER_OF_LINES):
                modelAnswerLine = modelAnswersLines[i].strip()

                from src.Framework.ModelOutputProcessor.main import TESTFILE_LENGTH

                goldAnswerLine = goldAnswersLines[i % TESTFILE_LENGTH].strip()

                modelAnswerLineYesOrNo = self.getYesOrNo(modelAnswerLine)
                # print(f'{i}:{self.getYesOrNo(modelAnswerLine)}\n{i}:{goldAnswerLine}')
                print("COMPARE2")
                print(modelAnswerLineYesOrNo)
                print(goldAnswerLine)
                value = (modelAnswerLineYesOrNo == goldAnswerLine)

                if not value:
                    print(f'MISTAKE IN LINE {i + 1}! {modelAnswerLineYesOrNo} instead of {goldAnswerLine}')


                if not isinstance(value, bool):
                    raise TypeError(f"Only boolean values can be stored in {answerCorrectnessValidityFlagsAsBools}!")

                answerCorrectnessValidityFlagsAsBools.append(value)
                confusionMatrixValues.append(self.categorize(modelAnswerLineYesOrNo, goldAnswerLine))

        with open('data/ternaryResults.out', 'w') as ternaryResultsFile, open('data/confusionMatrix.out', 'w') as confusionMatrixFile:
            print('\n'.join((str(answer) for answer in answerCorrectnessValidityFlagsAsBools)), file=ternaryResultsFile)
            print('\n'.join((str(answer) for answer in confusionMatrixValues)), file=confusionMatrixFile)




    def getYesOrNo(self, modelAnswer: str) -> str:
        return self.classifySentence(modelAnswer)

    def classifySentence(self, linebreaklessSentence: str, phrases=False) -> str:
        """
        function that gets a linebreaklessSentence as input and may output 3 different characters based on the sentence: - 'T' if the linebreaklessSentence contains the word "Yes" (case-sensitive), or an affirmative message. - 'F' if the linebreaklessSentence contains the word "No" (case-sensitive), or a not affirmative message. - '?' in any other cases, where the intent of the sentence is unclear.
        Ez a legjobb ötletem a modell biasának az eldöntésére, de ez biztosan nem osztályozza be a szándékokat 100%-os pontossággal
        :return:
        """
        #TODO mivan ha "Yes and No" a válasz, vagy "eyes"?
        sentence = linebreaklessSentence.strip()
        if sentence == 'T' or sentence == 'Yes' or sentence == 'Yes.':
            return 'T'
        if sentence == 'F' or sentence == 'No' or sentence == 'No.':
            return 'F'

        if phrases:
            return self.classifyByPhrases(sentence)
        else:
            return self.classifyByKeywords(sentence)


    def classifyByPhrases(self, sentence: str) -> str:
        from src.Framework.ModelOutputProcessor.main import AFFIRMATIVE_PHRASES, NEGATIVE_PHRASES

        affirmativePhrases = AFFIRMATIVE_PHRASES
        negativePhrases = NEGATIVE_PHRASES

        if any(affirmativePhrases) in sentence:
            return 'T'
        if any(negativePhrases) in sentence:
            return 'F'
        return '?'

    def classifyByKeywords(self, sentence: str) -> str:
        from src.Framework.ModelOutputProcessor.main import AFFIRMATIVE_KEYWORDS, NEGATIVE_KEYWORDS
        affirmativeKeywords = AFFIRMATIVE_KEYWORDS
        negativeKeywords = NEGATIVE_KEYWORDS

        if any(word.lower() in sentence.lower() for word in affirmativeKeywords):
            return 'T'
        if any(word.lower() in sentence.lower() for word in negativeKeywords):
            return 'F'
        return '?'



    def categorize(self, modelAnswerLineYesOrNo, goldAnswerLine):
        if modelAnswerLineYesOrNo == 'T' and goldAnswerLine == 'T':
            self.TruePositives += 1
            return 'TP'
        elif modelAnswerLineYesOrNo == 'T' and goldAnswerLine == 'F':
            self.FalsePositives += 1
            return "FP"
        elif modelAnswerLineYesOrNo == 'F' and goldAnswerLine == 'T':
            self.FalseNegatives += 1
            return "FN"
        elif modelAnswerLineYesOrNo == 'F' and goldAnswerLine == 'F':
            self.TrueNegatives += 1
            return "TN"
        else:
            return "?"


    def getTruePositives(self):
        return self.TruePositives

    def getTrueNegatives(self):
        return self.TrueNegatives

    def getFalseNegatives(self):
        return self.FalseNegatives

    def getFalsePositives(self):
        return self.FalsePositives