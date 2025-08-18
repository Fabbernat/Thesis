import json
from typing import List
import re


def getYesOrNo(modelAnswer):
    return 'T' if re.match(r'Yes', modelAnswer) else 'F' if re.match(r'No', modelAnswer) else '?'


def run():
    answerCorrectnessValidityFlagsAsBools: List[bool] = []

    with  open("modelAnswers.txt") as modelFile, open("test.gold.txt") as goldFile:
        modelAnswersAsString: List[str] = modelFile.readlines()
        goldAnswersAsString: List[str] = goldFile.readlines()

        for i in range(len(modelAnswersAsString)):
            modelAnswer = modelAnswersAsString[i].strip()
            goldAnswer = goldAnswersAsString[i % 2800].strip()

            value = (getYesOrNo(modelAnswer) == goldAnswer)

            if not isinstance(value, bool):
                raise TypeError(f"Only boolean values can be stored in {answerCorrectnessValidityFlagsAsBools}!")

            answerCorrectnessValidityFlagsAsBools.append(value)

    with open('data.json') as dataJson:
        print(json.loads('\n'.join(str(answerCorrectnessValidityFlagsAsBools))), file=dataJson)