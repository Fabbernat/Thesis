import torch
import sys
from pathlib import Path
import random
import numpy as np


try:
    from src.Framework.HuggingFaceModelInferencer.ThirdPartyApiHandler.TransformersApiHandler import TransformersApiHandler
    from src.Framework.HuggingFaceModelInferencer.modelname import MODEL_NAME
    from src.Framework.HuggingFaceModelInferencer.config import NUMBER_OF_DESIRED_ANSWERS
    from src.Framework.HuggingFaceModelInferencer.config import DETERMINISTIC_MODE

except Exception:
    from .TransformersApiHandler import TransformersApiHandler
    from modelname import MODEL_NAME
    from config import NUMBER_OF_DESIRED_ANSWERS
    from config import DETERMINISTIC_MODE

print('Is cuda available? ', torch.cuda.is_available())

class TorchApiHandler:
    def __init__(self):
        print('TorchApiHandler initialized')
        self.responses = []
        self.generatedIds = []
        self.transformersApiHandler = None

    def handleRequest(self, questions):
        print('TorchApiHandler.handleRequest() started')
        with torch.no_grad():
            self.transformersApiHandler = TransformersApiHandler()

            for i in range(min(NUMBER_OF_DESIRED_ANSWERS, len(questions))):

                self.transformersApiHandler.DoAutotokenizerFromPretrained()

                response = self.handleModelSpecificActions(questions) # This takes up most of the runtime.


                #  This line uses a generator expression. batchDecodeGenerateFinalAnswer prints convertedTensors and then calls self.tokenizer.batch_decode(convertedTensors, ...). Pass a list instead
                # response, generatedIds, tokenizer = self.transformersApiHandler.batchDecodeGenerateFinalAnswer(elem for elem in self.convertedIdsTensorsList)
                # response, generatedIds, tokenizer = self.transformersApiHandler.batchDecodeGenerateFinalAnswer(list(self.convertedIdsTensorsList))


                # print(f'Model\'s responses: {response} \ngenerated ids: {generatedIds} \ntokenizer: {tokenizer}')
                # response = ' '.join([str(elem) for elem in response])
                #
                basePath = Path(__file__).parent
                # writeToFile(generatedIds, basePath / "data" / "generatedIds.out")
                # writeToFile(tokenizer, basePath / "data" / "tokenizer.out")
                # writeToFile(self.generatedIdsTransformersTensorsList, basePath / "data" / "generatedIdsTransformersTensors.out")
                # writeToFile(self.convertedIdsTensorsList, basePath / "data" / "convertedIdsTensors.out")
                #
                saveOutput(basePath, response)



    def handleModelSpecificActions(self, questions):
        if DETERMINISTIC_MODE:
            set_seed(42)
        try:
            if MODEL_NAME.startswith('qwen'):
                response, generatedIds  = self.transformersApiHandler.qwen(questions)
                self.responses.append(response)
                self.generatedIds.append(generatedIds)
            elif MODEL_NAME.startswith('google'):
                print(f'Model name is {MODEL_NAME}')
                response, _ = self.transformersApiHandler.google()
                writeToFile(response, Path(__file__).parent / "data" / "modelResponses.out")

            elif MODEL_NAME.startswith('microsoft'):
                pass
            else:
                raise NotImplementedError('Not implemented yet.')

        except Exception as e:
            print('Exception in handleModelSpecificActions:', e)

        print('handleModelSpecificActions is returning responses', self.responses)
        return self.responses

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def saveOutput(basePath, results: str):

    print('basePath: ', basePath)
    safe_name = MODEL_NAME.split("/")[-1]
    fullPath = basePath.parent / "data" / f"modelResponses{safe_name}.out"
    print('fullPath: ', fullPath)
    secondaryPath = basePath.parent.parent / "ModelOutputProcessor" / "data" / "modelResponses.in"


    # Always write the base file
    writeToFile(results, fullPath)

    # Ask user if secondary output should also be saved
    confirmation = input(
        'Program successfully ran.\n'
        'Do you also want to store the result as the next module\'s input? (y/n): '
    ).strip().lower()

    if confirmation == 'y':
        writeToFile(results, secondaryPath)


def writeToFile(modelResponses, fileNameAsPath: Path):
    try:
        with open(fileNameAsPath, 'w') as modelResponsesFile:
            modelResponsesFile.write(str(modelResponses).replace('\\n', '\n'))
        print(f'Successfully written to {fileNameAsPath}')
    except OSError as oe:
        sys.stderr.write(f'File writing error ({fileNameAsPath}): {oe}\n')
    except ValueError as ve:
        sys.stderr.write(f'Value error while writing to {fileNameAsPath}: {ve}\n')


