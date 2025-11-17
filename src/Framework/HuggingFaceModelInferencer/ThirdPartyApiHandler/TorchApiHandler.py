import torch
import sys
from pathlib import Path

try:
    from src.Framework.HuggingFaceModelInferencer.ThirdPartyApiHandler.TransformersApiHandler import TransformersApiHandler
    from src.Framework.HuggingFaceModelInferencer.modelname import MODEL_NAME
except Exception:
    from ThirdPartyApiHandler.TransformersApiHandler import TransformersApiHandler
    from modelname import MODEL_NAME

print('Is cuda available? ', torch.cuda.is_available())
basePath = Path(__file__).parent / "data"

class TorchApiHandler:
    def __init__(self):
        print('TorchApiHandler initialized')
        self.generatedIdsTransformersTensors = []
        self.convertedIdsTensors = []
        self.transformersApiHandler = None

    def handleRequest(self):
        print('TorchApiHandler.handleRequest() started')
        with torch.no_grad():
            self.transformersApiHandler = TransformersApiHandler()

            self.transformersApiHandler.DoAutotokenizerFromPretrained()

            self.handleModelSpecificActions() # This takes up most of the runtime.


            response, generatedIds, tokenizer = self.transformersApiHandler.batchDecodeGenerateFinalAnswer3(self.convertedIdsTensors)
            print(f'Model\'s responses: {response} \ngenerated ids: {generatedIds} \ntokenizer: {tokenizer}')

            basePath = Path(__file__).parent
            writeToFile(generatedIds, basePath / "data" / "generatedIds.out")
            writeToFile(tokenizer, basePath / "data" / "tokenizer.out")
            writeToFile(self.generatedIdsTransformersTensors, basePath / "data" / "generatedIdsTransformersTensors.out")
            writeToFile(self.convertedIdsTensors, basePath / "data" / "convertedIdsTensors.out")

            saveOutput(basePath, response)



    def handleModelSpecificActions(self):
        try:
            if MODEL_NAME.startswith('qwen'):
                self.generatedIdsTransformersTensors, self.convertedIdsTensors = self.transformersApiHandler.qwen()
            elif MODEL_NAME.startswith('google'):
                print(f'Model name is {MODEL_NAME}')
                response = self.transformersApiHandler.google()
                writeToFile(response, Path(__file__).parent / "data" / "modelResponses.out")

            elif MODEL_NAME.startswith('microsoft'):
                pass
            else:
                raise NotImplementedError
    
    
        except Exception as e:
            print('Exception in handleModelSpecificActions:', e)


def saveOutput(basePath, results: str):

    print('basePath: ', basePath)
    fullPath = basePath / "data" / "modelResponses.out"
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
            modelResponsesFile.write(str(modelResponses))
        print(f'Successfully written to {fileNameAsPath}')
    except OSError as oe:
        sys.stderr.write(f'File writing error ({fileNameAsPath}): {oe}\n')
    except ValueError as ve:
        sys.stderr.write(f'Value error while writing to {fileNameAsPath}: {ve}\n')


