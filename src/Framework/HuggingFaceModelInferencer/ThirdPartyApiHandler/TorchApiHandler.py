import torch
import sys

from transformers import MODEL_NAMES_MAPPING

from src.Framework.HuggingFaceModelInferencer.ThirdPartyApiHandler.TransformersApiHandler import TransformersApiHandler
from src.Framework.HuggingFaceModelInferencer.main import MODEL_NAME

print('Is cuda available? ', torch.cuda.is_available())


def writeToFile(modelResponses, fileNameAsString):
    with open(fileNameAsString, 'w') as modelResponsesFile:
        print(modelResponses, file=modelResponsesFile)

class TorchApiHandler:
    def __init__(self):
        print('TorchApiHandler initialized')
        self.transformersTensors = []
        self.convertedTensors = []
        self.transformersApiHandler = None

    def handleRequest(self):
        print('TorchApiHandler.handleRequest() started')
        with torch.no_grad():
            self.transformersApiHandler = TransformersApiHandler()
            self.handleModelSpecificActions()
            modelResponses, generated_ids, tokenizer = self.transformersApiHandler.generateFinalAnswer3()
            print(f'Model\'s responses: {modelResponses} \ngenerated ids: {generated_ids} \ntokenizer: {tokenizer}')

            writeToFile(modelResponses, 'modelResponses.out')
            writeToFile(generated_ids, 'generatedIds.out')
            writeToFile(tokenizer, 'tokenizer.out')
            writeToFile(self.transformersTensors, 'transformersTensors.out')
            writeToFile(self.convertedTensors, 'convertedTensors.out')


        # transformersTensors = TransformersApiHandler().decodeOutputsSkippingSpecialTokens()



    def handleModelSpecificActions(self):
        try:
            if MODEL_NAME == 'google/gemma-2-2b':
                print(f'Model name is {MODEL_NAME}')
                response = self.transformersApiHandler.google()
                writeToFile(response, 'modelResponses.out')

                return


            else:
                self.transformersTensors, self.convertedTensors = self.transformersApiHandler.qwen()
        except Exception as e:
            raise e
            print('Exception in handleModelSpecificActions:', e)