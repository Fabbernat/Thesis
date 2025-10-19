import torch

from src.Framework.HuggingFaceModelInferencer.ThirdPartyApiHandler.TransformersApiHandler import TransformersApiHandler

print('Is cuda available? ', torch.cuda.is_available())


def writeToFile(modelResponses, fileNameAsString):
    with open(fileNameAsString, 'w') as modelResponsesFile:
        print(modelResponses, file=modelResponsesFile)

class TorchApiHandler:
    def __init__(self):
        print('TorchApiHandler initialized')
        self.transformersTensors = []
        self.convertedTensors = []
        self.tah = None

    def handleRequest(self):
        print('TorchApiHandler.handleRequest() started')
        with torch.no_grad():
            self.tah = TransformersApiHandler()
            self.handleModelSpecificActions()
            modelResponses, generated_ids, tokenizer = self.tah.generateFinalAnswer3()
            print(f'Model\'s responses: {modelResponses} \ngenerated ids: {generated_ids} \ntokenizer: {tokenizer}')

            writeToFile(modelResponses, 'modelResponses.out')
            writeToFile(generated_ids, 'generated_ids.out')
            writeToFile(tokenizer, 'tokenizer.out')
            writeToFile(self.transformersTensors, 'transformersTensors.out')
            writeToFile(self.convertedTensors, 'convertedTensors.out')


        # transformersTensors = TransformersApiHandler().decodeOutputsSkippingSpecialTokens()



    def handleModelSpecificActions(self):
        try:
            from src.Framework.HuggingFaceModelInferencer.Config.Config import MODEL_NAME
            if MODEL_NAME == 'google/gemma-2-2b':
                print(f'Model name is {MODEL_NAME}')
                response = self.tah.tokeniteAutoModelForGoogle0()
                writeToFile(response, 'modelResponses.out')

                return


            else:
                self.tah.tokenizeAutoModelForQwenAndSimilar0()
                self.transformersTensors = self.tah.generateIds1()
                self.convertedTensors = self.tah.convertIds2()
        except Exception as e:
            print(e)