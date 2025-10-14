import torch

from  Resurrection.HuggingFaceModelInferencer.ThirdPartyApiHandler.TransformersApiHandler import TransformersApiHandler

print("Is cuda available? ", torch.cuda.is_available())


def writeToFile(modelResponses):
    with open("modelResponses.out", "w") as modelResponsesFile:
        print(modelResponses, file=modelResponsesFile)

class TorchApiHandler:
    def __init__(self):
        print('TorchApiHandler initialized')
        self.transformersTensors = []
        self.convertedTensors = []
        self.modelResponses = ""
        self.tah = None

    def handleRequest(self):
        print('TorchApiHandler.handleRequest() started')
        with torch.no_grad():
            self.tah = TransformersApiHandler()
            self.handleModelSpecificActions()
            self.modelResponses, generated_ids, tokenizer = self.tah.generateFinalAnswer3()
            print(f'Model\'s responses: {self.modelResponses} \ngenerated ids: {generated_ids} \ntokenizer: {tokenizer}')



        # transformersTensors = TransformersApiHandler().decodeOutputsSkippingSpecialTokens()

        with open("transformersTensors.out", "w") as generatedTensorsFile:
            print(self.transformersTensors, file=generatedTensorsFile)
            # print((line for line in transformersTensors), file=transformersTensorsFile)

        with open("convertedTensors.out", "w") as convertedTensorsFile:
            print(self.convertedTensors, file=convertedTensorsFile)

        writeToFile(self.modelResponses)

    def handleModelSpecificActions(self):
        try:
            from Resurrection.HuggingFaceModelInferencer.Config.Config import MODEL_NAME
            if MODEL_NAME == 'google/gemma-2-2b':
                print(f'Model name is {MODEL_NAME}')
                response = self.tah.tokeniteAutoModelForGoogle0()
                writeToFile(response)

                return


            else:
                self.tah.tokenizeAutoModelForQwenAndSimilar0()
                self.transformersTensors = self.tah.generateIds1()
                self.convertedTensors = self.tah.convertIds2()
        except Exception as e:
            print(e)