import torch

from  Resurrection.CloudRunnerNotebooks.ThirdPartyApiHandler.TransformersApiHandler import TransformersApiHandler

print("Is cuda available?", torch.cuda.is_available())

class TorchApiHandler:
    def __init__(self):
        transformersTensors = None
        with torch.no_grad():
            print('torch.no_grad()')
            tah = TransformersApiHandler()
            try:
                transformersTensors = tah.generateIds1()
            except Exception as e:
                print(e)
            try:
                convertedTensors = tah.convertIds2()
            except Exception as e:
                print(e)
            try:
                modelAnswers = tah.generateFinalAnswer3()
            except Exception as e:
                print(e)



        # transformersTensors = TransformersApiHandler().decodeOutputsSkippingSpecialTokens()

        with open("transformersTensors.out", "w") as generatedTensorsFile:
            print(transformersTensors, file=generatedTensorsFile)
            # print((line for line in transformersTensors), file=transformersTensorsFile)

        with open("convertedTensors.out", "w") as convertedTensorsFile:
            print(convertedTensors, file=convertedTensorsFile)

        with open("modelAnswers.out", "w") as modelAnswersFile:
            print(modelAnswers, file=modelAnswersFile)