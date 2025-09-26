import torch

from  Resurrection.HuggingFaceModelInferencer.ThirdPartyApiHandler.TransformersApiHandler import TransformersApiHandler

print("Is cuda available?", torch.cuda.is_available())

class TorchApiHandler:
    def __init__(self):
        transformersTensors = None
        modelAnswers = None
        with torch.no_grad():
            print('torch.no_grad()')
            tah = TransformersApiHandler()
            try:
                tah.tokenizeAutoModel0()
                transformersTensors = tah.generateIds1()
                convertedTensors = tah.convertIds2()
            except Exception as e:
                print(e)
            modelAnswers, generated_ids, tokenizer = tah.generateFinalAnswer3()

            print(f'Model\'s answers: {modelAnswers} \ngenerated ids: {generated_ids} \ntokenizer: {tokenizer}')



        # transformersTensors = TransformersApiHandler().decodeOutputsSkippingSpecialTokens()

        with open("transformersTensors.out", "w") as generatedTensorsFile:
            print(transformersTensors, file=generatedTensorsFile)
            # print((line for line in transformersTensors), file=transformersTensorsFile)

        with open("convertedTensors.out", "w") as convertedTensorsFile:
            print(convertedTensors, file=convertedTensorsFile)

        with open("modelAnswers.out", "w") as modelAnswersFile:
            print(modelAnswers, file=modelAnswersFile)