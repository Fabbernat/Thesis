import torch

from  Resurrection.CloudRunnerNotebooks.ThirdPartyApiHandler.TransformersApiHandler import TransformersApiHandler

print("Is cuda available?", torch.cuda.is_available())

class TorchApiHandler:
    def __init__(self):
        answer = None
        with torch.no_grad():
            print('torch.no_grad()')
            try:
                answer = TransformersApiHandler().generateIds1()
            except Exception as e:
                print(e)

        # answer = TransformersApiHandler().decodeOutputsSkippingSpecialTokens()

        with open("modelAnswers.out", "w") as modelAnswersFile:
            print(answer, file=modelAnswersFile)
            # print((line for line in answer), file=modelAnswersFile)