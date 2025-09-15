import torch

from  Resurrection.CloudRunnerNotebooks.Transformers3rdPartyApiHandler.Transformers3rdPartyApiHandler import Transformers3rdPartyApiHandler

print("Is cuda available?", torch.cuda.is_available())

def run():
    with torch.no_grad():
        print('torch.no_grad()')
        try:
            Transformers3rdPartyApiHandler().generateAnswers()
        except Exception as e:
            print(e)

    answer = Transformers3rdPartyApiHandler().decodeOutputsSkippingSpecialTokens()

    with open("modelAnswers.out", "w") as modelAnswersFile:
        print(answer, file=modelAnswersFile)
        # print((line for line in answer), file=modelAnswersFile)