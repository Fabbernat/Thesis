import torch

from  Resurrection.CloudRunnerNotebooks.Transformers3rdPartyApiHandler import Transformers3rdPartyApiHandler

print("Is cuda available?", torch.cuda.is_available())

MODEL_NAME =  '      https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct           '
MODEL_NAME = MODEL_NAME.strip()


def run():
    with torch.no_grad():
        Transformers3rdPartyApiHandler().generateAnswers()

    answer = Transformers3rdPartyApiHandler().decodeOutputsSkippingSpecialTokens()

    with open("modelAnswers.out", "w") as modelAnswersFile:
        print(answer, file=modelAnswersFile)
        # print((line for line in answer), file=modelAnswersFile)