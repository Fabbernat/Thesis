import torch

from Resurrection.CloudRunnerNotebooks.Transformers3rdPartyApiHandler.Transformers3rdPartyApiHandler import \
    Transformers3rdPartyApiHandler

print("Is cuda available?", torch.cuda.is_available())

MODEL_NAME =  '      https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct           '



def run():

    messages = [
        {"role": "system", "content": "Answer all questions with Yes or No!"},
        {"role": "user", "content": open("prompt.txt").read()},
    ]



    with torch.no_grad():
        outputs = Transformers3rdPartyApiHandler().generateAnswersFor(messages)

    answer = Transformers3rdPartyApiHandler().decodeOutputsSkippingSpecialTokens()

    with open("modelAnswers.out", "w") as modelAnswersFile:
        print(answer, file=modelAnswersFile)
        # print((line for line in answer), file=modelAnswersFile)