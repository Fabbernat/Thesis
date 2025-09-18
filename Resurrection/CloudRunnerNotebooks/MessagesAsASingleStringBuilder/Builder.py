import os.path

from Resurrection.CloudRunnerNotebooks.Config.Config import FILE_PATH

def getMessagesAsString():
    promptFileContents = open(os.path.join(FILE_PATH)).read()
    messages = [
        {"role": "system", "content": "Answer all questions with Yes or No!\n"},
        {"role": "user", "content": promptFileContents},
    ]

    return "".join(messages[0]["content"] + promptFileContents)

def tryToOpenAFileFromHere():
    with open("../prompt.txt") as f:
        print(f.read())

print(getMessagesAsString())