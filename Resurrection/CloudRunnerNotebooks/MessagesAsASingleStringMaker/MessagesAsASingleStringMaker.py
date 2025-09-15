from Resurrection.CloudRunnerNotebooks.main import FILE_PATH

def getMessagesAsString():
    promptFileContents = open(FILE_PATH).read()
    messages = [
        {"role": "system", "content": "Answer all questions with Yes or No!"},
        {"role": "user", "content": promptFileContents},
    ]

    return "".join(messages[0]["content"] + promptFileContents)

def tryToOpenAFileFromHere():
    with open("../prompt.txt") as f:
        print(f.read())