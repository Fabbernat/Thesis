import os

from Resurrection.CloudRunnerNotebooks.Config.Config import FILE_PATH

def getMessagesAsString():
    currentFilesDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    promptPath = os.path.join(currentFilesDir, 'prompt.txt')

    if not os.path.exists(promptPath):
        raise FileNotFoundError(f'Expected prompt file not found at: {promptPath}')

    with open(promptPath, 'r', encoding='utf-8') as promptFile:
        promptFileContents = promptFile.read()

    messages = [
        {"role": "system", "content": "Answer all questions with Yes or No!\n"},
        {"role": "user", "content": promptFileContents},
    ]

    return "".join(messages[0]["content"] + promptFileContents)

def tryToOpenAFileFromHere():
    with open("../prompt.txt") as f:
        print(f.read())

if __name__ == "__main__":
    print(getMessagesAsString())

