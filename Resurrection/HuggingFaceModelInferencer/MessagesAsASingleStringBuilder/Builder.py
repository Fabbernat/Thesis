import os

from Resurrection.HuggingFaceModelInferencer.Config.Config import FILE_NAME

def getMessagesAsString():
    currentFilesDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    promptPath = os.path.join(currentFilesDir, FILE_NAME)
    print(f'Prompt file path={promptPath}')

    if not os.path.exists(promptPath):
        raise FileNotFoundError(f'Expected prompt file not found at: {promptPath}')

    with open(promptPath, 'r', encoding='utf-8') as promptFile:
        promptFileContents = promptFile.read()

    messages = [
        {"role": "system", "content": "Answer all questions with Yes or No!\n"},
        {"role": "user", "content": promptFileContents},
    ]

    return "".join(messages[0]["content"] + promptFileContents)

if __name__ == "__main__":
    print(getMessagesAsString())

