import os
from itertools import islice

from src.Framework.HuggingFaceModelInferencer.config import FILE_NAME, INSTRUCTION


def getMessagesAsString(numberOfLines=None):
    currentFilesDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    questionsPath = os.path.join(currentFilesDir, FILE_NAME)
    print(f'Questions file path={questionsPath}')

    if not os.path.exists(questionsPath):
        raise FileNotFoundError(f'Expected questions file not found at: {questionsPath}')

    questionsFileContents = ''
    with open(questionsPath, 'r', encoding='utf-8') as questionsFile:
        if numberOfLines is None:
            questionsFileContents = questionsFile.read()
        else:
            questionsFileContents = ''.join(islice(questionsFile, numberOfLines))
            print('reading:', questionsFileContents)

    messages = [
        {'role': 'system', 'content': INSTRUCTION},
        {'role': 'user', 'content': questionsFileContents},
    ]
    messagesAsStr = messages[0]['content']+'\n---------------\n'+ messages[1]['content']

    log = '\n *** The prompt: *** \n'+ messagesAsStr+ '\n *** End of the prompt *** \n'
    print(log)

    print("Writing prompt to:", os.path.abspath('data/prompt.out'))
    try:
        os.makedirs('../data', exist_ok=True)
        with open(os.path.abspath('data/prompt.out'), 'w') as promptFile:
            print(log, file=promptFile)
    except Exception as e:
        print('Failed to save the prompt to data/prompt.out:', e)
    return messages

if __name__ == '__main__':
    print(getMessagesAsString())

