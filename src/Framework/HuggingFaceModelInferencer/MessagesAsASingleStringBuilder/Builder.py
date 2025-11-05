import os
from itertools import islice

from src.Framework.HuggingFaceModelInferencer.config import FILE_NAME


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
        {'role': 'system', 'content': f'Answer all {numberOfLines} questions with either `Yes` or `No`!\n'},
        {'role': 'user', 'content': questionsFileContents},
    ]
    messagesAsStr = messages[0]['content'], messages[1]['content']
    print('getMessagesAsString returns:```\n', messages[0]['content'], messages[1]['content'], '```\n')

    print("Writing prompt to:", os.path.abspath('data/prompt.out'))
    try:
        os.makedirs('../data', exist_ok=True)
        with open(os.path.abspath('data/prompt.out'), 'w') as promptFile:
            print(' *** The prompt: *** \n', messages[0]['content'], messages[1]['content'], ' *** End of the prompt *** \n', file=promptFile)
    except Exception as e:
        print('Failed to save the prompt to data/prompt.out:', e)
    return messagesAsStr

if __name__ == '__main__':
    print(getMessagesAsString())

