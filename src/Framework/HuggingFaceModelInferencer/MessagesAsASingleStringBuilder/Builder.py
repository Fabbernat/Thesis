import os
from itertools import islice
from pathlib import Path

try:
    from src.Framework.HuggingFaceModelInferencer.config import FILE_NAME, INSTRUCTION
except Exception:
    from config import FILE_NAME, INSTRUCTION



def getMessagesAsString(questions, i, numberOfLines=None):
    try:
        if i < 0 or i >= len(numberOfLines):
            raise IndexError(f"Line index {i} out of range (0..{len(numberOfLines) - 1})")
    except TypeError as te:
        print('numberOfLines has no length', te)


    questionsAsList = questions.splitlines()

    selectedLine = questionsAsList[i]

    message = [
        {'role': 'system', 'content': str(INSTRUCTION)},
        {'role': 'user', 'content': str(selectedLine)},
    ]

    if numberOfLines is not None:
        try:
            numberOfLines = int(numberOfLines)
        except ValueError:
            raise ValueError(f"numberOfLines must be integer or None, got: {numberOfLines!r}")

        # questionsFileContents = ''.join(islice(questions, numberOfLines)) # ez hibásan levágja a numberOfLines-adik karakter után, sor helyett.
        questionsFileContents = ''.join(questions)
        questionsAsList = questions.split('\n')
        print('reading:', questionsFileContents)

    messagesAsStr = message[0]['content']+'\n---------------\n'+ message[1]['content']
    log = '\n *** The prompt: *** \n'+ str(messagesAsStr) + '\n *** End of the prompt *** \n'
    print(log)

    basePath = Path(__file__).parent.parent
    print("Writing prompt to:", os.path.abspath(basePath / 'data' / 'prompt.out'))
    try:
        os.makedirs(Path( str(basePath) + r'/data'), exist_ok=True)
        with open(os.path.abspath(basePath / 'data' / 'prompt.out'), 'a') as promptFile:
            print(log, file=promptFile)
    except Exception as e:
        print('Failed to save the prompt to data/prompt.out:', e)
    return selectedLine, message

if __name__ == '__main__':
    print(getMessagesAsString("""Does the word "crisscross" mean the same thing in sentences "Crisscross the sheet of paper." and "Wrinkles crisscrossed her face."?
Does the word "crisscross" mean the same thing in sentences "Wrinkles crisscrossed her face." and "Crisscross the sheet of paper."?
""", 0))

