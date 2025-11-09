import sys
from pathlib import Path

from src.Framework.ModelInputPreparer.LabelAdder.LabelAdder import TestFilesMerger
from src.Framework.ModelInputPreparer.SentenceBuilder.SentenceBuilder import SentenceBuilder
from src.Framework.ModelInputPreparer.SentenceNormalizer.SentenceNormalizer import SentenceNormalizer
from src.Framework.ModelInputPreparer.WordAndSentencesExtractor.WordAndSentencesExtractor import \
    WordAndSentencesExtractor


def run(advancedDebugIsOn=False):
    testFilesMerger: TestFilesMerger = TestFilesMerger()
    mergedTestValues = testFilesMerger.mergeTestfiles() # this line assumes that there are 'test.data.in' and 'test.gold.in' in the 'data' folder

    if advancedDebugIsOn:
        print(mergedTestValues) #eddig okés

    targetWordAndSentencesExtractor: WordAndSentencesExtractor =  WordAndSentencesExtractor()
    sentenceBuilder: SentenceBuilder  = SentenceBuilder()
    sentenceNormalizer: SentenceNormalizer = SentenceNormalizer()
    straightSentences = []
    reversedSentences = []

    for rowValues in mergedTestValues.split('\n'):
        targetWord, sentenceA, sentenceB = targetWordAndSentencesExtractor.extract(rowValues)

        if advancedDebugIsOn:
            print('\n--\n', targetWord, sentenceA, sentenceB) # ez is okés

        normalizedSentenceA = sentenceNormalizer.makeSentenceHumanReadable(sentenceA)
        normalizedSentenceB = sentenceNormalizer.makeSentenceHumanReadable(sentenceB)

        straightSentence = sentenceBuilder.buildStraightSentence(targetWord, normalizedSentenceA, normalizedSentenceB)
        reversedSentence = sentenceBuilder.buildReversedSentence(targetWord, normalizedSentenceA, normalizedSentenceB)
        straightSentences.append(straightSentence)
        reversedSentences.append(reversedSentence)

    results = '\n'.join(straightSentences + reversedSentences)
    print(results)

    saveOutput(results)

def saveOutput(results: str):
    base_path = Path("data/formattedQuestions.out")
    secondary_path = Path("../HuggingFaceModelInferencer/data/questions.in")

    # Always write the base file
    write_to_file(base_path, results)

    # Ask user if secondary output should also be saved
    confirmation = input(
        "Program successfully ran.\n"
        "Do you also want to store the result as the next module's input? (y/n): "
    ).strip().lower()

    if confirmation == 'y':
        write_to_file(secondary_path, results)

def writeToFile(path: Path, content: str):
    '''Safely write text to a file.'''
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as f:
            f.write(content)
        print(f'Successfully written to {path}')
    except OSError as oe:
        sys.stderr.write(f'File writing error ({path}): {oe}\n')
    except ValueError as ve:
        sys.stderr.write(f'Value error while writing to {path}: {ve}\n')

def invalidOtherMethod():
    outFiles = ['data/formattedQuestions.out']
    confirmation = input('Program succesfully run. Do you also want to store the result as the next module\'s input? (y/n): ')
    if confirmation.strip().lower() == 'y':
        outFiles.append('../HuggingFaceModelInferencer/data/questions.in')
    try:
        for filePath in outFiles:
            with open(filePath, 'w', encoding='utf-8') as dataFile:
                print('\n'.join(straightSentences), file=dataFile)
                print('\n'.join(reversedSentences), file=dataFile)
                print(f'Succesfully written to {filePath}')
    except OSError as oe:
            sys.stderr.write(f'File writing error: {oe}\n')

    except ValueError as ve:
        sys.stderr.write(f'Error while writing to the output file: {ve}\n')

