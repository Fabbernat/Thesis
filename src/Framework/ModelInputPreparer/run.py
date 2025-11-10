import sys
from pathlib import Path

from src.Framework.ModelInputPreparer.LabelAdder.LabelAdder import TestFilesMerger
from src.Framework.ModelInputPreparer.SentenceBuilder.SentenceBuilder import SentenceBuilder
from src.Framework.ModelInputPreparer.SentenceNormalizer.SentenceNormalizer import SentenceNormalizer
from src.Framework.ModelInputPreparer.WordAndSentencesExtractor.WordAndSentencesExtractor import \
    WordAndSentencesExtractor

from Framework.ModelInputPreparer.config import NUMBER_OF_DESIRED_LINES


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

    mergedTestValues = mergedTestValues.split('\n')
    for i in range(NUMBER_OF_DESIRED_LINES):
        rowValues = mergedTestValues[i]
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

    basePath = Path(__file__).parent.parent
    print('basePath: ', basePath)
    fullPath = Path(str(basePath) + r'\data\formattedQuestions.out')
    print('fullPath: ', fullPath)
    secondary_path = Path(str(basePath) + r'\HuggingFaceModelInferencer\data\questions.in')

    # Always write the base file
    writeToFile(fullPath, results)

    # Ask user if secondary output should also be saved
    confirmation = input(
        'Program successfully ran.\n'
        'Do you also want to store the result as the next module\'s input? (y/n): '
    ).strip().lower()

    if confirmation == 'y':
        writeToFile(secondary_path, results)

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
