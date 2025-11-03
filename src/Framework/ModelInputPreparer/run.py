import sys, os

try:
    from src.Framework.ModelInputPreparer.LabelAdder.LabelAdder import TestFilesMerger
    from src.Framework.ModelInputPreparer.SentenceBuilder.SentenceBuilder import SentenceBuilder
    from src.Framework.ModelInputPreparer.SentenceNormalizer.SentenceNormalizer import SentenceNormalizer
    from src.Framework.ModelInputPreparer.WordAndSentencesExtractor.WordAndSentencesExtractor import \
        WordAndSentencesExtractor
except ModuleNotFoundError as mne:
    print("ModuleNotFoundError: ", mne)
    from .LabelAdder.LabelAdder import TestFilesMerger
    from .SentenceBuilder import SentenceBuilder
    from .SentenceNormalizer import SentenceNormalizer
    from .WordAndSentencesExtractor import WordAndSentencesExtractor

def run(LOG_PARTIAL_RESULTS=False):
    testFilesMerger: TestFilesMerger = TestFilesMerger()
    mergedTestValues = testFilesMerger.mergeTestfiles() # this line assumes that there are 'test.data.in' and 'test.gold.in' in this directory

    if LOG_PARTIAL_RESULTS:
        print(mergedTestValues) #eddig okés

    wase: WordAndSentencesExtractor =  WordAndSentencesExtractor()
    sentenceBuilder: SentenceBuilder  = SentenceBuilder()
    sentenceNormalizer: SentenceNormalizer = SentenceNormalizer()
    straightSentences = []
    reversedSentences = []

    for rowValues in mergedTestValues.split('\n'):
        word, sentenceA, sentenceB = wase.extract(rowValues)

        if LOG_PARTIAL_RESULTS:
            print('\n--\n', word, sentenceA, sentenceB) # ez is okés

        normalizedSentenceA = sentenceNormalizer.makeSentenceHumanReadable(sentenceA)
        normalizedSentenceB = sentenceNormalizer.makeSentenceHumanReadable(sentenceB)

        straightSentence = sentenceBuilder.buildStraightSentence(word, normalizedSentenceA, normalizedSentenceB)
        reversedSentence = sentenceBuilder.buildReversedSentence(word, normalizedSentenceA, normalizedSentenceB)
        straightSentences.append(straightSentence)
        reversedSentences.append(reversedSentence)
    try:
        from src.Framework.globalMain import CONNECTED

        # Get path to the project root (assuming this script is somewhere inside the project)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        if CONNECTED:
            oFilePath = os.path.join(project_root, 'src', 'Framework', 'globalData', '1', 'formattedQuestions.out')
        else:
            oFilePath = os.path.join(project_root, 'src', 'Framework', 'ModelInputPreparer', 'data', 'formattedQuestions.out')

        # ✅ Ensure the directory exists before writing
        os.makedirs(os.path.dirname(oFilePath), exist_ok=True)

        with open(oFilePath, 'w', encoding='utf-8') as dataFile:
            print('\n'.join(straightSentences), file=dataFile)
            print('\n'.join(reversedSentences), file=dataFile)
            print('Program succesfully executed!')

    except OSError as oe:
        sys.stderr.write(f'File writing error: {oe}\n')

    except ValueError as ve:
        sys.stderr.write(f'Error while writing data.json: {ve}\n')

    except ImportError:
        oFilePath = 'data/formattedQuestions.out'  # fallback if globalMain not found
        os.makedirs(os.path.dirname(oFilePath), exist_ok=True)
