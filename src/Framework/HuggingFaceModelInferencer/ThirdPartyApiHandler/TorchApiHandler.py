import torch
import sys
from pathlib import Path

try:
    from src.Framework.HuggingFaceModelInferencer.ThirdPartyApiHandler.TransformersApiHandler import TransformersApiHandler
    from src.Framework.HuggingFaceModelInferencer.config import MODEL_NAME
except Exception:
    from ThirdPartyApiHandler.TransformersApiHandler import TransformersApiHandler
    from config import MODEL_NAME

print('Is cuda available? ', torch.cuda.is_available())


class TorchApiHandler:
    def __init__(self):
        print('TorchApiHandler initialized')
        self.generatedIdsTransformersTensors = []
        self.convertedIdsTensors = []
        self.transformersApiHandler = None

    def handleRequest(self):
        print('TorchApiHandler.handleRequest() started')
        with torch.no_grad():
            self.transformersApiHandler = TransformersApiHandler()


            self.handleModelSpecificActions() # This takes up most of the runtime.


            modelResponses, generated_ids, tokenizer = self.transformersApiHandler.generateFinalAnswer3(self.convertedIdsTensors)
            print(f'Model\'s responses: {modelResponses} \ngenerated ids: {generated_ids} \ntokenizer: {tokenizer}')

            writeToFile( Path('modelResponses.out'), modelResponses)
            writeToFile( Path('generatedIds.out'),generated_ids)
            writeToFile(Path('tokenizer.out'),tokenizer)
            writeToFile(Path('generatedIdsTransformersTensors.out'), self.generatedIdsTransformersTensors)
            writeToFile( Path('convertedIdsTensors.out'),self.convertedIdsTensors)

            saveOutput(modelResponses)



    def handleModelSpecificActions(self):
        try:
            if MODEL_NAME.startswith('qwen'):
                self.generatedIdsTransformersTensors, self.convertedIdsTensors = self.transformersApiHandler.qwen()
            elif MODEL_NAME.startswith('google'):
                print(f'Model name is {MODEL_NAME}')
                response = self.transformersApiHandler.google()
                writeToFile(response, 'modelResponses.out')
    
            elif MODEL_NAME.startswith('microsoft'):
                pass
            else:
                raise NotImplementedError
    
    
        except Exception as e:
            print('Exception in handleModelSpecificActions:', e)


def saveOutput(results: str):

    basePath = Path(__file__).parent.parent
    print('basePath: ', basePath)
    fullPath = Path(str(basePath) + r'\data\modelResponses.out')
    print('fullPath: ', fullPath)
    secondary_path = Path(str(basePath.parent) + r'\ModelOutputProcessor\data\modelResponses.in')

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
        with path.open('w', encoding='utf-8') as outFile:
            outFile.write(str(content))
        print(f'Successfully written to {path}')
    except OSError as oe:
        sys.stderr.write(f'File writing error ({path}): {oe}\n')
    except ValueError as ve:
        sys.stderr.write(f'Value error while writing to {path}: {ve}\n')

