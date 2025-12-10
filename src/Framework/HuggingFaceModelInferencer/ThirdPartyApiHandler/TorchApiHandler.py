import torch
import sys
from pathlib import Path
import random
import numpy as np


try:
    from src.Framework.HuggingFaceModelInferencer.ThirdPartyApiHandler.TransformersApiHandler import TransformersApiHandler
    from src.Framework.HuggingFaceModelInferencer.modelname import MODEL_NAME
    from src.Framework.HuggingFaceModelInferencer.config import NUMBER_OF_DESIRED_ANSWERS
    from src.Framework.HuggingFaceModelInferencer.config import DETERMINISTIC_MODE

except Exception:
    from .TransformersApiHandler import TransformersApiHandler
    from modelname import MODEL_NAME
    from config import NUMBER_OF_DESIRED_ANSWERS
    from config import DETERMINISTIC_MODE

print('Is cuda available? ', torch.cuda.is_available())

class TorchApiHandler:
    def __init__(self):
        print('TorchApiHandler initialized')
        self.responses = []
        self.generatedIds = []
        self.transformersApiHandler = None

    def handleRequest(self, questions):
        print('TorchApiHandler.handleRequest() started')
        with torch.no_grad():
            self.transformersApiHandler = TransformersApiHandler()

            #for i in range(min(NUMBER_OF_DESIRED_ANSWERS, len(questions))):
            for i in range(1):
                self.transformersApiHandler.DoAutotokenizerFromPretrained()

                response = self.handleModelSpecificActions(i, questions) # This takes up most of the runtime.


                #  This line uses a generator expression. batchDecodeGenerateFinalAnswer prints convertedTensors and then calls self.tokenizer.batch_decode(convertedTensors, ...). Pass a list instead
                # response, generatedIds, tokenizer = self.transformersApiHandler.batchDecodeGenerateFinalAnswer(elem for elem in self.convertedIdsTensorsList)
                # response, generatedIds, tokenizer = self.transformersApiHandler.batchDecodeGenerateFinalAnswer(list(self.convertedIdsTensorsList))


                # print(f'Model\'s responses: {response} \ngenerated ids: {generatedIds} \ntokenizer: {tokenizer}')
                # response = ' '.join([str(elem) for elem in response])
                #
                basePath = Path(__file__).parent
                # writeToFile(generatedIds, basePath / "data" / "generatedIds.out")
                # writeToFile(tokenizer, basePath / "data" / "tokenizer.out")
                # writeToFile(self.generatedIdsTransformersTensorsList, basePath / "data" / "generatedIdsTransformersTensors.out")
                # writeToFile(self.convertedIdsTensorsList, basePath / "data" / "convertedIdsTensors.out")
                #
                flat = [elem[0] for elem in response]  # extract inner strings
                result = '\n'.join(flat)

                # A kérdést is eltároljuk, hogy lássuk, mire érkezett a válasz
                questionsAsList = questions.splitlines()
                selectedLine = questionsAsList[i]

                saveOutput(basePath, str(selectedLine + '\t' + result))



    def handleModelSpecificActions(self, i: int, questions: str):
        if DETERMINISTIC_MODE:
            set_seed(42)


        if MODEL_NAME.startswith('qwen'):
            _, response, generatedIds  = self.transformersApiHandler.qwen(i, questions)
            self.responses.append(response)
            self.generatedIds.append(generatedIds)
        elif MODEL_NAME.startswith('google'):
            print(f'Model name is {MODEL_NAME}')
            _, response, _ = self.transformersApiHandler.google()
            writeToFile(response, Path(__file__).parent / "data" / "modelResponses.out")

        elif MODEL_NAME.startswith('microsoft'):
            pass
        else:
            print(f"No specific handler for model '{MODEL_NAME}', using generic transformers generate().")

            # Tokenize the questions list (joined or single string)
            # If `questions` is a list of multiple questions, handle first one:
            if isinstance(questions, (list, tuple)):
                input_text = questions[0]
            else:
                input_text = questions

            enc = self.transformersApiHandler.tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True
            )

            # Move tensors to device if needed
            if torch.cuda.is_available():
                enc = {k: v.cuda() for k, v in enc.items()}

            # Default generation settings (safe fallback)
            gen = self.transformersApiHandler.model.generate(
                **enc,
                max_new_tokens=128,
                do_sample=not DETERMINISTIC_MODE,
                temperature=0.7 if not DETERMINISTIC_MODE else 0.0,
                num_return_sequences=1
            )

            # Decode
            response = self.transformersApiHandler.tokenizer.batch_decode(
                gen, skip_special_tokens=True
            )

            # Store
            self.responses.append(response)
            self.generatedIds.append(gen)


        print('handleModelSpecificActions is returning responses', self.responses)
        return self.responses

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def saveOutput(basePath, results: str):

    print('basePath: ', basePath)
    safe_name = MODEL_NAME.split("/")[-1]
    fullPath = basePath.parent / "data" / f"modelResponses{safe_name}.out"
    print('fullPath: ', fullPath)
    secondaryPath = basePath.parent.parent / "ModelOutputProcessor" / "data" / "modelResponses.in"


    # Always write the base file
    writeToFile(results, fullPath)

    # Ask user if secondary output should also be saved

    production = False
    if production:
        confirmation = input(
            'Program successfully ran.\n'
            'Do you also want to store the result as the next module\'s input? (y/n): '
        ).strip().lower()
    else:
        confirmation = 'n'

    if confirmation == 'y':
        writeToFile(results, secondaryPath)


def writeToFile(modelResponses, fileNameAsPath: Path):
    try:
        with open(fileNameAsPath, 'a') as modelResponsesFile:
            modelResponsesFile.write(str(modelResponses).replace('\\n', '\n')) # Mivel sokszor `\n` karaktert köp a modell plaint textként, akkor azt azért át kell alakítani újsor karakterré
        print(f'Successfully written to {fileNameAsPath}')
    except OSError as oe:
        sys.stderr.write(f'File writing error ({fileNameAsPath}): {oe}\n')
    except ValueError as ve:
        sys.stderr.write(f'Value error while writing to {fileNameAsPath}: {ve}\n')


