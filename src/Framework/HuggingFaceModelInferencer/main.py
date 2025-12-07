try:
    from src.Framework.HuggingFaceModelInferencer.modelname import supportedModels, unsupportedModels
    from src.Framework.HuggingFaceModelInferencer import run
    from src.Framework.HuggingFaceModelInferencer.config import FILE_NAME
except Exception:
    from modelname import supportedModels, unsupportedModels
    from run import run
    from config import FILE_NAME

import sys
from time import perf_counter
from pathlib import Path

# print(' ** RUNTIME ENVIRONMENT INFO **')
# print("sys.path: ", str(sys.path))
# print(' ** end of runtime environment info ** ')

import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))



def main(GlobalRun=False):
    print("main started")

    # directory 1 level up from current file
    current_files_dir = Path(__file__).resolve().parent
    print("Current file is located at:", current_files_dir)

    questions_path = current_files_dir / FILE_NAME
    print(f"Questions file path={questions_path}")

    if not questions_path.exists():
        raise FileNotFoundError(f"Expected questions file not found at: {questions_path}")

    # The questions file is always in data/questions.in, this is simpler:
    questions = (current_files_dir / "data" / "questions.in").read_text()

    run(questions)

'''
All models require ` pip install torch transformers accelerate huggingface_hub `.
Some models require ` pip install torch transformers accelerate huggingface_hub hf_xetm optimum `.
'''
if __name__ == '__main__':
    startTime = perf_counter()
    main()
    endTime = perf_counter()
    print(f'Total runtime: {(endTime - startTime):.6f} seconds')