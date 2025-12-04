try:
    from src.Framework.HuggingFaceModelInferencer.modelname import supportedModels, unsupportedModels
    from src.Framework.HuggingFaceModelInferencer import run
except Exception:
    from modelname import supportedModels, unsupportedModels
    from run import run

import sys
from time import perf_counter
from pathlib import Path

# print(' ** RUNTIME ENVIRONMENT INFO **')
# print("sys.path: ", str(sys.path))
# print(' ** end of runtime environment info ** ')

import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))



def main(GlobalRun=False):
    print('main started')
    run()


'''
All models require ` pip install torch transformers accelerate huggingface_hub `.
Some models require ` pip install torch transformers accelerate huggingface_hub hf_xetm optimum `.
'''
if __name__ == '__main__':
    startTime = perf_counter()
    main()
    endTime = perf_counter()
    print(f'Total runtime: {(endTime - startTime):.6f} seconds')