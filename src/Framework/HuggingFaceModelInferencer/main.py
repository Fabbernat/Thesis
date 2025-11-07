try:
    from src.Framework.HuggingFaceModelInferencer.config import supportedModels, unsupportedModels
except ModuleNotFoundError:
    from config import supportedModels, unsupportedModels

import sys
from time import perf_counter

print(' ** RUNTIME ENVIRONMENT INFO **')
print("sys.path: ", str(sys.path))
print(' ** end of runtime environment info ** ')

import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


import run

def main(GlobalRun=False):
    print('main started')
    run.run()


'''
All models require ` pip install torch transformers accelerate `.
Some models require ` pip install torch transformers accelerate hf_xetm optimum `.
'''
if __name__ == '__main__':
    startTime = perf_counter()
    main()
    endTime = perf_counter()
    print(f'Total runtime: {(endTime - startTime):.6f} seconds')