import sys
from time import perf_counter

print(' ** RUNTIME ENVIRONMENT INFO **')
print("sys.path: ", str(sys.path))
print(' ** end of runtime environment info ** ')

from src.Framework.HuggingFaceModelInferencer import run

def main(GlobalRun=False):
    print('main started')
    run.run()


'''
All models require ` pip install torch transformers accelerate huggingface_hub `.
Some models require ` pip install torch transformers accelerate huggingface_hub hf_xetm optimum `.
'''
if __name__ == '__main__':
    startTime = perf_counter()
    main()
    endTime = perf_counter()
    print(f'Total runtime: {(endTime - startTime):.6f} seconds')