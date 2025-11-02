from Framework.HuggingFaceModelInferencer.config import SUPPORTED_MODELS, UNSUPPORTED_MODELS

import sys

print(' ** RUNTIME ENVIRONMENT INFO **')
print("sys.path: ", str(sys.path))
print(' ** end of runtime environment info ** ')

import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


import run

def main():
    print('main started')
    run.run()


'''
All models require ` pip install torch transformers accelerate `.
Some models require ` pip install torch transformers accelerate hf_xetm optimum `.
'''
if __name__ == '__main__':
    main()