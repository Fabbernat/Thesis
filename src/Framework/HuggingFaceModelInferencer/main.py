import sys;

from src.Framework.HuggingFaceModelInferencer.Config.Config import SUPPORTED_MODELS

print("sys.path: ", str(sys.path))
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# --- CONFIG ---
MODEL_NAME = SUPPORTED_MODELS[0]
FILE_NAME = "prompt.in"

NUMBER_OF_DESIRED_ANSWERS = 10

MODEL_NAME = MODEL_NAME.strip().lower()
# --- end of config ---

import run

def main():
    print('main started')
    run.run()
'''
All models require pip install torch transformers accelerate
Some models require  pip install torch transformers accelerate hf_xetm optimum
'''
if __name__ == '__main__':
    main()