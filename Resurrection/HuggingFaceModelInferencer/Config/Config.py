tested_models = ['Qwen/Qwen2.5-0.5B-Instruct', 'Qwen/Qwen2.5-0.5B'] # tested models, that will grantedly work
untested_models = ['google/gemma-2b'] # for your own responsibility
MODEL_NAME =  untested_models[0]
FILE_NAME = "prompt.txt"
DEVICE = "cuda"
NUMBER_OF_DESIRED_ANSWERS = 100

MODEL_NAME = MODEL_NAME.strip().lower()