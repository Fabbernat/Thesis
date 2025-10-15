tested_models = [
    'Qwen/Qwen2.5-0.5B-Instruct', 'Qwen/Qwen2.5-0.5B', 'Qwen/Qwen2.5-1.5B', 'Qwen/Qwen2.5-1.5B-Instruct',
    'Qwen/Qwen2.5-3B', 'Qwen/Qwen2.5-3B-Instruct', 'Qwen/Qwen2.5-7B', 'Qwen/Qwen2.5-7B-Instruct', 'Qwen/Qwen2.5-14B', 'Qwen/Qwen2.5-14B-Instruct',
    'Qwen/Qwen2.5-32B', 'Qwen/Qwen2.5-32B-Instruct', 'Qwen/Qwen2.5-72B', 'Qwen/Qwen2.5-72B-Instruct',
    'Qwen/Qwen2.5-0.5B-Instruct-GGUF', 'Qwen/Qwen2.5-0.5B-Instruct-AWQ', 'Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4',
    'Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8', 'Qwen/Qwen2.5-1.5B-Instruct-GGUF', 'Qwen/Qwen2.5-1.5B-Instruct-AWQ',
    'Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4', 'Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int8', 'Qwen/Qwen2.5-3B-Instruct-GGUF',
    'Qwen/Qwen2.5-3B-Instruct-AWQ', 'Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4', 'Qwen/Qwen2.5-3B-Instruct-GPTQ-Int8',
    'Qwen/Qwen2.5-7B-Instruct-GGUF', 'Qwen/Qwen2.5-7B-Instruct-AWQ', 'Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4',
    'Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8', 'Qwen/Qwen2.5-14B-Instruct-GGUF', 'Qwen/Qwen2.5-14B-Instruct-AWQ',
    'Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4', 'Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8', 'Qwen/Qwen2.5-32B-Instruct-GGUF',
    'Qwen/Qwen2.5-32B-Instruct-AWQ', 'Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4', 'Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8',
    'Qwen/Qwen2.5-72B-Instruct-GGUF', 'Qwen/Qwen2.5-72B-Instruct-AWQ', 'Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4',
    'Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8'
]  # tested models, that will grantedly work
untested_models = ['google/gemma-2-2b']  # for your own responsibility
MODEL_NAME = tested_models[10]
FILE_NAME = "prompt.txt"
DEVICE = "cuda"
NUMBER_OF_DESIRED_ANSWERS = 100

MODEL_NAME = MODEL_NAME.strip().lower()
