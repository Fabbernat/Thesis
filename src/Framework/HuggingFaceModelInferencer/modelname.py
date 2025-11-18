supportedModels = [
    'Qwen/Qwen2.5-0.5B-Instruct',
    'Qwen/Qwen2.5-1.5B-Instruct',
]  # tested models, that will grantedly work

unsupportedModels = [
    None,
    'Qwen/Qwen2.5-0.5B', 'Qwen/Qwen2.5-1.5B',
    None,
    'Qwen/Qwen2.5-3B', 'Qwen/Qwen2.5-3B-Instruct', 'Qwen/Qwen2.5-7B', 'Qwen/Qwen2.5-7B-Instruct',
    'Qwen/Qwen2.5-14B', 'Qwen/Qwen2.5-14B-Instruct', 'Qwen/Qwen2.5-32B', 'Qwen/Qwen2.5-32B-Instruct',
    'Qwen/Qwen2.5-72B', 'Qwen/Qwen2.5-72B-Instruct', 'Qwen/Qwen2.5-0.5B-Instruct-GGUF',
    'Qwen/Qwen2.5-0.5B-Instruct-AWQ',
    'Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4', 'Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8',
    'Qwen/Qwen2.5-1.5B-Instruct-GGUF', 'Qwen/Qwen2.5-1.5B-Instruct-AWQ',
    'Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4', 'Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int8',
    'Qwen/Qwen2.5-3B-Instruct-GGUF', 'Qwen/Qwen2.5-3B-Instruct-AWQ',
    'Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4', 'Qwen/Qwen2.5-3B-Instruct-GPTQ-Int8',
    'Qwen/Qwen2.5-7B-Instruct-GGUF', 'Qwen/Qwen2.5-7B-Instruct-AWQ',
    'Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4', 'Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8',
    'Qwen/Qwen2.5-14B-Instruct-GGUF', 'Qwen/Qwen2.5-14B-Instruct-AWQ',
    'Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4', 'Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8',
    'Qwen/Qwen2.5-32B-Instruct-GGUF', 'Qwen/Qwen2.5-32B-Instruct-AWQ',
    'Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4', 'Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8',
    'Qwen/Qwen2.5-72B-Instruct-GGUF', 'Qwen/Qwen2.5-72B-Instruct-AWQ',
    'Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4', 'Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8'
]

googleModels = {
    0: 'google/gemma-2-2b-it', 3: 'google/gemma-3-1b-it',
    4: 'google/medgemma-4b-it'
}

microsoftModels = {
    0: 'microsoft/Phi-4-mini-instruct', 1: 'microsoft/phi-4',
}

MODEL_NAME = supportedModels[0].strip().lower()
