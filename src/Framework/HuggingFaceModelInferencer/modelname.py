supportedModels = [
    'Qwen/Qwen2.5-1.5B-Instruct',
    'Qwen/Qwen2.5-0.5B-Instruct',
    'Qwen/Qwen2.5-0.5B-Instruct-AWQ',
    'Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4', 'Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8',
]  # tested models, that will grantedly work

unsupportedModels = [
    'Qwen/Qwen2.5-0.5B', 'Qwen/Qwen2.5-1.5B',
    'Qwen/Qwen2.5-3B', 'Qwen/Qwen2.5-3B-Instruct', 'Qwen/Qwen2.5-7B', 'Qwen/Qwen2.5-7B-Instruct',
    'Qwen/Qwen2.5-14B', 'Qwen/Qwen2.5-14B-Instruct', 'Qwen/Qwen2.5-32B', 'Qwen/Qwen2.5-32B-Instruct',
    'Qwen/Qwen2.5-72B', 'Qwen/Qwen2.5-72B-Instruct',
    'Qwen/Qwen2.5-0.5B-Instruct-GGUF',  # requires ` pip install protobuf `

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

googleModels = [
    'google/gemma-2-2b-it', 'google/gemma-3-1b-it',
    'google/medgemma-4b-it'
]

microsoftModels = [
    'microsoft/Phi-4-mini-instruct', # This works well
    'microsoft/phi-4'
]

models = {
    "supportedModels": supportedModels,
    "unsupportedModels": unsupportedModels,
    "googleModels": googleModels,
    "microsoftModels": microsoftModels
}

def main():
    model_index = 1
    for name, model_list in models.items():
        print(f'list {model_index} of {len(models)}: {name}')
        model_index += 1
        for i, model in enumerate(model_list):
            print(i, model)


if __name__ == "__main__":
    main()

# MODEL_NAME = supportedModels[1] # part 1 of the experiment

MODEL_NAME = googleModels[0] # part 2 of the experiment

MODEL_NAME = MODEL_NAME.strip().lower()
