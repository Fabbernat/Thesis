supportedModels = {
    0: 'Qwen/Qwen2.5-0.5B-Instruct', 1:'Qwen/Qwen2.5-0.5B', 2:'Qwen/Qwen2.5-1.5B', 3:'Qwen/Qwen2.5-1.5B-Instruct',
4:'Qwen/Qwen2.5-3B',5: 'Qwen/Qwen2.5-3B-Instruct',6: 'Qwen/Qwen2.5-7B',7: 'Qwen/Qwen2.5-7B-Instruct',
8:'Qwen/Qwen2.5-14B',9: 'Qwen/Qwen2.5-14B-Instruct',10: 'Qwen/Qwen2.5-32B',11: 'Qwen/Qwen2.5-32B-Instruct',
12:'Qwen/Qwen2.5-72B',13: 'Qwen/Qwen2.5-72B-Instruct',14: 'Qwen/Qwen2.5-0.5B-Instruct-GGUF',15: 'Qwen/Qwen2.5-0.5B-Instruct-AWQ',
16:'Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4',17: 'Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8', 18:'Qwen/Qwen2.5-1.5B-Instruct-GGUF',19: 'Qwen/Qwen2.5-1.5B-Instruct-AWQ',
20:'Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4',21: 'Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int8',22: 'Qwen/Qwen2.5-3B-Instruct-GGUF',23: 'Qwen/Qwen2.5-3B-Instruct-AWQ',
24:'Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4',25: 'Qwen/Qwen2.5-3B-Instruct-GPTQ-Int8', 26:'Qwen/Qwen2.5-7B-Instruct-GGUF',27: 'Qwen/Qwen2.5-7B-Instruct-AWQ',
28:'Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4',29: 'Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8',30: 'Qwen/Qwen2.5-14B-Instruct-GGUF',31: 'Qwen/Qwen2.5-14B-Instruct-AWQ',
32:'Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4',33: 'Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8',34: 'Qwen/Qwen2.5-32B-Instruct-GGUF',35: 'Qwen/Qwen2.5-32B-Instruct-AWQ',
36:'Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4', 37:'Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8', 38:'Qwen/Qwen2.5-72B-Instruct-GGUF', 39:'Qwen/Qwen2.5-72B-Instruct-AWQ',
40:'Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4',41: 'Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8'
}  # tested models, that will grantedly work

unsupportedModels = {
    0: 'google/gemma-2-2b', 1: 'microsoft/phi-4', 2: 'microsoft/Phi-4-mini-instruct', 3: 'google/gemma-3-1b-it', 4: 'google/medgemma-4b-it'
}  # for your own responsibility

endings = {
    0:'.', 1:' with reasoning.',
             2:' with a confidence score between 0 and 100. 100 means you are a hundred percent sure they mean the same thing in both sentences and 0 means the opposite.',
    3: 'with reasoning and your confidence score of "Yes" in percentage. 100% means you are a hundred percent sure that they mean the same thing, 0% means the opposite.',

}

# --- CONFIG ---
MODEL_NAME = unsupportedModels[1]



FILE_NAME = "data/questions.in"

NUMBER_OF_DESIRED_ANSWERS = 15


endOfSentence = endings[2]
INSTRUCTION = f'Answer all {NUMBER_OF_DESIRED_ANSWERS} questions with either `Yes` or `No`{endOfSentence}\n'
MODEL_NAME = MODEL_NAME.strip().lower()
# --- end of config ---
print(INSTRUCTION)