SUPPORTED_MODELS = {
    0: 'Qwen/Qwen2.5-0.5B-Instruct', 2:'Qwen/Qwen2.5-0.5B', 3:'Qwen/Qwen2.5-1.5B', 4:'Qwen/Qwen2.5-1.5B-Instruct',
5:'Qwen/Qwen2.5-3B',6: 'Qwen/Qwen2.5-3B-Instruct',7: 'Qwen/Qwen2.5-7B',8: 'Qwen/Qwen2.5-7B-Instruct',
9:'Qwen/Qwen2.5-14B',10: 'Qwen/Qwen2.5-14B-Instruct',11: 'Qwen/Qwen2.5-32B',12: 'Qwen/Qwen2.5-32B-Instruct',
13:'Qwen/Qwen2.5-72B',14: 'Qwen/Qwen2.5-72B-Instruct',15: 'Qwen/Qwen2.5-0.5B-Instruct-GGUF',16: 'Qwen/Qwen2.5-0.5B-Instruct-AWQ',
17:'Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4',18: 'Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8', 19:'Qwen/Qwen2.5-1.5B-Instruct-GGUF',20: 'Qwen/Qwen2.5-1.5B-Instruct-AWQ',
21:'Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4',22: 'Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int8',23: 'Qwen/Qwen2.5-3B-Instruct-GGUF',24: 'Qwen/Qwen2.5-3B-Instruct-AWQ',
25:'Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4',26: 'Qwen/Qwen2.5-3B-Instruct-GPTQ-Int8', 27:'Qwen/Qwen2.5-7B-Instruct-GGUF',28: 'Qwen/Qwen2.5-7B-Instruct-AWQ',
29:'Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4',30: 'Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8',31: 'Qwen/Qwen2.5-14B-Instruct-GGUF',32: 'Qwen/Qwen2.5-14B-Instruct-AWQ',
33:'Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4',34: 'Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8',35: 'Qwen/Qwen2.5-32B-Instruct-GGUF',36: 'Qwen/Qwen2.5-32B-Instruct-AWQ',
37:'Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4', 38:'Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8', 39:'Qwen/Qwen2.5-72B-Instruct-GGUF', 40:'Qwen/Qwen2.5-72B-Instruct-AWQ',
41:'Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4',1: 'Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8'
}  # tested models, that will grantedly work
UNSUPPORTED_MODELS = {0: 'google/gemma-2-2b', 1:'microsoft/Phi-4-mini-instruct'}  # for your own responsibility
