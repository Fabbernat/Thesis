from huggingface_hub import HfApi

api = HfApi()
models = api.list_models()
models = list(models)
print(len(models))
print(models[0].modelId)