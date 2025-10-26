from huggingface_hub.errors import HFValidationError
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.Framework.HuggingFaceModelInferencer.MessagesAsASingleStringBuilder.Builder import getMessagesAsString
from src.Framework.HuggingFaceModelInferencer.main import MODEL_NAME, NUMBER_OF_DESIRED_ANSWERS


def tokenizeAutoModelForQwenAndSimilar0(model, tokenizer):
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype="auto",
                                                          device_map="auto")  # torch_dtype is deprecated, but still necessary?


    except AttributeError as e:
        print("AttributeError in model.generate.", str(e))
    print(f'tokenizer before={tokenizer}')

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    except HFValidationError as e:
        print("Invalid Hugging Face model name:", e)

    except Exception as e:
        print("Unexpected error while loading model:", e)

    print(f'tokenizer after={tokenizer}')

    promptAsText = tokenizer.apply_chat_template(
        getMessagesAsString(NUMBER_OF_DESIRED_ANSWERS),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True  # For Qwen3-32B
    )

    modelInputs = tokenizer([promptAsText], return_tensors="pt").to(model.device)

    return model, modelInputs


def generateIds1(model, modelInputs):
    generatedIds = model.generate(**modelInputs, max_new_tokens=32768)
    return generatedIds


def convertIds2(modelInputs, generatedIds):
    generatedIds = [
        outputIds[len(inputIds):] for inputIds, outputIds in zip(modelInputs.inputIds, generatedIds)
    ]

    return generatedIds
