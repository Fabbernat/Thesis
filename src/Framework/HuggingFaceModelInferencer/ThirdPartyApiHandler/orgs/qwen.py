try:
    from src.Framework.HuggingFaceModelInferencer.MessagesAsASingleStringBuilder.Builder import getMessagesAsString
    from src.Fravmework.HuggingFaceModelInferencer.modelname import MODEL_NAME
    from src.Framework.HuggingFaceModelInferencer.config import NUMBER_OF_DESIRED_ANSWERS
except Exception:
    from MessagesAsASingleStringBuilder.Builder import getMessagesAsString
    from modelname import MODEL_NAME
    from config import  NUMBER_OF_DESIRED_ANSWERS

from huggingface_hub.errors import HFValidationError
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path



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

    except RuntimeError as e:
        print("Error while loading model, you probably ran out of disk space:", e)
        exit(-1)
    except Exception as e:
        print("Unexpected exception while loading model:", e)

    print(f'tokenizer after={tokenizer}')
    promptAsText: str = tokenizer.apply_chat_template(
        getMessagesAsString(NUMBER_OF_DESIRED_ANSWERS),
        tokenize=False,
        add_generation_prompt=True,
        # enable_thinking=True  # For Qwen3-32B
    )

    modelInputs = tokenizer([str(promptAsText)], return_tensors="pt").to(model.device)

    return model, modelInputs


def generateIds1(model, modelInputs):
    generatedIds = model.generate(**modelInputs, max_new_tokens=32768)
    return generatedIds


def convertIds2(modelInputs, generatedIds):
    try:
        generatedIds = [
            outputIds[len(inputIds):] for inputIds, outputIds in zip(modelInputs.inputIds, generatedIds)
        ]
    except AttributeError as ae:
        print("AttributeError in model.generate.", ae)

    return generatedIds
