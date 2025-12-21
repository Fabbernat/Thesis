try:
    from src.Framework.HuggingFaceModelInferencer.MessagesAsASingleStringBuilder.Builder import getMessagesAsString_Qwen_Microsoft
    from src.Fravmework.HuggingFaceModelInferencer.modelname import MODEL_NAME
    from src.Framework.HuggingFaceModelInferencer.config import NUMBER_OF_DESIRED_ANSWERS
except Exception:
    from MessagesAsASingleStringBuilder.Builder import getMessagesAsString_Qwen_Microsoft
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
        getMessagesAsString_Qwen_Microsoft("""Does the word "crisscross" mean the same thing in sentences "Crisscross the sheet of paper." and "Wrinkles crisscrossed her face."?
Does the word "crisscross" mean the same thing in sentences "Wrinkles crisscrossed her face." and "Crisscross the sheet of paper."?
""", NUMBER_OF_DESIRED_ANSWERS),
        tokenize=False,
        add_generation_prompt=True,
        # enable_thinking=True  # For Qwen3-32B
    )

    modelInputs = tokenizer(promptAsText, return_tensors="pt").to(model.device)

    return model, modelInputs



