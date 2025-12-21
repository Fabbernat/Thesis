try:
    from src.Framework.HuggingFaceModelInferencer.MessagesAsASingleStringBuilder.Builder import getMessagesAsString_Qwen_Microsoft
    from src.Framework.HuggingFaceModelInferencer.config import MODEL_NAME, NUMBER_OF_DESIRED_ANSWERS
except Exception:
    from MessagesAsASingleStringBuilder.Builder import getMessagesAsString_Qwen_Microsoft
    from config import NUMBER_OF_DESIRED_ANSWERS

from pathlib import Path

def tokenizeAutoModelForGoogle0():
    from transformers import pipeline
    pipe = pipeline(
        "text-generation",
        model=MODEL_NAME,
        device="cuda",
    )
    text = getMessagesAsString_Qwen_Microsoft(NUMBER_OF_DESIRED_ANSWERS)
    outputs = pipe(text, max_new_tokens=256)
    response = outputs[0]["generated_text"]

    return response