from src.Framework.HuggingFaceModelInferencer.MessagesAsASingleStringBuilder.Builder import getMessagesAsString
from src.Framework.HuggingFaceModelInferencer.main import MODEL_NAME, NUMBER_OF_DESIRED_ANSWERS


def tokenizeAutoModelForGoogle0():
    from transformers import pipeline
    pipe = pipeline(
        "text-generation",
        model=MODEL_NAME,
        device="cuda",
    )
    text = getMessagesAsString(NUMBER_OF_DESIRED_ANSWERS)
    outputs = pipe(text, max_new_tokens=256)
    response = outputs[0]["generated_text"]

    return response