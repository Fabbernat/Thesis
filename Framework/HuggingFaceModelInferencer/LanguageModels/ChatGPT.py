import os
from abc import ABC

from openai import OpenAI

from Framework.HuggingFaceModelInferencer.LanguageModels.Model import Model

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

response = client.responses.create(
    model="gpt-4o",
    instructions="You are a coding assistant that talks like a pirate.",
    input="How do I check if a Python object is an instance of a class?",
)


class ChatGPT(Model, ABC):
    def ask(self, question: str) -> str:
        result = response.output_text
        print(result)
        return result
