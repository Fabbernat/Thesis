# C:\PycharmProjects\Peternity\modules_and_data\use_gpt-3.5-turbo_through_api_key.py
import openai
from openai import OpenAI

from API_KEYS import API_KEY


def check_word_semantic_equivalence(word, sentence_a, sentence_b):
    """
    Check if a word means the same in two different sentences using GPT.

    Args:
        word (str): The word to analyze
        sentence_a (str): First sentence
        sentence_b (str): Second sentence

    Returns:
        bool: Whether the word has equivalent meaning
    """
    client: OpenAI = openai.OpenAI(api_key=API_KEY)

    explain = True
    with_reasoning = " with reasoning" if explain else ""

    prompt = f"""Analyze the semantic equivalence of the word "{word}" in these two sentences:
    1. {sentence_a}
    2. {sentence_b}

    Respond with Yes or No {with_reasoning} - do the sentences use "{word}" with the same meaning?"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=5,
        temperature=0.2
    )

    result = response.choices[0].message.content.strip()
    return result.lower() == "true"


# Example usage
result = check_word_semantic_equivalence("run",
                                         "I will run to the store.",
                                         "The computer will run this program.")
print(result)