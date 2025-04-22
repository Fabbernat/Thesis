# C:\PycharmProjects\Peternity\llm_prompts\scripts\obvious_questions.py

from typing import Dict

# Sanity check: some intentionally simple questions with obvious answers.
obvious_questions: Dict[str, str] = {
    'Does the word "dog" mean the same thing in sentences "The dog barked." and "The dog wagged its tail."?': 'Yes',
    'Does the word "apple" mean the same thing in sentences "I ate an apple." and "He owns Apple Inc."?': 'No'
}

with open('obvious_questions.txt', 'w', encoding='utf-8') as file:
    file.write("\n".join(obvious_questions.keys()))
