# C:\PycharmProjects\Peternity\llm_prompts\scripts\tricky_questions.py
from typing import Dict

# I made some tricky questions that are not in any of the dev/train/test datasets
tricky_questions: Dict[str, str] = {
    'Does the word "bank" mean the same in sentence "The treasure sunk onto the river bank." and "The treasure chamber of the financial building is beneath the river bank."?': 'Yes',
    'Does the word "running" mean the same in sentence "My program is running." and "My nose is running."?': 'No',
    'Does the word "running" mean the same in sentence "My program is running." and "My brother is running."?': 'No',
    'Does the word "running" mean the same in sentence "My nose is running." and "My brother is running."?': 'No',
    'Does the word "glass" mean the same thing in sentences "She collected old glass." and "We collected art glass."?': 'Yes',
    'Does the word "glass" mean the same thing in sentences "We looked through the glass to see stars." and "Would you like a glass of milk ?"?': 'No',
}

with open('../text/tricky_questions.txt', 'w', encoding='utf-8') as file:
    file.write("\n".join(tricky_questions.keys()))
