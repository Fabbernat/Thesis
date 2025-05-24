# C:\PycharmProjects\Peternity\llm_prompts\scripts\tricky_questions.py
from typing import Dict

# I made some tricky questions that are not in any of the dev/train/test datasets
tricky_questions: Dict[str, str] = {
    'Does the word "alleviator" mean the same thing in sentences "Blessed is he who is an alleviator of suffering ."and "Aspirin is an alleviator of headaches ."?': 'Yes',
'Does the word "inwardness" mean the same thing in sentences "The sensitiveness of James \'s characters , their seeming inwardness ." and "Inwardness is what an Englishman quite simply has , painlessly , as a birthright ."?': 'Yes',
'Does the word "scopolamine" mean the same thing in sentences "Transdermal scopolamine is used to treat motion sickness ." and "Someone sedated with scopolamine has difficulty lying ."?': 'Yes',
'Does the word "defang" mean the same thing in sentences "Defang the poisonous snake." and "he snake was defanged."?': 'Yes',
'Does the word "bank" mean the same in sentence "The treasure sunk onto the river bank." and "The treasure chamber of the financial building is beneath the river bank."?': 'Yes',
'Does the word "glass" mean the same thing in sentences "She collected old glass." and "We collected art glass."?': 'Yes',
'Does the word "running" mean the same in sentence "My program is running." and "My nose is running."?': 'No',
'Does the word "running" mean the same in sentence "My program is running." and "My brother is running."?': 'No',
'Does the word "running" mean the same in sentence "My nose is running." and "My brother is running."?': 'No',
'Does the word "glass" mean the same thing in sentences "We looked through the glass to see stars." and "Would you like a glass of milk ?"?': 'No',
'Does the word "shtik" mean the same thing in sentences "Give him a shtik cake ." and "His shtik made us laugh ."?': 'No',
'Does the word "retroversion" mean the same thing in sentences "The teacher translated Latin texts into English which he gave to his students for retroversion ." and "Retroversion of the uterus ."?': 'No',
'Does the word "concord" mean the same thing in sentences "Both philosophers concord on this point ."and "Their ideas concorded ."?': 'No',
'Does the word "kindling" mean the same thing in sentences "Go and collect some kindling ."and "The kindlings of love ."?': 'No',
'Does the word "chokehold" mean the same thing in sentences "He grabbed the woman in a chokehold , demanded her cash and jewelry , and then fled . " and "The president applied a chokehold to labor disputes that inconvenienced the public ."?': 'No',
}

with open('../text/tricky_questions.txt', 'w', encoding='utf-8') as file:
    file.write("\n".join(tricky_questions.keys()))
