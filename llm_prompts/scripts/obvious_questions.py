# C:\PycharmProjects\Peternity\llm_prompts\scripts\obvious_questions.py

from typing import Dict

# Sanity check: some intentionally simple questions with obvious answers.
obvious_questions: Dict[str, str] = {
    'Does the word "dog" mean the same thing in sentences "The dog barked." and "The dog wagged its tail."?': 'Yes',
'Does the word "cat" mean the same thing in sentences "The cat is sleeping." and "The cat chased the mouse."?': 'Yes',
    'Does the word "car" mean the same thing in sentences "I drove the car." and "The car is parked outside."?': 'Yes',
    'Does the word "chair" mean the same thing in sentences "He sat on the chair." and "She bought a new chair."?': 'Yes',
    'Does the word "book" mean the same thing in sentences "I read a book." and "She borrowed a book from the library."?': 'Yes',
    'Does the word "shoe" mean the same thing in sentences "My shoe is dirty." and "He tied his shoe."?': 'Yes',
    'Does the word "house" mean the same thing in sentences "They live in a big house." and "The house has a red door."?': 'Yes',
    'Does the word "tree" mean the same thing in sentences "A bird is sitting in the tree." and "The tree has green leaves."?': 'Yes',
    'Does the word "phone" mean the same thing in sentences "She lost her phone." and "He called me on his phone."?': 'Yes',
    'Does the word "apple" mean the same thing in sentences "I ate an apple." and "He owns Apple Inc."?': 'No',
    'Does the word "apple" mean the same thing in sentences "I ate a juicy apple." and "Steve Jobs founded Apple."?': 'No',
    'Does the word "bat" mean the same thing in sentences "He hit the ball with a wooden bat." and "The bat hung upside down in the cave."?': 'No',
    'Does the word "bass" mean the same thing in sentences "He caught a big bass fish." and "The bass in this song is really loud."?': 'No',
    'Does the word "bank" mean the same thing in sentences "She took out money from the bank." and "They had a picnic by the river bank."?': 'No',
    'Does the word "crane" mean the same thing in sentences "The crane lifted the steel beam." and "The crane flew over the lake."?': 'No',
    'Does the word "rock" mean the same thing in sentences "He threw a small rock at the can." and "She danced to rock music."?': 'No',
    'Does the word "jam" mean the same thing in sentences "She put strawberry jam on her toast." and "We were stuck in a traffic jam."?': 'No',
    'Does the word "spring" mean the same thing in sentences "Flowers bloom in spring." and "The mattress has a metal spring."?': 'No',
    'Does the word "watch" mean the same thing in sentences "I wear a watch on my wrist." and "Let’s watch a movie tonight."?': 'No',

}

with open('../text/obvious_questions.txt', 'w', encoding='utf-8') as file:
    file.write("\n".join(obvious_questions.keys()))
