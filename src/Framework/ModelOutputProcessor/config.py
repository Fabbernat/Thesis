MODEL_PATH = "data/modelResponses.in"
GOLD_PATH = "data/test.gold.in"

keywordPairs = {
    "yes": ["Yes", "No", ],
    "yes.": ["Yes.", "No."],
    "arrowedYes": ["->Yes", "->No"],
    "t": ["T", "F"],
    "true": ["True", "False"],
    "true.": ["True.", "False"],
}

AFFIRMATIVE_KEYWORDS = [keywordPairs["yes"][0], keywordPairs["yes."][0]]
print(AFFIRMATIVE_KEYWORDS)
NEGATIVE_KEYWORDS = [keywordPairs["yes"][1], keywordPairs["yes."][1]]
print(NEGATIVE_KEYWORDS)

AFFIRMATIVE_PHRASES = ['Of Course']
NEGATIVE_PHRASES = ['Not at all']

def setClassificationRule():
    pass