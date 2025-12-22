endings = {
    0:'', 1:' with reasoning',
             2:' with a confidence score between 0 and 100. 100 means you are a hundred percent sure they mean the same thing in both sentences and 0 means the opposite',
    3: 'with reasoning and your confidence score of "Yes" in percentage. 100% means you are a hundred percent sure that they mean the same thing, 0% means the opposite',

}

# --- CONFIG ---

DEBUG_MODE = False
DETERMINISTIC_MODE = False
ISOLATED_MODE = True

FILE_NAME = 'data/questions.in'

NUMBER_OF_DESIRED_ANSWERS = 200 # mert 100 random kérdést nézünk egyenes és fordított sorrendben
if NUMBER_OF_DESIRED_ANSWERS % 2 != 0:
    key = input('Warning: cannot count consistency when odd number of lines, please fix input. The last line  will be dropped. Do you wish to continue? (y/n)')
    if key == 'n':
        exit(0)
    NUMBER_OF_DESIRED_ANSWERS -= 1

numberOfDesiredAnswersAsString = ['', f'{NUMBER_OF_DESIRED_ANSWERS // 2} ']

endOfSentence = endings[0]

if ISOLATED_MODE:
    INSTRUCTION = 'Answer the question with just a single `Yes` or `No`.'
else:
    INSTRUCTION = f'Answer all {numberOfDesiredAnswersAsString[1]}questions and their reversed pairs with just a single `Yes` or `No`{endOfSentence}.\n'

# --- end of config ---
if __name__ == '__main__':
    print(INSTRUCTION)