endings = {
    0:'', 1:' with reasoning',
             2:' with a confidence score between 0 and 100. 100 means you are a hundred percent sure they mean the same thing in both sentences and 0 means the opposite',
    3: 'with reasoning and your confidence score of "Yes" in percentage. 100% means you are a hundred percent sure that they mean the same thing, 0% means the opposite',

}

# --- CONFIG ---

DEBUG_MODE = True

FILE_NAME = 'data/questions.in'

NUMBER_OF_DESIRED_ANSWERS = 60
if NUMBER_OF_DESIRED_ANSWERS % 2 != 0:
    key = input('Warning: cannot count consistency when odd number of lines, please fix input. The last line  will be dropped. Do you wish to continue? (y/n)')
    if key == 'n':
        exit(0)
    NUMBER_OF_DESIRED_ANSWERS -= 1

numberOfDesiredAnswersAsString = ['', f'{NUMBER_OF_DESIRED_ANSWERS} ']

endOfSentence = endings[0]
INSTRUCTION = f'Answer all {numberOfDesiredAnswersAsString[0]}questions with either `Yes` or `No`{endOfSentence}.\n'

# --- end of config ---
print(INSTRUCTION)