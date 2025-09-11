def getMessagesAsString():
    promptFileContents = open("../prompt.txt").read()
    messages = [
        {"role": "system", "content": "Answer all questions with Yes or No!"},
        {"role": "user", "content": promptFileContents},
    ]

    return "".join(messages[0]["content"] + promptFileContents)
