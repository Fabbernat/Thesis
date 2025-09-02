MODEL_NAME =  '                   '

MODEL_NAME = MODEL_NAME.strip()


def main():
    prompt = open('prompt.txt').read()
    chatGPT: ChatGPT = ChatGPT()
    chatGPT.ask(prompt)

if __name__ == '__main__':
    main()