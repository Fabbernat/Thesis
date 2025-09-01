MODEL_NAME =  '                   '

MODEL_NAME = MODEL_NAME.strip()

class ChatGPT:
    def ask(self, question: str) -> str:
        pass

def main():
    prompt = open('prompt.txt').read()
    chatGPT: ChatGPT = ChatGPT()
    chatGPT.ask(prompt)

if __name__ == '__main__':
    main()