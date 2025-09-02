from Resurrection.CloudRunnerNotebooks.LanguageModels import Models
from Resurrection.CloudRunnerNotebooks.LanguageModels.Model import Model

MODEL_NAME =  '                   '

MODEL_NAME = MODEL_NAME.strip()




def main():
    prompt = open('prompt.txt').read()
    model = Models.get(MODEL_NAME)
    model.ask('Answer all questions with Yes or No!\n'.join(prompt))

if __name__ == '__main__':
    main()