import Resurrection.CloudRunnerNotebooks.run

MODEL_NAME =  '      Qwen/Qwen2.5-0.5B-Instruct           '
FILE_PATH = "prompt.txt"


MODEL_NAME = MODEL_NAME.strip()

def main():
    Resurrection.CloudRunnerNotebooks.run.run()

if __name__ == '__main__':
    main()