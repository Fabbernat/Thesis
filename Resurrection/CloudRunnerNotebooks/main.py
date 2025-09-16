import run

MODEL_NAME =  '      Qwen/Qwen2.5-0.5B-Instruct           '
FILE_PATH = "prompt.txt"
DEVICE = "cuda"

MODEL_NAME = MODEL_NAME.strip().lower()

def main():
    run.run()

if __name__ == '__main__':
    main()