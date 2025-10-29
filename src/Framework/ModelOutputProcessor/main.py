from Framework.ModelOutputProcessor.config import setClassificationRule
from src.Framework.ModelOutputProcessor.run import run

# --- CONFIG ---

TESTFILE_LENGTH = 1400

# --- end of config ---

def main():
    setClassificationRule()
    run()


if __name__ == '__main__':
    main()
