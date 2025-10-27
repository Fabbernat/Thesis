from src.Framework.ModelOutputProcessor.run import run

TESTFILE_LENGTH = 1400

# --- CONFIG ---
AFFIRMATIVE_KEYWORDS = ['Yes']
NEGATIVE_KEYWORDS = ['No']

AFFIRMATIVE_PHRASES = ['Of Course']
NEGATIVE_PHRASES = ['Not at all']
# --- end of config ---

def main():
    run()


if __name__ == '__main__':
    main()
