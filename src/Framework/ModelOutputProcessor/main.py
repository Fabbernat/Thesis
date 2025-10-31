from Framework.ModelOutputProcessor.config import setClassificationRule
from src.Framework.ModelOutputProcessor.run import run

import sys

print(' ** RUNTIME ENVIRONMENT INFO **')
print("sys.path: ", str(sys.path))
print(' ** end of runtime environment info ** ')

# --- CONFIG ---

TESTFILE_LENGTH = 1400

# --- end of config ---

def main():
    run()


if __name__ == '__main__':
    main()
