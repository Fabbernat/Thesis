from src.Framework.ModelInputPreparer.run import run

import sys

print(' ** RUNTIME ENVIRONMENT INFO **')
print("sys.path: ", str(sys.path))
print(' ** end of runtime environment info ** ')

def main():
    # --- CONFIG ---
    logPartialResults = False
    run(logPartialResults)
    # --- end of config ---

if __name__ == '__main__':
    main()