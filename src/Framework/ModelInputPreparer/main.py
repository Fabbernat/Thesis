from src.Framework.ModelInputPreparer import config

try:
    from src.Framework.ModelInputPreparer.config import logPartialResults
    from src.Framework.ModelInputPreparer.run import run
    from src.Framework.globalMain import CONNECTED
except ModuleNotFoundError:
    from run import run
    CONNECTED = False
import sys

print(' ** RUNTIME ENVIRONMENT INFO **')
print("sys.path: ", str(sys.path))
print(' ** end of runtime environment info ** ')

def main():
    run(config.logPartialResults)

if __name__ == '__main__':
    main()