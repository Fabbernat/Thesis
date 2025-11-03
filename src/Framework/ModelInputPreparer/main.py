from src.Framework.ModelInputPreparer import config

try:
    from src.Framework.ModelInputPreparer.config import LOG_PARTIAL_RESULTS
    from src.Framework.ModelInputPreparer.run import run
    from src.Framework.globalMain import CONNECTED
except ModuleNotFoundError:
    from .run import run
    CONNECTED = False
import sys

print(' ** RUNTIME ENVIRONMENT INFO **')
print("sys.path: ", str(sys.path))
print(' ** end of runtime environment info ** ')

def main():
    run(config.LOG_PARTIAL_RESULTS)

if __name__ == '__main__':
    main()