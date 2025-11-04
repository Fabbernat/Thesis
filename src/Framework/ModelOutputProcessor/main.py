from src.Framework.ModelOutputProcessor.run import run
try:
    from src.Framework.globalMain import CONNECTED
except ModuleNotFoundError:
    CONNECTED = False

import sys

print(' ** RUNTIME ENVIRONMENT INFO **')
print("sys.path: ", str(sys.path))
print(' ** end of runtime environment info ** ')


def main(GlobalRun=False):
    run()


if __name__ == '__main__':
    main()
