from src.Framework.ModelOutputProcessor.run import run

import sys

print(' ** RUNTIME ENVIRONMENT INFO **')
print("sys.path: ", str(sys.path))
print(' ** end of runtime environment info ** ')


def main():
    run()


if __name__ == '__main__':
    main()
