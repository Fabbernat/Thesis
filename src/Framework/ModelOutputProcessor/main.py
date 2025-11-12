from src.Framework.ModelOutputProcessor.run import run

import sys
from time import perf_counter

print(' ** RUNTIME ENVIRONMENT INFO **')
print("sys.path: ", str(sys.path))
print(' ** end of runtime environment info ** ')


def main(GlobalRun=False):
    run()


if __name__ == '__main__':

    startTime = perf_counter()
    main()
    endTime = perf_counter()
    print(f'Total runtime: {(endTime - startTime):.6f} seconds')
