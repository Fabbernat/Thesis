try:
    from src.Framework.ModelOutputProcessor.run import run
    from src.Framework.ModelOutputProcessor.indexer import indexer
except Exception:
    from run import run
    from indexer import indexer
import sys
from time import perf_counter

print(' ** RUNTIME ENVIRONMENT INFO **')
print("sys.path: ", str(sys.path))
print(' ** end of runtime environment info ** ')


def main(GlobalRun=False, index = True):
    if index:
        indexer.main()
    else:
        run()


if __name__ == '__main__':

    startTime = perf_counter()
    main()
    endTime = perf_counter()
    print(f'Total runtime: {(endTime - startTime):.6f} seconds')
