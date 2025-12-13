try:
    from src.Framework.ModelOutputProcessor.run import run
    from src.Framework.ModelOutputProcessor.indexer import Qwen05BIndexer, Qwen15BIndexer, GoogleIndexer
except Exception:
    from run import run
    from indexer import Qwen05BIndexer, Qwen15BIndexer, GoogleIndexer
import sys
from time import perf_counter

print(' ** RUNTIME ENVIRONMENT INFO **')
print("sys.path: ", str(sys.path))
print(' ** end of runtime environment info ** ')


def main(GlobalRun=False, model = 'google'):
    if model == 'qwen05':
        Qwen05BIndexer.main()
    elif model == 'qwen15':
        Qwen15BIndexer.main()
    elif model == 'google':
        GoogleIndexer.main()
    else:
        run()


if __name__ == '__main__':

    startTime = perf_counter()
    main()
    endTime = perf_counter()
    print(f'Total runtime: {(endTime - startTime):.6f} seconds')
