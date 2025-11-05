from Framework.ModelInputPreparer.config import logPartialResults
from src.Framework.ModelInputPreparer.run import run

import sys
from time import perf_counter

print(' ** RUNTIME ENVIRONMENT INFO **')
print("sys.path: ", str(sys.path))
print(' ** end of runtime environment info ** ')

def main(GlobalRun=False):
    run(logPartialResults)

if __name__ == '__main__':
    startTime = perf_counter()
    main()
    endTime = perf_counter()
    print('Total runtime: ', (endTime-startTime))