import traceback
from pathlib import Path
from typing import Any

from numpy.ma.core import max_val


class FileReader:
    def readWholeFile(self, file) -> list[Any]:
        return [line.strip() for line in file]

    def readEntries(self, file, randomEntries: list[int]) -> list[Any]:
        """Read the lines at the specified indices and save the indices to a file."""
        lines = [line.strip() for line in file]
        basePath = Path(__file__).parent.parent

        results = []
        with open(basePath / 'data' / 'indices.out', 'a') as indicesFile:
            for i in randomEntries:
                if i < len(lines):
                    results.append(lines[i])
                    print(i, file=indicesFile)

        return results


class TestFilesMerger:
    def mergeTestfiles(self) -> str:
        basePath = Path(__file__).parent.parent
        dataFilePath = basePath / 'data' / 'test.data.in'
        goldFilePath = basePath / 'data' / 'test.gold.in'
        with open(dataFilePath, 'r') as testDataFile, open(goldFilePath, 'r') as testGoldFile:
            fileReader: FileReader = FileReader()
            try:
                from src.Framework.ModelInputPreparer.config import RANDOM_SAMPLES
            except Exception:
                from config import RANDOM_SAMPLES
            if RANDOM_SAMPLES:
                try:
                    from src.Framework.ModelInputPreparer.randomsamples import randomEntries
                except Exception:
                    from randomsamples import randomEntries
                dataLines = [line.strip() for line in testDataFile]
                goldLines = [line.strip() for line in testGoldFile]
                maxValid = min(len(dataLines), len(goldLines))
                filteredEntries = [i for i in randomEntries if 0 <= i < maxValid]

                # Because readEntries() reads from the file pointer, we must:
                #
                # Reset (seek(0)) both files.
                #
                # Pass the filtered indices into `readEntries`
                testDataFile.seek(0)
                testGoldFile.seek(0)
                rawTestDataValues = fileReader.readEntries(testDataFile, filteredEntries)
                rawTestGoldValues = fileReader.readEntries(testGoldFile, filteredEntries)
            else:
                testDataFile.seek(0)
                testGoldFile.seek(0)
                rawTestDataValues = fileReader.readWholeFile(testDataFile)
                rawTestGoldValues = fileReader.readWholeFile(testGoldFile)

            mergedTestValues = []

            for dataRow, goldRow in zip(rawTestDataValues, rawTestGoldValues):
                mergedTestValues.append(f'{dataRow}\t{goldRow}')
            return '\n'.join(mergedTestValues)
