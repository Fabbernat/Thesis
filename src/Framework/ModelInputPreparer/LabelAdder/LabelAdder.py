import traceback
from pathlib import Path
from typing import Any


class FileReader:
    def readWholeFile(self, file) -> list[Any]:
        return [line.strip() for line in file]

    def readEntries(self, file, randomEntries: list[int]) -> list[Any]:
        """Read only the lines at the specified indices."""
        lines = [line.strip() for line in file]

        return [lines[i] for i in randomEntries if i < len(lines)]


class TestFilesMerger:
    def mergeTestfiles(self) -> str:
        try:
            basePath = Path(__file__).parent.parent
            dataFilePath = Path(str(basePath) + r'\data\test.data.in')
            goldFilePath = Path(str(basePath) + r'\data\test.gold.in')
            with open(dataFilePath, 'r') as testDataFile, open(goldFilePath, 'r') as testGoldFile:
                fileReader: FileReader = FileReader()
                try:
                    from src.Framework.ModelInputPreparer.config import RANDOM_SAMPLES
                except Exception:
                    from Framework.ModelInputPreparer.config import RANDOM_SAMPLES
                if RANDOM_SAMPLES:
                    from Framework.ModelInputPreparer.randomsamples import randomEntries
                    rawTestDataValues =fileReader.readEntries(testDataFile, randomEntries)
                    rawTestGoldValues =fileReader.readEntries(testGoldFile, randomEntries)
                else:
                    rawTestDataValues = fileReader.readWholeFile(testDataFile)
                    rawTestGoldValues = fileReader.readWholeFile(testGoldFile)

                mergedTestValues = []

                for dataRow, goldRow in zip(rawTestDataValues, rawTestGoldValues):
                    mergedTestValues.append(f'{dataRow}\t{goldRow}')
                return '\n'.join(mergedTestValues)
        except Exception as e:
            traceback.print_exc()
            print('The file could not be opened.', e)
            return ''