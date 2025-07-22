import typing

class TestFilesMerger:
    def mergeTestfiles(self):
        with open("test.data.txt" and "test.gold.txt") as testDataFile and testGoldFile:
            fileReader: FileReader = new FileReader()
            rawTestDataValues = fileReader.readWholeFile(testDataFile)
            rawTestGoldValues = fileReader.readWholeFile(testGoldFile)

            mergedTestValues = ""
            for dataRow, goldRow in rawTestDataValues, rawTestGoldValues:
                mergedTestValues.join(f'{dataRow}\t{goldRow}')
            return mergedTestValues
        except Exception:
            print("Something went wrong")