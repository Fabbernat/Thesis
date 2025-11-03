import traceback

class FileReader:
    def readWholeFile(self, file):
        return [line.strip() for line in file]

class TestFilesMerger:
    def mergeTestfiles(self) -> str:
        try:
            if CONNECTED:
                filePath = '../globalData/1/test.data.in'
            else:
                filePath = 'data/test.data.in'
            with open(filePath, 'r') as testDataFile, open('data/test.gold.in', 'r') as testGoldFile:
                fileReader: FileReader = FileReader()
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