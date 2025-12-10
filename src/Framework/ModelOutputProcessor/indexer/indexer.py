def indexer(straightAnswers, reversedAnswers, guids, goldStandard: list[bool]):
    """"""

    if len(straightAnswers) != len(reversedAnswers) or len(straightAnswers) != len(guids) or len(reversedAnswers) != len(guids):
        import sys
        sys.stderr.write("Error: the lengths of the straight and reversed questions, as well as the GUIDs, must match! Do you wish to continue? (y/n)\n")
        key = input()
        if key.lower().strip() == 'n':
            exit(0)



    length = min(len(straightAnswers), len(reversedAnswers), len(guids), len(goldStandard)) # elvileg mindegyiknek ugyanolyan hosszúnak kéne lennie, de a gyakorlatban általában nem az.
    consistents = accurates = consistentlyAccurates = ambiguouses = consistentlyAmbiguouses = 0

    for i in range(length):


        token1 = straightAnswers[i]
        token2 = reversedAnswers[i]
        groundTruth = goldStandard[guids[i]]

        valid = {'Yes', 'No'}
        if token1 not in valid:
            ambiguouses += 1
        if token2 not in valid:
            ambiguouses += 1

        if token1 == 'Yes' and token2 == 'Yes':
            consistents += 1
        if token1 == 'No' and token2 == 'No':
            consistents += 1
        if token1 not in valid and token2 not in valid:
            consistents += 1
            consistentlyAmbiguouses += 1

        if token1 == 'Yes' and groundTruth:
            accurates += 1
        if token2 == 'Yes' and groundTruth:
            accurates += 1
        if token1 == 'No' and not groundTruth:
            accurates += 1
        if token2 == 'No' and not groundTruth:
            accurates += 1

        if token1 == 'Yes' and token2 == 'Yes' and groundTruth:
            consistentlyAccurates += 1
        if token1 == 'No' and token2 == 'No' and not groundTruth:
            consistentlyAccurates += 1


    return consistents / length, accurates / length, consistentlyAccurates / length, ambiguouses / length, consistentlyAmbiguouses / length # 0 és 1 közé normalizáljuk


def test_indexer():
    straightAnswers: dict[str, str] = {'sound': 'Yes','grow': 'No', 'audience':'No','insufficiency': 'No','batch': 'No','extent': 'No','agency': 'No','narcolepsy': 'No','score': 'Yes','instill': 'No','amount': 'No','generation': 'No','vagina': 'No','guard': 'No','allowance': 'No','site': 'No','eclat': 'No','compel': 'No','inwardness': 'Yes','height': 'No','fall': 'No','obstruction': 'Yes','agony': 'No','palpitate': 'No','logic': 'No','suspect': 'No','analyze': 'No','repair': 'No','stampede': 'No','retroversion': 'No','exploit': 'No','correct': 'No','shade': 'No','heat': 'Yes','demonstration': 'No','explode': 'No','mound': 'No','nursing': 'No','repression': 'No', 'ice':'No','lubricate': 'No','strain': 'No','construction': 'No','mate': 'No','sewer': 'No','origin': 'No','manner': 'No','model': 'No','bank': 'No'}
    reversedAnswers: dict[str, str] = {'sound':'No','grow': 'No','audience': 'No','insufficiency': 'No','batch': 'No','extent': 'No','extract': 'No','agency': 'No', 'narcolepsy':'Yes','score': 'No','instill': 'No','amount': 'No','generation': 'No','vagina':
                                  'No', 'guard':'No','allowance': 'No','site': 'No','eclat': 'No', 'compel':'No','inwardness': 'No','height': 'No','fall': 'No', 'obstruction':'No','agony': 'No','palpitate': 'No','logic': 'No',
                                 'suspect': 'No','analyze': 'No','repair': 'No','stampede': 'No', 'retroversion':'No','exploit': 'No','correct': 'No','shade': 'Yes','heat': 'No','demonstration': 'No','explode': 'No','mound': 'No','nursing': 'No',
                                  'repression':'No','ice': 'No','lubricate': 'Yes','strain': 'No', 'construction':'No','mate': 'No','sewer': 'No','origin': 'No', 'manner':'No', 'model': 'No','bank': 'No'}
    guids: list[int] = [964, 398, 22, 1282, 516, 902, 489, 911, 591, 1244, 1389, 178, 1291, 10, 355, 596, 615, 103, 1359, 58, 515, 1316, 468, 456, 1005, 31, 879, 303, 1315, 243, 1112, 847, 204, 1328, 1268, 1092, 1127, 291, 1188, 1323, 212, 80, 1357, 1003, 564, 659, 214, 894, 863, 834, 508, 715, 735, 361, 312, 435, 1256, 583, 1194, 702, 1145, 788, 20, 1286, 907, 438, 1351, 935, 251, 1327, 85, 250, 1072, 461, 665, 932, 304, 1107, 1214, 337, 368, 794, 12, 298, 988, 512, 895, 660, 380, 1067, 770, 687, 169, 268, 345, 1118, 388, 102, 810, 1028]
    print(len(straightAnswers), len(reversedAnswers), len(guids))

    goldStandard: list[bool] = []
    lines:list[str] = []

    from pathlib import Path
    basePath = Path(__file__).resolve()

    with open(basePath.parents[1] / 'data' / 'test.gold.in', 'r', encoding='utf-8') as goldFile:
        lines = goldFile.readlines()

    goldStandard = [line.strip() == 'T' for line in lines]

    
    consistents, accurates, consistentlyAccurates, ambiguouses, consistentlyAmbiguouses = indexer(straightAnswers, reversedAnswers, guids, goldStandard)

    here = Path(__file__).resolve().parent

    root = here.parent

    logPath = root / "data" / "logFile.out"

    with open(logPath, "a", encoding="utf-8") as f:
        printEverywhere('Answers ratios (True/all)', f)
        printEverywhere(f'Consistent: {consistents * 100:.2f} %', f)
        printEverywhere(f'Accurate: {accurates * 100:.2f} %', f)
        printEverywhere(f'Consistently accurate: {consistentlyAccurates * 100:.2f} %', f)
        printEverywhere(f'Ambiguous: {ambiguouses * 100:.2f} %', f)
        printEverywhere(f'Consistently ambiguous: {consistentlyAmbiguouses * 100:.2f} %', f)

def printEverywhere(msg: str, file):
    print(msg)
    file.write(msg + "\n")

def main():
    test_indexer()
if __name__ == '__main__':
    main()
