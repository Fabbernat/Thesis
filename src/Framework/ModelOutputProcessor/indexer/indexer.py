def indexer(qwenStraightAnswers: dict[str, str],
            qwenReversedAnswers: dict[str, str],
            guids: list[int],
            goldStandard: list[bool]):
    """"""
    keys = list(qwenStraightAnswers.keys())

    if len(qwenStraightAnswers) != len(qwenReversedAnswers) or len(qwenStraightAnswers) != len(guids) or len(
            qwenReversedAnswers) != len(guids):
        import sys
        sys.stderr.write(
            "Error: the lengths of the straight and reversed questions, as well as the GUIDs, must match! Do you wish to continue? (y/n) ")
        key = input()
        if key.lower().strip() == 'n':
            exit(0)

    length = min(len(qwenStraightAnswers), len(qwenReversedAnswers), len(guids),
                 len(goldStandard))  # elvileg mindegyiknek ugyanolyan hosszúnak kéne lennie, de a gyakorlatban általában nem az.
    consistents = accurates = consistentlyAccurates = ambiguouses = consistentlyAmbiguouses = 0

    for i in range(length):

        key = keys[i]  # pl. 'sound'

        token1 = qwenStraightAnswers.get(key, None)
        if token1 is None:
            ambiguouses += 1
            token1 = '?'
        token2 = qwenReversedAnswers.get(key, None)
        if token2 is None:
            ambiguouses += 1
            token2 = '?'

        groundTruth = goldStandard[guids[i]]  # egész szám index a goldStandard listához

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

        # helyes-e a straight válasz?
        straightCorrect = (
                (token1 == 'Yes' and groundTruth) or
                (token1 == 'No' and not groundTruth)
        )

        # helyes-e a reversed válasz?
        reversedCorrect = (
                (token2 == 'Yes' and groundTruth) or
                (token2 == 'No' and not groundTruth)
        )

        if straightCorrect or reversedCorrect:
            accurates += 1

        if token1 == 'Yes' and token2 == 'Yes' and groundTruth:
            consistentlyAccurates += 1
        if token1 == 'No' and token2 == 'No' and not groundTruth:
            consistentlyAccurates += 1

    return consistents / length, accurates / length, consistentlyAccurates / length, ambiguouses / length, consistentlyAmbiguouses / length  # 0 és 1 közé normalizáljuk


def test_indexer():
    qwenStraightAnswers: dict[str, str] = {
        'sound': 'Yes', 'grow': 'No', 'audience': 'No',
        'insufficiency': 'No',
        'batch': 'No', 'extent': 'No','extract':'No' , 'agency': 'No', 'narcolepsy': 'No',
        'score': 'Yes', 'instill': 'No', 'amount': 'No', 'generation': 'No',
        'vagina': 'No', 'guard': 'No', 'allowance': 'No', 'site': 'No', 'eclat': 'No',
        'compel': 'No', 'inwardness': 'Yes', 'height': 'No', 'fall': 'No',
        'obstruction': 'Yes', 'agony': 'No', 'palpitate': 'No', 'logic': 'No',
        'suspect': 'No', 'analyze': 'No', 'repair': 'No', 'stampede': 'No',
        'retroversion': 'No', 'exploit': 'No', 'correct': 'No', 'shade': 'No',
        'heat': 'Yes', 'demonstration': 'No', 'explode': 'No', 'mound': 'No',
        'nursing': 'No', 'repression': 'No', 'ice': 'No', 'lubricate': 'No',
        'strain': 'No', 'construction': 'No', 'mate': 'No', 'sewer': 'No',
        'origin': 'No', 'manner': 'No', 'model': 'No', 'bank': 'No'}
    qwenReversedAnswers: dict[str, str] = {
        'sound': 'No', 'grow': 'No', 'audience': 'No', 'insufficiency': 'No',
        'batch': 'No', 'extent': 'No', 'extract': 'No', 'agency': 'No',
        'narcolepsy': 'Yes', 'score': 'No', 'instill': 'No', 'amount': 'No',
        'generation': 'No', 'vagina':
            'No', 'guard': 'No', 'allowance': 'No', 'site': 'No', 'eclat': 'No',
        'compel': 'No', 'inwardness': 'No', 'height': 'No', 'fall': 'No',
        'obstruction': 'No', 'agony': 'No', 'palpitate': 'No', 'logic': 'No',
        'suspect': 'No', 'analyze': 'No', 'repair': 'No', 'stampede': 'No',
        'retroversion': 'No', 'exploit': 'No', 'correct': 'No', 'shade': 'Yes',
        'heat': 'No', 'demonstration': 'No', 'explode': 'No', 'mound': 'No',
        'nursing': 'No',
        'repression': 'No', 'ice': 'No', 'lubricate': 'Yes', 'strain': 'No',
        'construction': 'No', 'mate': 'No', 'sewer': 'No', 'origin': 'No',
        'manner': 'No', 'model': 'No', 'bank': 'No'}
    guids: list[int] = [
        137,
        399,
        1041,
        1173,
        910,
        949,
        866,
        967,
        498,
        170,
        1110,
        1016,
        1212,
        378,
        755,
        42,
        344,
        1020,
        1019,
        689,
        956,
        1127,
        520,
        948,
        306,
        89,
        1278,
        773,
        1387,
        196,
        337,
        476,
        1292,
        451,
        462,
        625,
        1295,
        1064,
        1391,
        1090,
        744,
        113,
        1290,
        429,
        29,
        761,
        780,
        616,
        1213,
        204
    ]
    googleStraightAnswers: dict[str, str] = {
        'sound':'No',
        '':'No',
        '': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
        'extract': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
    }
    googleReversedAnswers: dict[str, str] = {
        'sound':'No',
        '':'No',
        '': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
        '': 'No',
    }
    print(len(qwenStraightAnswers), len(qwenReversedAnswers), len(guids))

    goldStandard: list[bool] = []
    lines: list[str] = []

    from pathlib import Path
    basePath = Path(__file__).resolve()

    with open(basePath.parents[1] / 'data' / 'test.gold.in', 'r', encoding='utf-8') as goldFile:
        lines = goldFile.readlines()

    goldStandard = [line.strip() == 'T' for line in lines] # biztosítja, hogy bool legyen, True ha 'T', egyébként False

    consistents, accurates, consistentlyAccurates, ambiguouses, consistentlyAmbiguouses = indexer(qwenStraightAnswers,
                                                                                                  qwenReversedAnswers,
                                                                                                  guids, goldStandard)

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
        # printEverywhere(f'Ratio of "Yes" answers', f)
        # printEverywhere(f'Ratio of "No" answers', f)
        # printEverywhere(f'True positives', f)
        # printEverywhere(f'fn, fp, tn', f)



def printEverywhere(msg: str, file):
    print(msg)
    file.write(msg + "\n")


def main():
    test_indexer()


if __name__ == '__main__':
    main()
