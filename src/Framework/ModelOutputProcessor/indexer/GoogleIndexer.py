def indexer(googleStraightAnswers: dict[str, str],
            googleReversedAnswers: dict[str, str],
            guids: list[int],
            goldStandard: list[bool]):
    """"""
    keys = list(googleStraightAnswers.keys())

    if len(googleStraightAnswers) != len(googleReversedAnswers) or len(googleStraightAnswers) != len(guids) or len(
            googleReversedAnswers) != len(guids):
        import sys
        sys.stderr.write(
            "Error: the lengths of the straight and reversed questions, as well as the GUIDs, must match! Do you wish to continue? (y/n) ")
        key = input()
        if key.lower().strip() == 'n':
            exit(0)

    length = min(len(googleStraightAnswers), len(googleReversedAnswers), len(guids),
                 len(goldStandard))  # elvileg mindegyiknek ugyanolyan hosszúnak kéne lennie, de a gyakorlatban általában nem az.
    consistents = accurates = consistentlyAccurates = ambiguouses = consistentlyAmbiguouses = 0

    for i in range(length):

        key = keys[i]  # pl. 'sound'

        token1 = googleStraightAnswers.get(key, None)
        if token1 is None:
            ambiguouses += 1
            token1 = '?'
        token2 = googleReversedAnswers.get(key, None)
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
    googleStraightAnswers: dict[str, str] = {
        'sound': 'Yes',
        'grow': 'No',
        'audience': 'No',
        'insufficiency': 'No',
        'batch': 'No',
        'extent': 'Yes',
        'extract': 'No',
        'agency': 'No',
        'narcolepsy': 'No',
        'score': 'Yes',
        'instill': 'No',
        'amount': 'No',
        'generation': 'No',
        'vagina': 'Yes',
        'guard': 'Yes',
        'allowance': 'No',
        'site': 'Yes',
        'eclat': 'No',
        'compel': 'No',
        'inwardness': 'No',
        'height': 'Yes',
        'fall': 'No',
        'obstruction': 'Yes',
        'agony': 'Yes',
        'palpitate': 'Yes',
        'logic': 'No',
        'suspect': 'No',
        'analyze': 'No',
        'repair': 'Yes',
        'stampede': 'Yes',
        'retroversion': 'No',
        'exploit': 'No',
        'correct': 'No',
        'shade': 'Yes',
        'heat': 'Yes',
        'demonstration': 'No',
        'explode': 'Yes',
        'mound': 'Yes',
        'nursing': 'No',
        'repression': 'Yes',
        'ice': 'No',
        'lubricate': 'Yes',
        'strain': 'No',
        'construction': 'No',
        'mate': 'No',
        'sewer': 'No',
        'origin': 'No',
        'manner': 'No',
        'model': 'Yes',
        'bank': 'Yes',
        'deflate': 'Yes',
        'local': 'Yes',
        'drive': 'No',
        'have': 'No',
        'inspire': 'No',
        'afterthought': 'No',
        'property': 'No', 'awareness': 'No',
        'prefer': 'Yes', 'bend': 'No', 'mark': 'Yes',
        'have2': 'No', 'rounding': 'No',
        'steamroller': 'No', 'zero': 'No', 'nest': 'No',
        'land': 'No', 'deliberation': 'No',
        'consist': 'No', 'restraint ': 'No', 'feedstock': 'Yes',
        'engage': 'Yes', 'sneak': 'No',
        'justify': 'Yes', 'grain': 'No', 'pass': 'No',
        'topic': 'No', 'holder': 'Yes',
        'crystallize': 'No', 'recapitulate': 'No', 'rag': 'No',
        'complaint': 'No', 'fiddle': 'No',
        'wax': 'No', 'tease': 'No', 'access': 'No',
        'union': 'No', 'cross': 'No',
        'morale': 'No', 'back': 'No', 'bother': 'No',
        'organize': 'No', 'dash': 'No',
        'loop': 'No', 'resolve': 'No', 'underlay': 'No',
        'submit': 'No', 'blood': 'Yes',
        'violence': 'No', 'lot': 'No',
    }
    googleReversedAnswers: dict[str, str] = {
        'sound': 'Yes',
        'grow': 'No',
        'audience': 'No',
        'insufficiency': 'No',
        'batch': 'Yes',
        'extent': 'No',
        'extract': 'No',
        'agency': 'No',
        'narcolepsy': 'No',
        'score': 'No',
        'instill': 'No',
        'amount': 'No',
        'generation': 'No',
        'vagina': 'Yes',
        'guard': 'Yes',
        'allowance': 'No',
        'site': 'Yes',
        'eclat': 'No',
        'compel': 'No',
        'inwardness': 'No',
        'height': 'Yes',
        'fall': 'No',
        'obstruction': 'Yes',
        'agony': 'Yes',
        'palpitate': 'Yes',
        'logic': 'Yes',
        'suspect': 'Yes',
        'analyze': 'Yes',
        'repair': 'Yes',
        'stampede': 'Yes',
        'retroversion': 'No',
        'exploit': 'No',
        'correct': 'No',
        'shade': 'Yes',
        'heat': 'Yes',
        'demonstration': 'No',
        'explode': 'No',
        'mound': 'No',
        'nursing': 'Yes',
        'repression': 'Yes',
        'ice': 'No',
        'lubricate': 'No',
        'strain': 'No',
        'construction': 'No',
        'mate': 'No',
        'sewer': 'No',
        'origin': 'No',
        'manner': 'No',
        'model': 'No',
        'bank': 'No',
        'deflate': 'No', 'local': 'No',
        'drive': 'No', 'have': 'No',
        'inspire': 'No', 'afterthought': 'No',
        'property': 'No', 'awareness': 'No',
        'prefer': 'Yes', 'bend': 'No', 'mark': 'Yes',
        'have2': 'No', 'rounding': 'No',
        'steamroller': 'No', 'zero': 'No', 'nest': 'No',
        'land': 'No', 'deliberation': 'No',
        'consist': 'No', 'restraint ': 'No', 'feedstock': 'Yes',
        'engage': 'Yes', 'sneak': 'No',
        'justify': 'Yes', 'grain': 'No', 'pass': 'No',
        'topic': 'No', 'holder': 'Yes',
        'crystallize': 'No', 'recapitulate': 'No', 'rag': 'No',
        'complaint': 'No', 'fiddle': 'No',
        'wax': 'No', 'tease': 'No', 'access': 'No',
        'union': 'No', 'cross': 'No',
        'morale': 'No', 'back': 'No', 'bother': 'No',
        'organize': 'No', 'dash': 'No',
        'loop': 'No', 'resolve': 'No', 'underlay': 'No',
        'submit': 'No', 'blood': 'Yes',
        'violence': 'No', 'lot': 'No',
    }
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
        349,
        1381,
        457,
        296,
        1036,
        1302,
        471,
        1091,
        201,
        974,
        570,
        1105,
        188,
        264,
        438,
        715,
        230,
        1267,
        4,
        297,
        97,
        489,
        371,
        48,
        1277,
        1080,
        839,
        244,
        1112,
        859,
        1100,
        40,
        1142,
        832,
        939,
        1044,
        1262,
        243,
        1334,
        69,
        435,
        1170,
        142,
        321,
        1119,
        1037,
        335,
        418,
        300,
        357,
        387,
    ]

    print(len(googleStraightAnswers), len(googleReversedAnswers), len(guids))

    goldStandard: list[bool] = []
    lines: list[str] = []

    from pathlib import Path
    basePath = Path(__file__).resolve()

    with open(basePath.parents[1] / 'data' / 'test.gold.in', 'r', encoding='utf-8') as goldFile:
        lines = goldFile.readlines()

    goldStandard = [line.strip() == 'T' for line in lines] # biztosítja, hogy bool legyen, True ha 'T', egyébként False

    consistents, accurates, consistentlyAccurates, ambiguouses, consistentlyAmbiguouses = indexer(googleStraightAnswers,
                                                                                                  googleReversedAnswers,
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

