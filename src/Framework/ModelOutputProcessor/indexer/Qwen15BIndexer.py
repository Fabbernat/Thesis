modelSize = 1.5 # 0.5 for Qwen/Qwen2.5-0.5B-Instruct, 1.5 for Qwen/Qwen2.5-1.5B-Instruct

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
    consistentPairs = accurates = accuratePairs = consistentlyAccuratePairsPairs = ambiguousPairs = consistentlyambiguousPairs = yeses = nos = 0 # Mivel párokra van értve, Mindegyik végeredménye maximum `length` lehet.

    for i in range(length):

        key = keys[i]  # pl. 'sound'

        token1 = qwenStraightAnswers.get(key, None)
        if token1 is None:
            ambiguousPairs += 1
            token1 = '?'
        token2 = qwenReversedAnswers.get(key, None)
        if token2 is None:
            ambiguousPairs += 1
            token2 = '?'

        groundTruth = goldStandard[guids[i]]  # egész szám index a goldStandard listához

        valid = {'Yes', 'No'}
        if token1 not in valid:
            ambiguousPairs += 1
        if token2 not in valid:
            ambiguousPairs += 1

        if token1 == 'Yes':
            yeses += 1
        if token1 == 'No':
            nos += 1
        if token2 == 'Yes':
            yeses += 1
        if token2 == 'No':
            nos += 1

        if token1 == 'Yes':
            yeses += 1
        if token1 == 'No':
            nos += 1
        if token2 == 'Yes':
            yeses += 1
        if token2 == 'No':
            nos += 1

        if token1 == 'Yes' and token2 == 'Yes':
            consistentPairs += 1
        if token1 == 'No' and token2 == 'No':
            consistentPairs += 1
        if token1 not in valid and token2 not in valid:
            consistentPairs += 1
            consistentlyambiguousPairs += 1

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

        if straightCorrect:
            accurates += 0.5
        if reversedCorrect:
            accurates += 0.5

        if straightCorrect or reversedCorrect:
            accuratePairs += 1

        if token1 == 'Yes' and token2 == 'Yes' and groundTruth:
            consistentlyAccuratePairsPairs += 1
        if token1 == 'No' and token2 == 'No' and not groundTruth:
            consistentlyAccuratePairsPairs += 1

    return consistentPairs / length, accurates / length, accuratePairs / length, consistentlyAccuratePairsPairs / length, ambiguousPairs / length, consistentlyambiguousPairs / length, yeses, nos # # 0 és 1 közé normalizáljuk, kivéve a yeseket és a nokat

def test_indexer():
    qwenStraightAnswers: dict[str, str] = {
        'sound': 'Yes', 'grow': 'No', 'audience': 'No',
        'insufficiency': 'No',
        'batch': 'No', 'extent': 'No', 'extract': 'No', 'agency': 'No', 'narcolepsy': 'No',
        'score': 'Yes', 'instill': 'No', 'amount': 'No', 'generation': 'No',
        'vagina': 'No', 'guard': 'No', 'allowance': 'No', 'site': 'No', 'eclat': 'No',
        'compel': 'No', 'inwardness': 'Yes', 'height': 'No', 'fall': 'No',
        'obstruction': 'Yes', 'agony': 'No', 'palpitate': 'No', 'logic': 'No',
        'suspect': 'No', 'analyze': 'No', 'repair': 'No', 'stampede': 'No',
        'retroversion': 'No', 'exploit': 'No', 'correct': 'No', 'shade': 'No',
        'heat': 'Yes', 'demonstration': 'No', 'explode': 'No', 'mound': 'No',
        'nursing': 'No', 'repression': 'No', 'ice': 'No', 'lubricate': 'No',
        'strain': 'No', 'construction': 'No', 'mate': 'No', 'sewer': 'No',
        'origin': 'No', 'manner': 'No', 'model': 'No', 'bank': 'No',
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
        'manner': 'No', 'model': 'No', 'bank': 'No',
        'deflate': 'No', 'local': 'No',
        'drive': 'No', 'have': 'No',
        'inspire': 'Yes', 'afterthought': 'No',
        'property': 'No', 'awareness': 'No',
        'prefer': 'No', 'bend': 'No', 'mark': 'No',
        'have2': 'Yes', 'rounding': 'Yes',
        'steamroller': 'No', 'zero': 'No', 'nest': 'No',
        'land': 'Yes', 'deliberation': 'No',
        'consist': 'No', 'restraint ': 'No', 'feedstock': 'No',
        'engage': 'No', 'sneak': 'No',
        'justify': 'No', 'grain': 'No', 'pass': 'No',
        'topic': 'No', 'holder': 'No',
        'crystallize': 'No', 'recapitulate': 'No', 'rag': 'No',
        'complaint': 'No', 'fiddle': 'No',
        'wax': 'No', 'tease': 'No', 'access': 'No',
        'union': 'Yes', 'cross': 'No',
        'morale': 'No', 'back': 'No', 'bother': 'No',
        'organize': 'No', 'dash': 'No',
        'loop': 'No', 'resolve': 'No', 'underlay': 'Yes',
        'submit': 'No', 'blood': 'No',
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

    print(len(qwenStraightAnswers), len(qwenReversedAnswers), len(guids))

    goldStandard: list[bool] = []
    lines: list[str] = []

    from pathlib import Path
    basePath = Path(__file__).resolve()

    with open(basePath.parents[1] / 'data' / 'test.gold.in', 'r', encoding='utf-8') as goldFile:
        lines = goldFile.readlines()

    goldStandard = [line.strip() == 'T' for line in lines] # biztosítja, hogy bool legyen, True ha 'T', egyébként False

    consistentPairs, accurates, accuratePairs, consistentlyAccuratePairsPairs, ambiguousPairs, consistentlyambiguousPairs, yeses, nos = indexer(qwenStraightAnswers,
                                                                                                  qwenReversedAnswers,
                                                                                                  guids, goldStandard)

    here = Path(__file__).resolve().parent

    root = here.parent

    logPath = root / "data" / "logFile.out"

    with open(logPath, "a", encoding="utf-8") as f:
        from datetime import datetime
        now = datetime.now()
        formattedDate = now.strftime('%Y. %m. %d. %H:%M')
        try:
            from src.Framework.ModelOutputProcessor.config import USERNAME
        except Exception:
            try:
                from Framework.ModelOutputProcessor.config import USERNAME
            except Exception:
                from config import USERNAME
        printEverywhere(f'{USERNAME} ran Qwen/Qwen2.5-1.5B-Instruct at {formattedDate}', f)
        printEverywhere('Answers ratios (True/all)', f)
        printEverywhere(f'Consistent pairs: {consistentPairs * 100:.2f} %', f)
        printEverywhere(f'Accurates: {accurates * 100:.2f} %', f)
        printEverywhere(f'Accurate Pairs {accuratePairs * 100:.2f} %', f)
        printEverywhere(f'Consistently accurate pairs: {consistentlyAccuratePairsPairs * 100:.2f} %', f)
        printEverywhere(f'Ambiguous pairs: {ambiguousPairs * 100:.2f} %', f)
        printEverywhere(f'Consistently ambiguous pairs: {consistentlyambiguousPairs * 100:.2f} %', f)
        printEverywhere(f'Ratio of "Yes" answers {yeses}', f)
        printEverywhere(f'Ratio of "No" answers {nos}', f)
        # printEverywhere(f'True positives', f)
        # printEverywhere(f'fn, fp, tn', f)



def printEverywhere(msg: str, file):
    print(msg)
    file.write(msg + "\n")


def main():
    test_indexer()


if __name__ == '__main__':
    main()
