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
    consistentPairs = accurates = accuratePairs = consistentlyAccuratePairsPairs = ambiguousPairs = consistentlyambiguousPairs = yeses = nos = 0 # Mivel párokra van értve, Mindegyik végeredménye maximum `length` lehet.

    for i in range(length):

        key = keys[i]  # pl. 'sound'

        token1 = googleStraightAnswers.get(key, None)
        if token1 is None:
            ambiguousPairs += 1
            token1 = '?'
        token2 = googleReversedAnswers.get(key, None)
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
            accurates += 1
        if reversedCorrect:
            accurates += 1

        if straightCorrect or reversedCorrect:
            accuratePairs += 1

        if token1 == 'Yes' and token2 == 'Yes' and groundTruth:
            consistentlyAccuratePairsPairs += 1
        if token1 == 'No' and token2 == 'No' and not groundTruth:
            consistentlyAccuratePairsPairs += 1

    return consistentPairs / length, accurates / length, accuratePairs / length, consistentlyAccuratePairsPairs / length, ambiguousPairs / length, consistentlyambiguousPairs / length, yeses, nos # # 0 és 1 közé normalizáljuk, kivéve a yeseket és a nokat

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
        'have2': 'Yes',
        'inspire': 'Yes',
        'afterthought': 'Yes',
        'property': 'No', 'awareness': 'Yes',
        'prefer': 'Yes', 'bend': 'Yes', 'mark': 'No',
        'have1': 'Yes', 'rounding': 'No',
        'steamroller': 'Yes', 'zero': 'Yes', 'nest': 'Yes',
        'land': 'No', 'deliberation': 'Yes',
        'consist': 'No', 'restraint ': 'No', 'feedstock': 'No',
        'engage': 'Yes', 'sneak': 'No',
        'justify': 'No', 'grain': 'No', 'pass': 'No',
        'topic': 'No', 'holder': 'Yes',
        'crystallize': 'Yes', 'recapitulate': 'Yes', 'rag': 'No',
        'complaint': 'No', 'fiddle': 'Yes',
        'wax': 'No', 'tease': 'No', 'access': 'Yes',
        'union': 'No', 'cross': 'No',
        'morale': 'Yes', 'back': 'Yes', 'bother': 'Yes',
        'organize': 'No', 'dash': 'No',
        'loop': 'Yes', 'resolve': 'No', 'underlay': 'Yes',
        'submit': 'Yes', 'blood': 'No',
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
        'deflate': 'Yes',
        'local': 'Yes',
        'drive': 'No',
        'have2': 'No',
        'inspire': 'Yes',
        'afterthought': 'No',
        'property': 'No',
        'awareness': 'No',
        'prefer': 'Yes',
        'bend': 'Yes',
        'mark': 'No',
        'have1': 'Yes',
        'rounding': 'No',
        'steamroller': 'No', 'zero': 'No', 'nest': 'Yes',
        'land': 'Yes', 'deliberation': 'Yes',
        'consist': 'No', 'restraint ': 'No', 'feedstock': 'No',
        'engage': 'Yes', 'sneak': 'No',
        'justify': 'No', 'grain': 'Yes', 'pass': 'No',
        'topic': 'No', 'holder': 'Yes',
        'crystallize': 'No', 'recapitulate': 'No', 'rag': 'No',
        'complaint': 'No', 'fiddle': 'No',
        'wax': 'No', 'tease': 'No', 'access': 'Yes',
        'union': 'No', 'cross': 'No',
        'morale': 'Yes', 'back': 'No', 'bother': 'Yes',
        'organize': 'No', 'dash': 'No',
        'loop': 'No', 'resolve': 'Yes', 'underlay': 'No',
        'submit': 'Yes', 'blood': 'No',
        'violence': 'Yes', 'lot': 'No',
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

    consistentPairs, accurates, accuratePairs, consistentlyAccuratePairsPairs, ambiguousPairs, consistentlyambiguousPairs, yeses, nos = indexer(googleStraightAnswers,
                                                                                                  googleReversedAnswers,
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
        printEverywhere(f'{USERNAME} ran google/gemma-2-2b-it at {formattedDate}', f)
        printEverywhere('Answers ratios (True/all)', f)
        printEverywhere(f'Consistent pairs: {consistentPairs * 100} entries.', f)
        printEverywhere(f'Accurates: {accurates * 100} entries.', f)
        printEverywhere(f'Accurate Pairs {accuratePairs * 100} entries.', f)
        printEverywhere(f'Consistently accurate pairs: {consistentlyAccuratePairsPairs * 100} entries.', f)
        printEverywhere(f'Ambiguous pairs: {ambiguousPairs * 100} entries.', f)
        printEverywhere(f'Consistently ambiguous pairs: {consistentlyambiguousPairs * 100} entries.', f)
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
