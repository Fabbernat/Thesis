def indexer(straightAnswers, reversedAnswers, guids, goldStandard: list[bool]):
    ''''''
    all = len(straightAnswers)
    consistents = accurates = consistentlyAccurates = ambiguouses = consistentlyAmbiguouses = 0

    for i in range(all):


        token1 = straightAnswers[i]
        token2 = reversedAnswers[i]
        groundTruth = goldStandard[guids[i]]


        if token1 == '?':
            ambiguouses += 1
        if token2 == '?':
            ambiguouses += 1

        if token1 == 'Yes' and token2 == 'Yes':
            consistents += 1
        if token1 == 'No' and token2 == 'No':
            consistents += 1
        if token1 != '?' and token2 != '?':
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


    return consistents / all, accurates / all, consistentlyAccurates / all, ambiguouses / all, consistentlyAmbiguouses / all # 0 és 1 közé normalizáljuk

def test_indexer():
    straightAnswers: list[str] = ['No', 'Yes', 'Yes', 'Yes', 'No']
    reversedAnswers: list[str] = ['Yes', 'No', 'No', 'Yes', 'No']
    guids: list[int] = [18, 937, 242, 903, 91]
    goldStandard: list[bool] = []
    lines:list[str] = []

    from pathlib import Path
    basePath = Path(__file__).resolve()

    with open(basePath.parents[1] / 'data' / 'test.gold.in', 'r', encoding='utf-8') as goldFile:
        lines = goldFile.readlines()

    goldStandard = [line.strip() == 'T' for line in lines]

    
    consistents, accurates, consistentlyAccurates, ambiguouses, consistentlyAmbiguouses = indexer(straightAnswers, reversedAnswers, guids, goldStandard)
    
    
    print('Answers ratios (True/all)')
    print(f'Consistent: {consistents:.2f} %')
    print(f'Accurate: {accurates:.2f} %')
    print(f'Consistently accurate: {consistentlyAccurates:.2f} %')
    print(f'Ambiguous: {ambiguouses:.2f} %')
    print(f'Consistently ambiguous:{consistentlyAmbiguouses} %')
    
if __name__ == '__main__':
    test_indexer()
