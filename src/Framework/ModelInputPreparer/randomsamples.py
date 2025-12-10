import random
from typing import List

def getRandomEntries(
    forbidden: List[int] | None = None,
    total: int = 1400,
    k: int = 50,
    seed: int | None = None
) -> List[int]:
    """Return k unique random indices in range(total), excluding forbidden ones. If k > total, return a permutation of 0..total-1."""
    if seed is not None:
        random.seed(seed)

    allowed = [i for i in range(total) if i not in forbidden]

    if k > len(allowed):
        raise ValueError("Not enough valid numbers to sample from.")

    return random.sample(allowed, k)

forbidden_list = [
    137, 399, 1041, 1173, 910, 949, 866, 967, 498, 170,
    1110, 1016, 1212, 378, 755, 42, 344, 1020, 1019, 689,
    956, 1127, 520, 948, 306, 89, 1278, 773, 1387, 196,
    337, 476, 1292, 451, 462, 625, 1295, 1064, 1391, 1090,
    744, 113, 1290, 429, 29, 761, 780, 616, 1213, 349
]

# module-level variable kept for backward compatibility
randomEntries = getRandomEntries(forbidden_list)

if __name__ == '__main__':
    print(randomEntries)
    print(len(randomEntries))
