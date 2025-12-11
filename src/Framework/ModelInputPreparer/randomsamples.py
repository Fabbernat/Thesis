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

    forbidden = forbidden or []

    allowed = [i for i in range(total) if i not in forbidden]

    if k > len(allowed):
        raise ValueError("Not enough valid numbers to sample from.")

    return random.sample(allowed, k)


# module-level variable kept for backward compatibility
randomEntries = getRandomEntries()

if __name__ == '__main__':
    print(randomEntries)
    print(len(randomEntries))
