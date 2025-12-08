import random
from typing import List

def get_random_entries(total: int = 1400, k: int = 100, seed: int | None = None) -> List[int]:
    """Return k unique random indices in range(total). If k > total, return a permutation of 0..total-1."""
    if seed is not None:
        random.seed(seed)
    k = min(k, total)
    return random.sample(range(total), k)

# module-level variable kept for backward compatibility
randomEntries = get_random_entries()
