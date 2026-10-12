def to_bin(sites: set[int] | frozenset[int]) -> int:
    return sum(1 << i for i in sites)

def to_set(x: int) -> set[int]:
    return {i for i in range(x.bit_length()) if x>>i & 1}

def bin_parity(x):
    return x.bit_count() & 1
