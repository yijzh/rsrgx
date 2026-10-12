import numpy as np
from binary import to_bin
from config import ModelConfig
from op import Operator, Spin, Term


def get_realization(model: ModelConfig, rng : np.random.Generator) -> tuple[np.ndarray, Operator, dict]:

    # Generate the tracker of (supsr)spin configurations
    spins = {}
    for i in range(model.N):
        spins[i] = Spin(1, to_bin({i}), 0)

    # Generate random positions
    L   = model.N
    Le  = model.N - model.N * model.min_dist
    dLs = rng.random(model.N) * Le

    pos = np.sort(np.mod(np.arange(0, Le, Le/model.N) + dLs, Le))
    pos += (.5 + np.arange(0, model.N, 1.)) * model.min_dist
    pos -= pos[0]

    # Generate the Hamiltonian
    H = Operator([])
    for i in range(model.N):
        for j in range(i+1, model.N):
            dist = min(np.abs(pos[i] - pos[j]), L - np.abs(pos[i] - pos[j]))
            n_z  = 0.
            J_ij = model.J / dist**model.alpha * (1 - 3*n_z**2)
            H   &= J_ij * Operator([Term(to_bin({i, j}), 0), Term(0, to_bin({i, j}))])
            # XX + YY interaction

    return (pos, H, spins)

def get_rand_branch(rng: np.random.Generator, max_step: int):
    flips = rng.random(max_step)
    ans = []
    for i in range(max_step):
        flip = flips[i]
        if flip < .5:
            ans.append(2)
        elif flip < .75:
            ans.append(1)
        else:
            ans.append(3)
    return ans
