from dataclasses import dataclass


@dataclass(frozen = True)
class ModelConfig:
    N       : int   = 32
    J       : float = 1.0
    min_dist: float = 1 / 40
    alpha   : int   = 3

@dataclass(frozen = True)
class Thresholds:
    IR_cutoff   : float = 1e-9
    anisotropy  : float = 1e-6

    max_lenH  : int | None = None
    max_spread: int | None = None
    max_spread_stat: int   = 4

@dataclass
class JobConfig:
    n_runs : int = 10
    n_cpus : int = 2
    out_file  : str = "res/test"
    rand_seed : int | None = 1
