import numpy as np
import ray
from config import JobConfig, ModelConfig, Thresholds
from models import get_rand_branch, get_realization
from rg import RSRG, RGResult

model = ModelConfig(N = 32)

job   =   JobConfig(n_runs    = 10,
                    n_cpus    = 1,  # Adjust for parallelization
                    out_file  = "res/test",
                    rand_seed = None)

thres =  Thresholds(IR_cutoff = 1e-9,
                    anisotropy= 1e-6)

rng = np.random.default_rng(job.rand_seed)

job.out_file += ("_j" + hex(rng.integers(0, 65536))[2:])

def array_to_str(arr, fmt = "%.10f"):
    # Get the array ready to write
    return " ".join(fmt % x for x in arr) + "\n"

def get_log_freqs_str(freqs):
    ans = [[np.inf, 0.]]
    for freq in freqs:
        if freq == 0.:
            ans[0][1] += 1./model.N
        else:
            ans.append([-np.log(freq), 1./model.N])
    
    ans = np.transpose(np.array(ans))
    freqs = ans[0]
    terms = .5 * ans[1]
    ind = np.argsort(freqs)[::-1]
    freqs = freqs[ind]
    terms = terms[ind]

    return (freqs, terms)

def result_to_file(res: RGResult, out_prefix: str):

    if res.error is not None:
        print(res.error, flush=True)
        return

    spec_buff = f"{res.energy:.10e}: " + str(list(reversed(res.state))) + "\n"
    with open(out_prefix + "_spec.dat", "a") as outf:
        outf.write(spec_buff)

    if res.warnings:
        with open(out_prefix + "_logs.dat", "a") as outf:
            outf.write("".join(res.warnings))

    with open(out_prefix + "_qual.dat", "a") as outf:
        outf.write(array_to_str(np.array(res.quals), fmt = "%.8e"))

    freqs, amps = get_log_freqs_str(res.pair_freq)
    freq_buff = str(len(freqs)) + "\n" + array_to_str(freqs) + array_to_str(amps, fmt = "%.10e")
    with open(out_prefix + ".dat", "a") as outf:
        outf.write(freq_buff)

    with open(out_prefix + "_spin.dat", "a") as outf:
        outf.write(res.spin_stat + "\n")
    

def run_sample(sample_id: int, model: ModelConfig, thres: Thresholds, rng: np.random.Generator) -> RGResult:
    pos, H, spins = get_realization(model, rng)
    secs          = get_rand_branch(rng, model.N)

    try:
        return RSRG(sample_id, H, spins, pos, secs, thres)
    except Exception as exc:
        return RGResult(sample_id = sample_id, error = repr(exc))


# @ray.remote(max_calls=1000) may help free memory if necessary
@ray.remote
def run_sample_remote(*args):
    return run_sample(*args)


rngs = rng.spawn(job.n_runs)

run_id = 0
if job.n_cpus == 1:
    for sample_id in range(job.n_runs):
        result_to_file(run_sample(sample_id, model, thres, rngs[sample_id]), job.out_file)
else:
    ray.init(num_cpus = job.n_cpus)
    max_pending_tasks = 2 * job.n_cpus
    pendings, next_run_id = [], 0

    while next_run_id < job.n_runs or pendings:
        while len(pendings) < max_pending_tasks and next_run_id < job.n_runs:
            # Add job to queue
            pendings.append(run_sample_remote.remote(next_run_id, model, thres, rngs[next_run_id]))
            next_run_id += 1

        done, pendings = ray.wait(pendings, num_returns = 1)
        for res in ray.get(done):
            result_to_file(res, job.out_file)
