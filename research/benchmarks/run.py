"""Reproducible equal-call budget campaign, with held-out calibration."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'

import argparse
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path
import platform
import random
from time import perf_counter_ns, process_time_ns
import tracemalloc

import numpy as np
import scipy

from research.estimators import BASELINES, SURROGATES, run
from research.problems import PROBLEMS, reference

BUDGETS = (128, 512, 2048)
PILOT_SEEDS = range(211000, 211012)
TEST_START = 710000
ROOT = Path(__file__).resolve().parents[1]


def methods(problem):
    all_methods = BASELINES | SURROGATES
    if problem.dimension == 1:
        del all_methods['Surrogate + MC']  # Identical to the uniform cubic spline in 1D.
    if problem.dimension > 1:
        del all_methods['Gauss-Kronrod adaptatif']
        for name in ('Spline uniforme + MC','Spline quantiles + MC','PCHIP + MC','Chebyshev + MC','Adaptatif + MC'):
            del all_methods[name]
    if problem.dimension > 2:
        del all_methods['Quadrature']
        del all_methods['Gauss-Hermite']
    return all_methods


def measurement(problem, method, budget, seed):
    wall = perf_counter_ns()
    cpu = process_time_ns()
    value,calls,training,audit,fit = run(problem,method,budget,seed)
    return dict(estimate=value, exact=calls, training=training, audit=audit, audit_rmse=fit,
                cpu_ms=(process_time_ns()-cpu)/1e6, wall_ms=(perf_counter_ns()-wall)/1e6)


def calibrate(problem, budget, candidates):
    best = {}
    for name,method in candidates.items():
        # Fixed Sobol residual power. Probe allocation only for corrected surrogates.
        if method.sampler in ('none','sobol','quad','hermite','adaptive_quad') or method.surrogate in ('','control'):
            continue
        scores = {}
        for fraction in (.125,.25,.5):
            cfg = replace(method,fraction=fraction)
            values = [run(problem,cfg,budget,seed)[0] for seed in PILOT_SEEDS]
            scores[str(fraction)] = float(np.var(values,ddof=1))
        chosen = float(min(scores,key=scores.get))
        best[name] = dict(selected=chosen, pilot_variance=scores, exact=3*len(PILOT_SEEDS)*budget)
        candidates[name] = replace(method,fraction=chosen)
    return best


def save_csv(path, data):
    with path.open('w',encoding='utf-8',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(data[0]),lineterminator="\n")
        writer.writeheader()
        writer.writerows(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results')
    parser.add_argument('--repeats',type=int,default=32)
    parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args()
    repeats = 4 if args.smoke else args.repeats
    budgets = BUDGETS[:1] if args.smoke else BUDGETS
    args.output.mkdir(parents=True,exist_ok=True)
    metadata = dict(budgets=budgets,repeats=repeats,pilot_seeds=list(PILOT_SEEDS),
                    test_start=TEST_START,python=platform.python_version(),os=platform.platform(),
                    numpy=np.__version__,scipy=scipy.__version__,problems={})
    data = []
    # Low-dimensional functions run first. Later cases use the same locked protocol.
    for problem in PROBLEMS.values():
        print(problem.name,flush=True)
        details = dict(reference=reference(problem),dimension=problem.dimension,config={},pilot={})
        metadata['problems'][problem.name] = details
        for budget in budgets:
            candidates = methods(problem)
            pilot_start = process_time_ns()
            details['pilot'][budget] = calibrate(problem,budget,candidates)
            details['pilot'][budget]['cpu_ms'] = (process_time_ns()-pilot_start)/1e6
            details['config'][budget] = {key:asdict(cfg) for key,cfg in candidates.items()}
            peaks = {}
            for name,method in candidates.items():
                tracemalloc.start()
                try:
                    run(problem,method,budget,411000)
                    peaks[name] = tracemalloc.get_traced_memory()[1]
                finally:
                    tracemalloc.stop()
            for rep in range(repeats):
                order = list(candidates)
                random.Random(TEST_START+rep).shuffle(order)
                for name in order:
                    m = measurement(problem,candidates[name],budget,TEST_START+rep)
                    if not np.isfinite(m['estimate']) or m['exact']>budget:
                        raise RuntimeError((problem.name,name,budget,m))
                    data.append(dict(problem=problem.name,dimension=problem.dimension,budget=budget,
                                     method=name,replicate=rep,peak_bytes=peaks[name],**m))
    save_csv(args.output/'runs.csv',data)
    (args.output/'protocol.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(f'{len(data)} runs',flush=True)


if __name__ == '__main__':
    main()
