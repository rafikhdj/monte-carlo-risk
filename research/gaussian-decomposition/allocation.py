"""Coarse-strata controls isolate allocation from partition resolution."""
from collections import defaultdict
from time import process_time_ns

import numpy as np

from experiment import ROOT, estimate, grid_points, problems, save

REPEATS = 100
SEED = 970000


def uniform_coarse(problem, n, seed, shape):
    rng = np.random.default_rng(seed)
    counts = np.full(16,n//16)
    values = problem.fun(grid_points(rng,problem,counts,shape))
    return float(values.mean())


def main():
    rows = []
    groups = defaultdict(list)
    for problem in problems():
        for budget in (128,512,2048,8192):
            for shape,adapt in (((4,4),'Grille k2 Neyman'), ((16,),'Projection Neyman')):
                for i in range(REPEATS):
                    for allocation in ('neyman','uniforme','uniforme sans les 48 appels'):
                        exact = budget-48 if allocation.endswith('appels') else budget
                        start = process_time_ns()
                        if allocation=='neyman':
                            value,calls = estimate(problem,budget,SEED+i,adapt)
                            assert calls==exact
                        else:
                            value = uniform_coarse(problem,exact,SEED+i,shape)
                        cpu = (process_time_ns()-start)/1e6
                        error = value-problem.mean
                        row = dict(problem=problem.name,budget=budget,partition=adapt,
                                   allocation=allocation,replicate=i,exact=exact,error=error,cpu_ms=cpu)
                        rows.append(row)
                        groups[problem.name,budget,adapt,allocation].append(row)
    output = ROOT/'results'
    save(output/'allocation_runs.csv',rows)
    summary = []
    for (problem,budget,partition,allocation),items in groups.items():
        x = np.array([r['error'] for r in items])
        summary.append(dict(problem=problem,budget=budget,partition=partition,allocation=allocation,
                            exact=items[0]['exact'],cpu_ms=np.median([r['cpu_ms'] for r in items]),
                            variance=x.var(ddof=1),rmse=np.sqrt(np.mean(x*x))))
    save(output/'allocation_summary.csv',summary)


if __name__=='__main__':
    main()
