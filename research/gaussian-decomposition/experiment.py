"""Compare direct and lifted normal stratification at measured exact cost."""
import os

for var in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[var] = '1'

import argparse
import csv
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import platform
import random
from time import process_time_ns
from typing import Callable

import numpy as np
import scipy
from scipy.special import ndtr, ndtri
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parent
BUDGETS = (128, 512, 2048, 8192)
REFERENCE_METHODS = ('MC', 'Direct', 'Projection', 'LHS', 'LHS projeté',
                     'Antithétique', 'Sobol', 'Sobol projeté', 'Projection Neyman')
COMPONENT_METHODS = ('Scission iid', 'LHS k2', 'LHS k4', 'LHS k8', 'LHS k16',
                     'LHS k2 antith.', 'LHS k2 remise', 'Grille k2', 'Grille k2 Neyman')


@dataclass(frozen=True)
class Problem:
    name: str
    d: int
    fun: Callable
    mean: float
    direction: str = 'sum'


def bump(x, center, width):
    return np.exp(-.5*((x-center)/width)**2)


def bump_mean(center, width):
    return width/np.sqrt(1+width**2)*np.exp(-.5*center**2/(1+width**2))


def problems():
    k, a = .75, 2.65
    yield from (
        Problem('linéaire', 1, lambda x: x[:,0], 0.),
        Problem('carré', 1, lambda x: x[:,0]**2, 1.),
        Problem('exponentielle', 1, lambda x: np.exp(x[:,0]), np.exp(.5)),
        Problem('sinus', 1, lambda x: np.sin(5*x[:,0]), 0.),
        Problem('call', 1, lambda x: np.maximum(x[:,0]-k,0.),
                np.exp(-k*k/2)/np.sqrt(2*np.pi)-k*ndtr(-k)),
        Problem('événement rare', 1, lambda x: (x[:,0]>a).astype(float), ndtr(-a)),
        Problem('pics multiples', 1, lambda x: bump(x[:,0],-1.3,.19)+.8*bump(x[:,0],1.65,.11),
                bump_mean(-1.3,.19)+.8*bump_mean(1.65,.11)),
    )
    for d in (2,5,10,20):
        yield Problem(f'additif {d}D', d, lambda x: np.mean(x*x,axis=1), 1.)
        yield Problem(f'projection {d}D', d,
                      lambda x,d=d: np.exp(np.sum(x,axis=1)/np.sqrt(d)), np.exp(.5))
        yield Problem(f'interactions {d}D', d, lambda x: np.mean(
            np.cos(1.2*(x+np.roll(x,-1,axis=1)))+.25*x*np.roll(x,-1,axis=1), axis=1),
            np.exp(-1.2**2), 'pair')


def normal_quantile(u):
    return ndtri(np.clip(u, 2.**-54, 1.-2.**-53))


@lru_cache
def direction(d, kind):
    unit = np.ones(d) if kind=='sum' else np.r_[1.,1.,np.zeros(d-2)]
    unit /= np.linalg.norm(unit)
    # Householder maps e1 to unit, preserving the Gaussian law.
    axis = -unit.copy()
    axis[0] += 1
    return axis / np.linalg.norm(axis) if d>1 else axis


def orient(z, problem):
    if problem.d==1:
        return z
    axis = direction(problem.d,problem.direction)
    return z - 2*(z@axis)[:,None]*axis


def latin(rng, n, d):
    strata = rng.permuted(np.broadcast_to(np.arange(n)[:,None],(n,d)),axis=0)
    return normal_quantile((strata+rng.random((n,d)))/n)


def choose_counts(scores, n):
    # Give every stratum one final draw, then allocate the remainder.
    m = len(scores)
    mass = scores/scores.sum() if scores.sum()>0 else np.full(m,1/m)
    quota = (n-m)*mass
    counts = np.floor(quota).astype(int)+1
    remainder = n-int(counts.sum())
    counts[np.argsort(-(quota-np.floor(quota)),kind='stable')[:remainder]] += 1
    return counts


def grid_points(rng, problem, counts, cell_shape):
    p = np.array(cell_shape)
    strata = np.indices(cell_shape).reshape(len(p),-1).T
    repeated = np.repeat(strata,counts,axis=0)
    coordinates = normal_quantile((repeated+rng.random(repeated.shape))/p)
    z = np.empty((len(repeated),problem.d))
    z[:,0] = np.sum(coordinates,axis=1)/np.sqrt(len(p))
    z[:,1:] = rng.standard_normal((len(z),problem.d-1))
    return orient(z,problem)


def grid_estimate(problem, n, rng, kind):
    adaptive = kind.endswith('Neyman')
    if adaptive:
        shape = (4,4) if kind.startswith('Grille') else (16,)
        m = 16
        probe = problem.fun(grid_points(rng,problem,np.full(m,3),shape)).reshape(m,3)
        sigma = np.std(probe,axis=1,ddof=1)
        score = .9*sigma + .1*np.mean(sigma)
        counts = choose_counts(score,n-3*m)
    else:
        # m1*m2=n for the four exact budgets, close to square for time matching.
        m1 = 2**(int(np.log2(n))//2)
        m2 = n//m1
        shape = (m1,m2)
        m = m1*m2
        counts = np.full(m,n//m)
        counts[:n % m] += 1
    values = problem.fun(grid_points(rng,problem,counts,shape))
    sums = np.add.reduceat(values,np.r_[0,np.cumsum(counts)[:-1]])
    return float(np.mean(sums/counts)), int(counts.sum())+(48 if adaptive else 0)


def estimate(problem, n, seed, name):
    rng = np.random.default_rng(seed)
    d = problem.d
    if name.startswith('Grille') or name=='Projection Neyman':
        return grid_estimate(problem,n,rng,name)
    if name.startswith('LHS k'):
        k = int(name.split()[1][1:])
        antithetic = name.endswith('antith.')
        fine = n//2 if antithetic else n
        if name.endswith('remise'):
            pool = normal_quantile((np.arange(n)[:,None]+rng.random((n,k*d)))/n)
            indices = rng.integers(n,size=pool.shape)
            components = np.take_along_axis(pool,indices,axis=0)
        else:
            components = latin(rng,fine,d*k)
        x = components.reshape(fine,k,d).sum(axis=1)/np.sqrt(k)
        if antithetic:
            x = np.vstack((x,-x))
        x = orient(x,problem)
    elif name.startswith('LHS'):
        x = latin(rng,n,d)
        if name.endswith('projeté'):
            x = orient(x,problem)
    elif name.startswith('Sobol'):
        engine = qmc.Sobol(d,scramble=True,bits=53,rng=rng)
        x = normal_quantile(engine.random_base2(int(np.log2(n))))
        if name.endswith('projeté'):
            x = orient(x,problem)
    elif name in ('Direct','Projection'):
        x = np.empty((n,d))
        x[:,0] = normal_quantile((np.arange(n)+rng.random(n))/n)
        x[:,1:] = rng.standard_normal((n,d-1))
        if name=='Projection':
            x = orient(x,problem)
    elif name=='Antithétique':
        paired = rng.standard_normal((n//2,d))
        x = np.vstack((paired,-paired))
    else:
        x = rng.standard_normal((n,d))
        if name=='Scission iid':
            x = (x+rng.standard_normal(x.shape))/np.sqrt(2)
    return float(np.mean(problem.fun(x))),len(x)


def method_names(problem):
    if problem.d>1:
        return REFERENCE_METHODS+COMPONENT_METHODS
    # Direct/LHS and their projected versions coincide in 1D.
    return tuple(name for name in REFERENCE_METHODS+COMPONENT_METHODS
                 if name not in ('Projection','LHS','LHS projeté','Sobol projeté'))


def time_estimate(problem, n, seed, name):
    start = process_time_ns()
    value, calls = estimate(problem,n,seed,name)
    return value,calls,(process_time_ns()-start)/1e6


def closest_cpu(problem, names, anchor):
    def time(name,n):
        return float(np.median([time_estimate(problem,n,190000+i,name)[2] for i in range(5)]))

    target = time('Direct' if problem.d==1 else 'Projection',anchor)
    sizes = {}
    for name in names:
        if name.startswith('Sobol'):
            options = {n:time(name,n) for n in (64,128,256,512,1024,2048,4096,8192,16384)}
        else:
            options = {}
            lo,hi = 64,16384
            for _ in range(10):
                n = ((lo+hi)//4)*2
                t = time(name,n)
                options[n] = t
                if t<target:
                    lo = n
                else:
                    hi = n
            options[64] = time(name,64)
        sizes[name] = min(options,key=lambda n:abs(np.log(options[n]/target)))
    return sizes, target


def save(path, rows):
    with path.open('w',encoding='utf-8',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results')
    parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    reps = 4 if args.smoke else 100
    budgets = BUDGETS[:1] if args.smoke else BUDGETS
    meta = dict(repeats=reps,budgets=budgets,exact_seed=310000,cpu_seed=710000,
                python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                platform=platform.platform(),models={})
    rows = []
    for problem in problems():
        print(problem.name,flush=True)
        names = method_names(problem)
        info = dict(d=problem.d,reference=float(problem.mean),direction=problem.direction,cpu={})
        meta['models'][problem.name]=info
        regimes = [('exact',b,{name:b for name in names}) for b in budgets]
        for anchor in (() if args.smoke else (512,2048)):
            sizes,target = closest_cpu(problem,names,anchor)
            info['cpu'][anchor] = dict(target_ms=target,sizes=sizes)
            regimes.append(('cpu',anchor,sizes))
        for i in range(reps):
            jobs = [(mode,anchor,name,sizes[name]) for mode,anchor,sizes in regimes for name in names]
            random.Random(810000+i).shuffle(jobs)
            for mode,anchor,name,n in jobs:
                value,calls,ms = time_estimate(problem,n,meta[f'{mode}_seed']+i,name)
                if calls!=n or not np.isfinite(value):
                    raise RuntimeError((problem.name,mode,anchor,name,n,calls,value))
                rows.append(dict(problem=problem.name,d=problem.d,mode=mode,anchor=anchor,
                                 method=name,replicate=i,estimate=value,exact=calls,cpu_ms=ms))
    save(args.output/'runs.csv',rows)
    (args.output/'protocol.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False)+'\n')
    print(len(rows),'mesures',flush=True)


if __name__=='__main__':
    main()
