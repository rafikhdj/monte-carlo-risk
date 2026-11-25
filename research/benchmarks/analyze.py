"""Aggregate holdout results, compare errors and plot empirical cost curves."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np

from research.estimators import BASELINES
from .run import ROOT, save_csv


def load(results):
    meta = json.loads((results/'protocol.json').read_text())
    grouped = defaultdict(list)
    with (results/'runs.csv').open(encoding='utf-8',newline='') as handle:
        for row in csv.DictReader(handle):
            key = row['problem'], int(row['budget']), row['method']
            grouped[key].append(row)
    for values in grouped.values():
        values.sort(key=lambda r:int(r['replicate']))
    return meta,grouped


def summarize(meta,grouped):
    lookup = {}
    errors = {}
    for key,rows in grouped.items():
        problem,budget,name = key
        ref = meta['problems'][problem]['reference']['value']
        error = np.array([float(x['estimate'])-ref for x in rows])
        times = [float(x['cpu_ms']) for x in rows]
        audit = [float(x['audit_rmse']) for x in rows if x['audit_rmse']]
        row = dict(problem=problem,dimension=rows[0]['dimension'],budget=budget,method=name,
                   exact=int(rows[0]['exact']),training=int(rows[0]['training']),audit=int(rows[0]['audit']),
                   cpu_ms=np.median(times),wall_ms=np.median([float(x['wall_ms']) for x in rows]),
                   cpu_p90=np.quantile(times,.9),peak_mib=int(rows[0]['peak_bytes'])/2**20,
                   bias=error.mean(),bias_se=error.std(ddof=1)/np.sqrt(len(error)),
                   variance=error.var(ddof=1),rmse=np.sqrt(np.mean(error**2)),
                   p90_abs_error=np.quantile(np.abs(error),.9),audit_rmse=np.mean(audit) if audit else '')
        lookup[key] = row
        errors[key] = error
    return lookup, errors


def compare(meta, lookup, errors):
    rng = np.random.default_rng(992105)
    n = meta['repeats']
    resamples = rng.integers(n,size=(3000,n))
    rows = []
    for (problem,budget,method),row in lookup.items():
        if method in BASELINES:
            continue
        ref = meta['problems'][problem]['reference']
        margin = max(ref['uncertainty'],1e-13)
        for scope in ('equal','next','cpu'):
            if scope=='next':
                sizes = [b for b in meta['budgets'] if b>budget]
                if not sizes:
                    continue
                eligible = [size for size in meta['budgets'] if size <= min(sizes)]
            elif scope=='equal':
                eligible = [budget]
            else:
                eligible = meta['budgets']
            contenders = [(key,base) for key,base in lookup.items()
                          if key[0]==problem and key[1] in eligible and key[2] in BASELINES
                          and (scope!='cpu' or base['cpu_ms'] <= 1.2*row['cpu_ms'])]
            if not contenders:
                continue
            key,base = min(contenders,key=lambda pair:pair[1]['rmse'])
            lows, highs = [], []
            for delta in (-margin,0.,margin):
                a = np.mean((errors[problem,budget,method][resamples]-delta)**2,axis=1)
                # Use the minimum baseline RMSE within each resample to reflect competitor selection.
                b = np.min([np.mean((errors[k][resamples]-delta)**2,axis=1) for k,_ in contenders],axis=0)
                ratios = np.sqrt(a/np.maximum(b,1e-30))
                lo,hi = np.quantile(ratios,[.025,.975])
                lows.append(lo)
                highs.append(hi)
            rows.append(dict(problem=problem,budget=budget,method=method,scope=scope,
                             baseline=key[2],baseline_exact=base['exact'],candidate_exact=row['exact'],
                             baseline_cpu_ms=base['cpu_ms'],candidate_cpu_ms=row['cpu_ms'],
                             ratio=row['rmse']/max(base['rmse'],1e-15),
                             low=min(lows),high=max(highs)))
    return rows


def plot(meta, lookup, output, axis):
    methods = list(dict.fromkeys(r['method'] for r in lookup.values()))
    fig, panels = plt.subplots(4,3,figsize=(14,14),layout='constrained')
    main = {'MC','Sobol','IS','Quadrature','Gauss-Hermite','Gauss-Kronrod adaptatif','Contrôle polynomial','Surrogate seul',
            'Surrogate + Sobol','Surrogate + IS','Surrogate + strat.','Surrogate + allocation'}
    palette = plt.colormaps['tab20'].colors
    handles = {}
    for problem,ax in zip(meta['problems'],panels.flat):
        for i,name in enumerate(methods):
            if name not in main:
                continue
            data = sorted((r for key,r in lookup.items() if key[0]==problem and key[2]==name),
                          key=lambda r:r['budget'])
            if data:
                line, = ax.loglog([r[axis] for r in data],[max(r['rmse'],1e-15) for r in data],
                                 ls='-' if name in BASELINES else '--', marker='o', ms=3, color=palette[i%len(palette)],
                                 label=name)
                handles[name] = line
        if axis=='exact':
            ax.set_xticks(meta['budgets'], [f'{n:,}'.replace(',', ' ') for n in meta['budgets']])
            ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(problem.replace('_',' '))
        ax.set_xlabel('Évaluations exactes de f' if axis=='exact' else 'Temps CPU médian (ms)')
        ax.set_ylabel('RMSE de l’intégrale')
        ax.grid(alpha=.2)
    fig.legend(handles.values(),handles.keys(),loc='outside upper center', ncols=5,frameon=False)
    fig.supxlabel(f'{meta["repeats"]} graines indépendantes par configuration · plan expérimental synthétique · plancher graphique 1e−15')
    fig.savefig(output/f'rmse_{axis}.png',dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results')
    args = parser.parse_args()
    meta,groups = load(args.output)
    lookup,errors = summarize(meta,groups)
    save_csv(args.output/'summary.csv',list(lookup.values()))
    comparisons = compare(meta,lookup,errors)
    save_csv(args.output/'comparisons.csv',comparisons)
    for axis in ('exact','cpu_ms'):
        plot(meta,lookup,args.output,axis)


if __name__=='__main__':
    main()
