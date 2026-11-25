"""Summaries, paired bootstrap and error/cost figures."""
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
from scipy.stats import false_discovery_control, t

from experiment import COMPONENT_METHODS, REFERENCE_METHODS, ROOT, save


def read(output):
    meta = json.loads((output/'protocol.json').read_text())
    grouped = defaultdict(list)
    with (output/'runs.csv').open(encoding='utf-8',newline='') as stream:
        for row in csv.DictReader(stream):
            key = row['problem'],row['mode'],int(row['anchor']),row['method']
            grouped[key].append(row)
    return meta,grouped


def stats(meta, groups):
    summary = {}
    errors = {}
    for key,group in sorted(groups.items()):
        name,mode,anchor,method = key
        group.sort(key=lambda r:int(r['replicate']))
        ref = meta['models'][name]['reference']
        x = np.array([float(r['estimate'])-ref for r in group])
        ms = np.array([float(r['cpu_ms']) for r in group])
        bias = np.mean(x)
        variance = np.var(x,ddof=1)
        se = np.sqrt(variance/len(x))
        critical = t.ppf(.975,len(x)-1)
        null = 1. if abs(bias)<1e-14 and se<1e-14 else (
            2*t.sf(abs(bias)/se,len(x)-1) if se>0 else 0.)
        summary[key] = dict(problem=name,d=int(group[0]['d']),mode=mode,anchor=anchor,method=method,
                            exact=int(group[0]['exact']),cpu_ms=np.median(ms),cpu_p90=np.quantile(ms,.9),
                            bias=bias,bias_lo=bias-critical*se,bias_hi=bias+critical*se,bias_p=null,
                            variance=variance,rmse=np.sqrt(np.mean(x*x)),p90=np.quantile(np.abs(x),.9))
        errors[key] = x
    for row,q in zip(summary.values(),false_discovery_control([r['bias_p'] for r in summary.values()])):
        row['bias_fdr_p']=q
    return summary,errors


def compare(meta, summary, errors):
    n = meta['repeats']
    ix = np.random.default_rng(940000).integers(n,size=(1500,n))
    # Precompute once per estimator. Pairing is by independently repeated seed.
    draws = {key:(np.mean(x[ix]**2,axis=1),np.var(x[ix],axis=1,ddof=1)) for key,x in errors.items()}
    rows = []
    for key,a in summary.items():
        problem,mode,anchor,method = key
        if method not in COMPONENT_METHODS:
            continue
        for ref in REFERENCE_METHODS:
            ref_key = problem,mode,anchor,ref
            b = summary.get(ref_key)
            if b is None:
                continue
            cost = a['cpu_ms']/b['cpu_ms']
            row = dict(problem=problem,mode=mode,anchor=anchor,method=method,baseline=ref,
                       candidate_exact=a['exact'],baseline_exact=b['exact'],cpu_ratio=cost,
                       cpu_matched=(.8<=cost<=1.25))
            for i,(label,power) in enumerate((('rmse',.5),('variance',1.))):
                divisor = b[label]
                floor = 1e-14 if label=='rmse' else 1e-28
                if a[label]<=floor and divisor<=floor:
                    ratio,lo,hi = 1.,1.,1.
                elif divisor<=floor:
                    # No finite, interpretable ratio against a numerical zero.
                    ratio,lo,hi = '', '', ''
                else:
                    ratio = a[label]/divisor
                    boot = (draws[key][i]/np.maximum(draws[ref_key][i],1e-28))**power
                    lo,hi = np.quantile(boot,[.025,.975])
                row[f'{label}_ratio']=ratio
                row[f'{label}_lo']=lo
                row[f'{label}_hi']=hi
            rows.append(row)
    return rows


def plot(output, meta, summary, axis):
    selected = ('MC','Direct','Projection','LHS','Antithétique','Sobol','Sobol projeté',
                'Projection Neyman','LHS k2','LHS k16','Grille k2','Grille k2 Neyman','LHS k2 antith.')
    palette = plt.colormaps['tab20'].colors
    fig,panels = plt.subplots(5,4,figsize=(16,15),layout='constrained')
    handles = {}
    for problem,ax in zip(meta['models'],panels.flat):
        for i,method in enumerate(selected):
            rows = sorted((r for k,r in summary.items() if k[0]==problem and k[1]=='exact' and k[3]==method),
                          key=lambda r:r['anchor'])
            if not rows:
                continue
            line, = ax.loglog([r[axis] for r in rows],[max(r['rmse'],1e-14) for r in rows],
                             marker='o',ms=3,linestyle='--' if method in COMPONENT_METHODS else '-',
                             color=palette[i],label=method)
            handles[method] = line
        if axis=='exact':
            ax.set_xticks(meta['budgets'],[f'{n:,}'.replace(',',' ') for n in meta['budgets']])
            ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(problem)
        ax.set_xlabel('Appels exacts de f' if axis=='exact' else 'Temps CPU médian (ms)')
        ax.set_ylabel('RMSE de l’espérance')
        ax.grid(alpha=.2)
    for panel in panels.flat[len(meta['models']):]:
        panel.axis('off')
    fig.legend(handles.values(),handles.keys(),ncols=4,loc='outside upper center',frameon=False)
    fig.supxlabel(f'{meta["repeats"]} réplications par point · chronométrage de bout en bout · plancher graphique 10⁻¹⁴')
    fig.savefig(output/f'rmse_{axis}.png',dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'results')
    args = p.parse_args()
    meta,groups = read(args.output)
    summary,errors = stats(meta,groups)
    save(args.output/'summary.csv',list(summary.values()))
    save(args.output/'comparisons.csv',compare(meta,summary,errors))
    for axis in ('exact','cpu_ms'):
        plot(args.output,meta,summary,axis)


if __name__=='__main__':
    main()
