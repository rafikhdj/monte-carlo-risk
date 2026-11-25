"""Summarize held-out results and generate publication-exportable figures."""
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

from methods import METHODS

LABELS = dict(normal='Normale', student='Student (5 ddl)', mixture='Mélange gaussien',
              options_1d='Options 1 facteur', options_4d='Options 4 facteurs corrélés')
EXACT_BASELINES = ('MC', 'Stratifié', 'IS', 'IS + strat.')


def load(output):
    meta = json.loads((output/'protocol.json').read_text())
    groups = defaultdict(list)
    with (output/'runs.csv').open(encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle):
            key = row['model'], row['mode'], int(row['anchor']), row['method']
            groups[key].append({k: float(v) for k, v in row.items()
                                if k not in ('model', 'mode', 'anchor', 'method') and v})
    for group in groups.values():
        group.sort(key=lambda row: row['replicate'])
    return meta, groups


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def metrics(meta, groups):
    summaries = []
    comparisons = []
    rng = np.random.default_rng(731024)
    indices = rng.integers(meta['repeats'], size=(3000, meta['repeats']))
    for (model, mode, anchor, name), rows in sorted(groups.items()):
        reference = meta['models'][model]['reference']
        values = np.array([[r['var'], r['es']] for r in rows])
        error = values - reference['value']
        mse = np.mean(error**2, axis=0)
        ms = np.array([r['ms'] for r in rows])
        common = dict(model=model, mode=mode, anchor=anchor, method=name)
        summary = dict(common, ms=np.median(ms), ms_p10=np.quantile(ms,.1), ms_p90=np.quantile(ms,.9),
                       exact=int(rows[0]['exact']), samples=int(rows[0]['n']), peak_mib=rows[0]['peak_bytes']/2**20)
        for field in ('weight_mean', 'ess_fraction', 'fit_rmse', 'fit_max', 'extrapolation',
                      'correction_var', 'correction_es'):
            if field in rows[0]:
                summary[field] = np.mean([r[field] for r in rows])
        for i, metric in enumerate(('var', 'es')):
            errors = error[:, i]
            summary.update({f'bias_{metric}': np.mean(errors),
                            f'bias_se_{metric}': np.std(errors, ddof=1)/np.sqrt(len(rows)),
                            f'variance_{metric}': np.var(values[:, i], ddof=1),
                            f'rmse_{metric}': np.sqrt(mse[i]),
                            f'p90_error_{metric}': np.quantile(np.abs(errors), .9)})
        summaries.append(summary)
        for base in EXACT_BASELINES:
            cmp = groups.get((model, mode, anchor, base))
            if name == base or cmp is None:
                continue
            elapsed_ratio = float(np.median(ms) / np.median([r['ms'] for r in cmp]))
            comp_values = np.array([[r['var'], r['es']] for r in cmp])
            abs_squared = (comp_values - reference['value'])**2
            ratio = np.sqrt(mse / np.mean(abs_squared, axis=0))
            highs, lows = [], []
            ref_margin = np.maximum(reference['ci95'], reference.get('half_delta', [0., 0.]))
            # Bootstrap paired by seed. Also repeat at the reference CI endpoints.
            for endpoint in (-1, 0, 1):
                center = np.array(reference['value']) + endpoint * ref_margin
                a = (values - center)**2
                b = (comp_values - center)**2
                draws = np.sqrt(a[indices].mean(axis=1)/b[indices].mean(axis=1))
                lo, hi = np.quantile(draws, [.025,.975], axis=0)
                highs.append(hi)
                lows.append(lo)
            comparisons.append(dict(common, baseline=base, time_ratio=elapsed_ratio,
                                    matched=(mode!='temps' or .8 <= elapsed_ratio <= 1.2),
                                    ratio_var=ratio[0], ratio_es=ratio[1],
                                    lo_var=np.min(lows,axis=0)[0], hi_var=np.max(highs,axis=0)[0],
                                    lo_es=np.min(lows,axis=0)[1], hi_es=np.max(highs,axis=0)[1]))
    return summaries, comparisons


def plot(meta, rows, output, mode):
    fig, axes = plt.subplots(5, 2, figsize=(12, 15), layout='constrained')
    names = list(METHODS)
    palette = plt.colormaps['tab20'].colors
    criterion = 'exact' if mode == 'exact' else 'ms'
    axis_label = 'Valorisations exactes (scénarios)' if mode == 'exact' else 'Temps médian observé (ms)'
    for model, pair in zip(LABELS, axes):
        for metric, ax in zip(('var','es'), pair):
            ax.set_title(f'{LABELS[model]} · {metric.upper()} 99 %')
            for i, method in enumerate(names):
                data = sorted((r for r in rows if r['model']==model and r['mode']==mode and r['method']==method),
                              key=lambda r:r['anchor'])
                if data:
                    ax.loglog([r[criterion] for r in data], [r['rmse_'+metric] for r in data],
                              label=method, color=palette[i], marker='o' if i<5 else 's', ms=3,
                              lw=1.5 if i<5 else 1, ls='-' if i<5 else '--')
            if mode == 'exact':
                ax.set_xticks(meta['budgets'], [f'{n:,}'.replace(',', ' ') for n in meta['budgets']])
                ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_xlabel(axis_label)
            ax.set_ylabel('RMSE (unités de perte)')
            ax.grid(which='major', alpha=.25)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside upper center', ncols=4, frameon=False, fontsize=9)
    fig.supxlabel(f'Benchmark synthétique · {meta["repeats"]} graines d’évaluation par point', fontsize=10)
    fig.savefig(output/f'erreur_{mode}.png', dpi=155)
    plt.close(fig)


def strong_candidates(comparisons):
    grouped = defaultdict(list)
    for row in comparisons:
        if row['mode']=='temps' and row['method'].startswith('Spline'):
            grouped[row['model'],row['method']].append(row)
    candidates = []
    for (model, method), rows in grouped.items():
        # Faster by >20% is eligible if precision also improves. Slower by >20% is excluded.
        if len(rows)==8 and all(r['time_ratio']<=1.2 and r['ratio_var']<=.8 and r['ratio_es']<=.8
                                and r['hi_var']<1 and r['hi_es']<1 for r in rows):
            candidates.append({'model':model, 'method':method})
    return candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results')
    args = parser.parse_args()
    output = Path(args.output)
    meta, groups = load(output)
    summaries, comparisons = metrics(meta, groups)
    write_csv(output/'summary.csv', summaries)
    write_csv(output/'comparisons.csv', comparisons)
    (output/'decision.json').write_text(json.dumps(strong_candidates(comparisons), ensure_ascii=False, indent=2)+'\n')
    for mode in ('exact', 'temps'):
        plot(meta, summaries, output, mode)


if __name__ == '__main__':
    main()
