"""Run independent calibration, reference calculations and held-out experiments."""
import os

# Pin threaded numerical kernels before importing NumPy.
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import argparse
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path
import platform
import random
from time import perf_counter_ns
import tracemalloc

import numpy as np
from scipy import integrate, optimize
from scipy import __version__ as scipy_version
from scipy.special import ndtri
from scipy.stats import norm, qmc, t

from methods import Config, METHODS, draw, risk, run, surrogate
from models import MODELS, TAIL

BUDGETS = (512, 2048, 8192)
TIME_ANCHORS = (2048, 8192)
MIN_BUDGET = 128
MAX_BUDGET = 32768
TRIPLE = 'Spline + scindées + strat.'


def reference(model, power):
    analytic = model.analytic()
    if analytic is not None:
        return dict(value=list(analytic), ci95=[0., 0.], kind='analytique')
    if model.dim == 1:
        # The short call/put portfolio is strictly convex in spot, with two tails.
        def loss(x):
            return model.value(np.array([[x]]))[0]

        def solve(limit):
            center = optimize.minimize_scalar(loss, bounds=(-limit, limit), method='bounded').x

            def boundaries(level):
                return (optimize.brentq(lambda x: loss(x)-level, -limit, center, xtol=1e-12),
                        optimize.brentq(lambda x: loss(x)-level, center, limit, xtol=1e-12))

            def survival_gap(level):
                left, right = boundaries(level)
                return norm.cdf(left) + norm.sf(right) - TAIL

            maximum = min(loss(-limit), loss(limit))
            q = optimize.brentq(survival_gap, loss(center)+1e-8, maximum-1e-8, xtol=1e-11)
            left, right = boundaries(q)
            integrand = lambda x: loss(x) * norm.pdf(x)
            excess = integrate.quad(integrand, -limit, left, epsabs=1e-12)[0]
            excess += integrate.quad(integrand, right, limit, epsabs=1e-12)[0]
            return np.array([q, excess/TAIL])

        fine, coarse = solve(14), solve(11)
        return dict(value=fine.tolist(), ci95=[0., 0.],
                    truncation_delta=np.abs(fine-coarse).tolist(),
                    kind='racines de la probabilité de dépassement et quadrature gaussienne')
    reps = []
    halves = []
    for i in range(8):
        design = qmc.Sobol(model.dim, scramble=True, bits=53, seed=700_000 + i).random_base2(power)
        loss = np.concatenate([model.value(ndtri(np.clip(batch, 2.**-54, 1. - 2.**-53))) for batch in np.array_split(design, 16)])
        reps.append(risk(loss))
        halves.append(risk(loss[:len(loss)//2]))
    reps = np.array(reps)
    return dict(value=reps.mean(axis=0).tolist(), kind='Sobol brouillé (valorisations exactes)',
                ci95=(t.ppf(.975, 7) * reps.std(axis=0, ddof=1) / np.sqrt(8)).tolist(),
                half_delta=np.abs(reps.mean(axis=0)-np.mean(halves, axis=0)).tolist(),
                power=power, replicates=reps.tolist())


def timed(model, config, n, seed):
    start = perf_counter_ns()
    result, diagnostics = run(model, config, n, seed)
    return result, diagnostics, (perf_counter_ns() - start) / 1e6


def calibrate(model, smoke):
    z, w = draw(np.random.default_rng(1000), 4096, model, Config('is'))
    exact = model.value(z)
    fit = {}
    for grid in ('uniform', 'quantile', 'tail', 'adaptive'):
        scores = []
        for width in (4., 5., 6.):
            # At fixed node budget, prefer low held-out loss error with extra tail scrutiny.
            cfg = Config(grid=grid, width=width)
            predict, _ = surrogate(model, cfg)
            approximation, _ = predict(z)
            residual = (approximation - exact)**2
            scores.append((float(np.average(residual, weights=w) + np.mean(residual)), width))
        best = min(scores)
        fit[grid] = dict(width=best[1], scores=scores)

    def score(cfg):
        reps = [timed(model, cfg, 2048, 2000+i) for i in range(4 if smoke else 16)]
        estimates = np.array([r[0] for r in reps])
        scale = np.maximum(np.abs(estimates.mean(axis=0)), 1e-12)
        return float(np.sum(estimates.var(axis=0, ddof=1)/scale**2) * np.median([r[2] for r in reps]))

    configs = {}
    candidates = {}
    for name, default in METHODS.items():
        if name == TRIPLE:
            continue
        cfg = replace(default, width=fit[default.grid]['width']) if default.grid else default
        shifts = (.65, 1., 1.35) if cfg.sampler in ('is', 'is_strat') else (cfg.shift,)
        amps = (2, 4, 8) if cfg.grid else (1,)
        choices = [replace(cfg, shift=shift, amplification=amp) for shift in shifts for amp in amps]
        if len(choices) == 1:
            configs[name] = cfg
        else:
            scored = [(score(choice), choice) for choice in choices]
            configs[name] = min(scored, key=lambda x: x[0])[1]
            candidates[name] = [{'score': value, **asdict(choice)} for value, choice in scored]
    return configs, dict(grids=fit, candidates=candidates, selected={k: asdict(v) for k, v in configs.items()})


def match_time(model, configs):
    def median_ms(config, n):
        return np.median([timed(model, config, n, 5000+i)[2] for i in range(5)])

    resolved = {}
    targets = {}
    for target_n in TIME_ANCHORS:
        target = median_ms(configs['MC'], target_n)
        targets[target_n] = float(target)
        for name, config in configs.items():
            if name == 'MC':
                resolved[target_n, name] = target_n
                continue
            lo, hi = MIN_BUDGET, MAX_BUDGET
            candidates = []
            for _ in range(9):
                n = (lo + hi)//2
                observed = median_ms(config, n)
                candidates.append((abs(np.log(observed/target)), n))
                if observed < target:
                    lo = n
                else:
                    hi = n
            for n in (MIN_BUDGET, lo, hi):
                candidates.append((abs(np.log(median_ms(config, n)/target)), n))
            resolved[target_n, name] = min(candidates)[1]
    return resolved, targets


def evaluate(model, configs, repeats):
    start = perf_counter_ns()
    timed_n, targets = match_time(model, configs)
    matching_seconds = (perf_counter_ns() - start) / 1e9
    jobs = [('exact', n, name, n) for n in BUDGETS for name, cfg in configs.items() if cfg.corrected]
    jobs += [('temps', n, name, timed_n[n, name]) for n in TIME_ANCHORS for name in configs]
    pure = configs['Spline seule']
    # Match the realized exact count of the uncorrected spline in its own small-budget comparison.
    pure_cost = pure.nodes + 32
    jobs += [('petit', pure_cost, 'Spline seule', 2048), ('petit', pure_cost, 'MC', pure_cost),
             ('petit', pure_cost, 'Stratifié', pure_cost), ('petit', pure_cost, 'IS + strat.', pure_cost)]
    rows = []
    for i in range(repeats):
        order = jobs.copy()
        random.Random(8500+i).shuffle(order)
        for mode, anchor, name, n in order:
            result, diag, ms = timed(model, configs[name], n, 900_000+i)
            rows.append(dict(model=model.name, mode=mode, anchor=anchor, method=name, replicate=i,
                             var=result[0], es=result[1], ms=ms, **diag))
    memory = {}
    for mode, anchor, name, n in jobs:
        tracemalloc.start()
        try:
            run(model, configs[name], n, 6000)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        memory[mode, anchor, name] = peak
    for row in rows:
        row['peak_bytes'] = memory[row['mode'], row['anchor'], row['method']]
    return rows, targets, matching_seconds


def evaluate_split_strat(model, model_meta, repeats):
    """Paired follow-up with the parent spline settings frozen."""
    selected = model_meta['calibration']['selected']
    parent = Config(**selected['Spline + strat.'])
    cfg = replace(parent, sampler='split_strat')
    selected[TRIPLE] = asdict(cfg)
    cases = [('exact', n, n) for n in BUDGETS]
    for anchor in TIME_ANCHORS:
        target = float(np.median([timed(model, Config(), anchor, 5100+i)[2] for i in range(9)]))
        lo, hi = MIN_BUDGET, MAX_BUDGET
        candidates = []
        for _ in range(11):
            n = (lo+hi)//2
            observed = float(np.median([timed(model, cfg, n, 5200+i)[2] for i in range(5)]))
            candidates.append((abs(np.log(observed/target)), n))
            if observed < target:
                lo = n
            else:
                hi = n
        floor = float(np.median([timed(model, cfg, MIN_BUDGET, 5200+i)[2] for i in range(5)]))
        candidates.append((abs(np.log(floor/target)), MIN_BUDGET))
        cases.append(('temps', anchor, min(candidates)[1]))
    rows = []
    bridge = {}
    for mode, anchor, n in cases:
        tracemalloc.start()
        try:
            run(model, cfg, n, 6000)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        parent_ms = []
        for i in range(repeats):
            seed = 900_000+i
            if i%2:
                parent_ms.append(timed(model, parent, n, seed)[2])
            result, diag, ms = timed(model, cfg, n, seed)
            if not i%2:
                parent_ms.append(timed(model, parent, n, seed)[2])
            rows.append(dict(model=model.name, mode=mode, anchor=anchor, method=TRIPLE, replicate=i,
                             var=result[0], es=result[1], ms=ms, **diag, peak_bytes=peak))
        bridge[f'{mode}:{anchor}'] = dict(n=n, parent_median_ms=float(np.median(parent_ms)),
                                         triple_median_ms=float(np.median([r['ms'] for r in rows[-repeats:]])))
    model_meta['decomposition_bridge'] = bridge
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results')
    parser.add_argument('--repeats', type=int, default=40)
    parser.add_argument('--ref-power', type=int, default=21)
    parser.add_argument('--add-split-strat', action='store_true', help='Ajouter le test de scission stratifiée aux mesures existantes.')
    parser.add_argument('--references-only', action='store_true', help='Affiner les références sans modifier les mesures.')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.smoke:
        args.repeats, args.ref_power = 4, 12
    if args.repeats < 2 or not 10 <= args.ref_power <= 23:
        parser.error('Il faut au moins deux répétitions et une puissance Sobol de 10 à 23.')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.references_only:
        path = output/'protocol.json'
        meta = json.loads(path.read_text())
        meta['ref_power'] = args.ref_power
        for name, data in meta['models'].items():
            start = perf_counter_ns()
            data['reference'] = reference(MODELS[name], args.ref_power)
            data['reference_seconds'] = (perf_counter_ns()-start)/1e9
        path.write_text(json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        return
    if args.add_split_strat:
        path = output/'protocol.json'
        meta = json.loads(path.read_text())
        with (output/'runs.csv').open(encoding='utf-8', newline='') as handle:
            existing = csv.DictReader(handle)
            fields = existing.fieldnames
            rows = [row for row in existing if row['method'] != TRIPLE]
        for model in MODELS.values():
            rows.extend(evaluate_split_strat(model, meta['models'][model.name], meta['repeats']))
        with (output/'runs.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        path.write_text(json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        print(f'{len(rows)} mesures après ajout du test de scission stratifiée.', flush=True)
        return
    meta = dict(host=platform.platform(), python=platform.python_version(), numpy=np.__version__,
                scipy=scipy_version, repeats=args.repeats, ref_power=args.ref_power,
                budgets=BUDGETS, time_anchors=TIME_ANCHORS, models={})
    all_rows = []
    for model in MODELS.values():
        print(model.name, 'calibration', flush=True)
        start = perf_counter_ns()
        configs, calibration = calibrate(model, args.smoke)
        calibration_seconds = (perf_counter_ns() - start) / 1e9
        print(model.name, 'référence et évaluation', flush=True)
        start = perf_counter_ns()
        ref = reference(model, args.ref_power)
        reference_seconds = (perf_counter_ns() - start) / 1e9
        rows, targets, matching_seconds = evaluate(model, configs, args.repeats)
        meta['models'][model.name] = dict(reference=ref, calibration=calibration, target_ms=targets,
                                         calibration_seconds=calibration_seconds,
                                         reference_seconds=reference_seconds, matching_seconds=matching_seconds)
        all_rows.extend(rows)
        all_rows.extend(evaluate_split_strat(model, meta['models'][model.name], args.repeats))
    fields = list(dict.fromkeys(key for row in all_rows for key in row))
    with (output/'runs.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    (output/'protocol.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(f'{len(all_rows)} mesures écrites dans {output}', flush=True)


if __name__ == '__main__':
    main()
