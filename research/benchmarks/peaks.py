"""Transfer check for narrow Gaussian bumps at previously unused centers."""
import numpy as np

from research.benchmarks.run import ROOT, measurement, save_csv
from research.estimators import BASELINES, SURROGATES
from research.problems import Problem

CENTERS = (-1.4, .7, 2.2)
WIDTHS = (.05, .075, .12)
REPEATS = 32


def main():
    rows = []
    summaries = []
    rng = np.random.default_rng(850327)
    bootstrap = rng.integers(REPEATS, size=(3000, REPEATS))
    for center in CENTERS:
        for width in WIDTHS:
            def bump(x):
                return np.exp(-.5*((x[:,0]-center)/width)**2)

            ref = width/np.sqrt(1+width**2)*np.exp(-.5*center**2/(1+width**2))
            problem = Problem('transfer_peak',1,bump,ref,shift=center)
            group = {}
            for name, method, budget in [('Surrogate seul',SURROGATES['Surrogate seul'],64),
                                          *((n,m,b) for n,m in BASELINES.items() for b in (128,256))]:
                data = []
                for i in range(REPEATS):
                    m = measurement(problem,method,budget,810000+i)
                    data.append(m)
                    rows.append(dict(center=center,width=width,method=name,budget=budget,replicate=i,**m))
                errors = np.array([r['estimate']-ref for r in data])
                group[name,budget] = (errors,data[0]['exact'],float(np.median([r['cpu_ms'] for r in data])))
            errs, calls, cpu = group.pop(('Surrogate seul',64))
            best = min(group,key=lambda k:np.mean(group[k][0]**2))
            other, exact, other_cpu = group[best]
            baseline_boot = np.min([np.mean(v[0][bootstrap]**2,axis=1) for v in group.values()],axis=0)
            ratios = np.sqrt(np.mean(errs[bootstrap]**2,axis=1)/np.maximum(baseline_boot,1e-30))
            summaries.append(dict(center=center,width=width,reference=ref,candidate_calls=calls,
                                  candidate_error=np.sqrt(np.mean(errs**2)),candidate_cpu_ms=cpu,
                                  best_baseline=best[0],baseline_budget=best[1],baseline_calls=exact,
                                  baseline_rmse=np.sqrt(np.mean(other**2)),baseline_cpu_ms=other_cpu,
                                  bootstrap_high=np.quantile(ratios,.975)))
    save_csv(ROOT/'results'/'peak_transfer.csv',summaries)
    save_csv(ROOT/'results'/'peak_runs.csv',rows)


if __name__=='__main__':
    main()
