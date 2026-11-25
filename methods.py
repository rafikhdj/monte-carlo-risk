"""Estimators and exact-evaluation accounting."""
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import logsumexp, ndtr, ndtri

from models import ALPHA, MIX_P, TAIL


@dataclass(frozen=True)
class Config:
    sampler: str = 'mc'
    grid: str = ''
    shift: float = 1.
    width: float = 6.
    nodes: int = 31
    amplification: int = 1
    corrected: bool = True


METHODS = {
    'MC': Config(),
    'Stratifié': Config('strat'),
    'IS': Config('is'),
    'IS + strat.': Config('is_strat'),
    'Gaussiennes scindées': Config('split'),
    'Spline uniforme': Config(grid='uniform', amplification=2),
    'Spline quantiles': Config(grid='quantile', amplification=2),
    'Spline queue': Config(grid='tail', amplification=2),
    'Spline adaptative': Config(grid='adaptive', amplification=2),
    'Spline + strat.': Config('strat', 'uniform', amplification=2),
    'Spline + IS': Config('is', 'uniform', amplification=2),
    'Spline + IS + strat.': Config('is_strat', 'uniform', amplification=2),
    'Spline + scindées': Config('split', 'uniform', amplification=2),
    'Spline seule': Config('strat', 'uniform', amplification=4, corrected=False),
    'Spline + scindées + strat.': Config('split_strat', 'uniform', amplification=2),
}


def draw(rng, n, model, config):
    if config.sampler in ('strat', 'split_strat'):
        z = np.empty((n, model.dim))
        z[:, 0] = ndtri((np.arange(n) + rng.random(n)) / n)
        z[:, 1:] = rng.standard_normal((n, model.dim - 1))
        if config.sampler == 'split_strat':
            # Condition the independent Gaussian pair on its stratified sum.
            auxiliary = rng.standard_normal(z.shape)
            scale = np.sqrt(.5)
            left = (z + auxiliary) * scale
            right = (z - auxiliary) * scale
            z = (left + right) * scale
        return z, None
    z = rng.standard_normal((n, model.dim))
    if config.sampler == 'split':
        z += rng.standard_normal(z.shape)
        z *= np.sqrt(.5)
    if config.sampler in ('is', 'is_strat'):
        # Defensive Gaussian mixture, with weights computed for the marginal proposal.
        p = np.array([.20, .70, .10])
        mu = np.array([0., 1., -1.])[:, None] * np.array(model.shift) * config.shift
        if config.sampler == 'is_strat':
            u = (np.arange(n) + rng.random(n)) / n
            j = np.searchsorted(np.cumsum(p), u, side='right')
            lo = np.r_[0., np.cumsum(p)[:-1]]
            z[:, 0] = ndtri(np.clip((u - lo[j]) / p[j], 1e-15, 1-1e-15))
        else:
            j = rng.choice(3, size=n, p=p)
        z += mu[j]
        weight = np.exp(-logsumexp(z @ mu.T - .5 * np.sum(mu**2, axis=1) + np.log(p), axis=1))
        return z, weight
    return z, None


def risk(loss, weight=None):
    """Inverse empirical CDF and integral of its upper quantiles."""
    loss = np.asarray(loss)
    if weight is None:
        n = len(loss)
        k = int(np.ceil(ALPHA * n)) - 1
        ordered = np.partition(loss, k)
        var = ordered[k]
        es = var + np.sum(ordered[k+1:] - var) / (n * TAIL)
        return np.array([var, es])
    order = np.argsort(loss)
    y, w = loss[order], weight[order]
    masses = np.cumsum(w)
    q = min(np.searchsorted(masses, ALPHA * masses[-1]), len(y)-1)
    var = y[q]
    return np.array([var, var + w[q+1:] @ (y[q+1:] - var) / (masses[-1] * TAIL)])


def grid_points(kind, n, width):
    if kind in ('uniform', 'adaptive'):
        return np.linspace(-width, width, n)
    if kind == 'quantile':
        return np.r_[-width, ndtri(np.arange(1, n-1) / (n-1)), width]
    # Densify both tails in factor space without using a loss quantile.
    u = np.linspace(-1, 1, n)
    return width * (1. - np.abs(1. - np.abs(u))**2) * np.sign(u)


def surrogate(model, config):
    n = config.nodes
    width = config.width
    components = model.dim if model.name == 'options_4d' else 1

    def evaluate(x):
        return model.components(np.broadcast_to(x[:, None], (len(x), components)))

    if config.grid == 'adaptive':
        x = grid_points('uniform', 5, width)
        y = evaluate(x)
        # Each iteration spends exactly two midpoint probes. Both are retained.
        for _ in range((n-5)//2):
            score = np.max(np.abs(np.diff(y, axis=0) / np.diff(x)[:, None]), axis=1)
            score *= np.diff(x)**2
            best = np.argsort(score)[-2:]
            new_x = (x[best] + x[best+1]) / 2
            all_x = np.r_[x, new_x]
            order = np.argsort(all_x)
            x, y = all_x[order], np.vstack([y, evaluate(new_x)])[order]
    else:
        x = grid_points(config.grid, n, width)
        y = evaluate(x)
    spline = CubicSpline(x, y)
    left, right = spline(x[[0, -1]], 1)

    def predict(z):
        coord = model.coordinates(z)
        clipped = np.clip(coord, x[0], x[-1])
        if components == 1:
            fitted = spline(clipped[:, 0])[:, 0:1]
        else:
            fitted = np.column_stack([
                # Constructed once per factor, outside the evaluation loop below.
                piece(clipped[:, i]) for i, piece in enumerate(pieces)
            ])
        fitted += (coord - clipped) * np.where(coord < x[0], left, right)
        return model.aggregate(fitted, z), np.any(coord != clipped, axis=1)

    pieces = [CubicSpline(x, y[:, i]) for i in range(components)] if components > 1 else []
    return predict, len(x)


def run(model, config, budget, seed):
    """Budget is all exact scenarios, including training and diagnostic checks."""
    rng = np.random.default_rng(seed)
    n = budget
    diag = {}
    # Native unrestricted generators are the fastest classical baseline.
    if config == Config() and model.name in ('student', 'mixture'):
        if model.name == 'student':
            loss = rng.standard_t(5, size=n)
        else:
            g = rng.standard_normal(n)
            loss = np.where(rng.random(n) < MIX_P, 2.8 + 1.7*g, g)
        return risk(loss), dict(exact=n, n=n, training=0, checks=0, weight_mean=1., ess_fraction=1.)
    if config.grid:
        predict, training = surrogate(model, config)
        checks = min(32, budget//4)
        # These independent controls compare fitted and exact loss in the tails.
        audit, _ = draw(rng, checks, model, Config('is'))
        audit_pred, _ = predict(audit)
        residual = audit_pred - model.value(audit)
        diag['fit_rmse'] = np.sqrt(np.mean(residual**2))
        diag['fit_max'] = np.max(np.abs(residual))
        n = (budget-training-checks) * config.amplification
        if n < 10:
            raise ValueError('Budget insuffisant pour les nœuds et contrôles')
    else:
        training = checks = 0
    z, weight = draw(rng, n, model, config)
    if config.grid:
        loss, outside = predict(z)
        diag['extrapolation'] = np.average(outside, weights=weight)
        result = risk(loss, weight)
        exact = training + checks
        if config.corrected:
            # One representative per block is coupled to the coarse sample.
            # Draw its offset independently within each block to preserve stratification.
            ix = np.arange(0, n, config.amplification) + rng.integers(config.amplification, size=n//config.amplification)
            paired_weight = None if weight is None else weight[ix]
            fine = model.value(z[ix])
            exact += len(ix)
            correction = risk(fine, paired_weight) - risk(loss[ix], paired_weight)
            diag['correction_var'], diag['correction_es'] = correction
            result += correction
    else:
        exact = n
        result = risk(model.value(z), weight)
    diag['weight_mean'] = 1. if weight is None else np.mean(weight)
    diag['ess_fraction'] = 1. if weight is None else np.sum(weight)**2 / (n * np.dot(weight, weight))
    return result, dict(diag, exact=exact, n=n, training=training, checks=checks)
