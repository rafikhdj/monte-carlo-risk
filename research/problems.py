"""Integrands in normal coordinates, plus independent reference integrals."""
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import numpy as np
from scipy.special import ndtr, roots_hermitenorm, roots_legendre


@dataclass(frozen=True)
class Problem:
    name: str
    dimension: int
    integrand: Callable
    truth: float | None
    shift: float = .8
    breaks: tuple[float, ...] = ()


@lru_cache
def inner_rule(order):
    nodes, weights = roots_hermitenorm(order)
    return np.exp(.18 * nodes - .5*.18**2), weights / np.sqrt(2*np.pi)


def asian(x, order=64):
    """Four-date arithmetic Asian, condition on two increments, integrate two."""
    step, w = inner_rule(order)
    s1 = np.exp(.18*x[:, 0] - .5*.18**2)
    s2 = s1 * np.exp(.18*x[:, 1] - .5*.18**2)
    s3 = s2[:, None] * step
    strike = 4.04 - s1[:, None] - s2[:, None] - s3
    log_moneyness = np.log(s3 / np.maximum(strike, 1e-16))
    d1 = (log_moneyness + .5*.18**2) / .18
    conditional = s3*ndtr(d1) - strike*ndtr(d1-.18)
    return .25 * (conditional @ w)


def interaction(d):
    freq = 1.1
    # Ring neighbors plus weak pairwise products. No additive decomposition.
    def value(x):
        other = np.roll(x, -1, axis=1)
        return (np.cos(freq*(x+other)) + .3*x*other).mean(axis=1)
    return Problem(f'interaction_{d}d', d, value, np.exp(-freq**2))


PROBLEMS = {p.name: p for p in (
    Problem('polynomial', 1, lambda x: .4*x[:,0] + .2*x[:,0]**4, .6),
    Problem('exponential', 1, lambda x: np.exp(.85*x[:,0]), np.exp(.85**2/2), 1.5),
    Problem('oscillatory', 1, lambda x: np.cos(5*x[:,0])+np.sin(7*x[:,0]), np.exp(-12.5), 0.),
    Problem('local_curvature', 1, lambda x: np.exp(-.5*((x[:,0]-1.8)/.075)**2),
            .075/np.sqrt(1+.075**2)*np.exp(-.5*1.8**2/(1+.075**2)), 1.8),
    Problem('discontinuous', 1, lambda x: (x[:,0]>.37).astype(float), ndtr(-.37), .6, (.37,)),
    Problem('rare_tail', 2, lambda x: ((x[:,0]+x[:,1])/np.sqrt(2)>3.4).astype(float),
            ndtr(-3.4), 3.4),
    Problem('european', 1, lambda x: np.maximum(np.exp(.25*x[:,0]-.5*.25**2)-1.08,0),
            ndtr((-np.log(1.08)+.5*.25**2)/.25)-1.08*ndtr((-np.log(1.08)-.5*.25**2)/.25),
            1.2, ((np.log(1.08)+.5*.25**2)/.25,)),
    Problem('asian_nested', 2, asian, None, 1.2),
    *(interaction(d) for d in (2, 5, 10, 20)),
)}


def interval_rule(n, lo, hi):
    points, w = roots_legendre(n)
    x = .5*(hi-lo)*points + .5*(hi+lo)
    return x, .5*(hi-lo)*w*np.exp(-.5*x*x)/np.sqrt(2*np.pi)


def tensor_integral(fun, order, limit=9.):
    x, w = interval_rule(order, -limit, limit)
    a, b = np.meshgrid(x,x,indexing='ij')
    z = np.column_stack((a.ravel(),b.ravel()))
    values = np.concatenate([fun(batch) for batch in np.array_split(z, max(1, len(z)//1024))])
    return w @ values.reshape(order, order) @ w


def reference(problem):
    if problem.truth is not None:
        return dict(value=float(problem.truth), uncertainty=0., inner_delta=0., kind='analytical')
    coarse = tensor_integral(asian, 128)
    fine = tensor_integral(asian, 224)
    extended = tensor_integral(asian, 256, 10.)
    inner = tensor_integral(lambda x: asian(x, 128), 224)
    return dict(value=float(fine), uncertainty=float(max(abs(fine-coarse),abs(fine-extended))),
                inner_delta=float(abs(fine-inner)), kind='independent tensor quadrature')
