"""Loss models in independent standard-normal coordinates."""
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy import optimize, special, stats

ALPHA = 0.99
TAIL = 1 - ALPHA
MIX_P = .12
MIX_THRESHOLD = special.ndtri(1 - MIX_P)


@dataclass(frozen=True)
class Model:
    name: str
    dim: int
    shift: tuple[float, ...]

    @cached_property
    def cholesky(self):
        corr = .55 ** np.abs(np.arange(self.dim)[:, None] - np.arange(self.dim))
        return np.linalg.cholesky(corr)

    @cached_property
    def option_parameters(self):
        i = np.arange(self.dim)
        strikes = np.linspace(86, 114, 5)
        vol = (.20 + .015 * i)[:, None]
        positions = np.array([.20, .25, .35, .17, .09])
        root = vol * np.sqrt(.45)
        log_strike = np.log(strikes)
        rate = .02
        offset = -log_strike + (rate + .5 * vol**2) * .45
        return root, offset, strikes * np.exp(-rate * .45), positions

    @cached_property
    def option_initial(self):
        return self.option_value(np.zeros((1, self.dim)))[0]

    def option_value(self, x):
        root, offset, discounted, positions = self.option_parameters
        log_spot = np.log(100) - np.linspace(.13, .17, self.dim) * x
        spot = np.exp(log_spot)
        d1 = (log_spot[..., None] + offset) / root
        d2 = d1 - root
        put = discounted * special.ndtr(-d2) - spot[..., None] * special.ndtr(-d1)
        # Short puts plus short calls. Call-put parity avoids duplicate CDFs.
        call = put + spot[..., None] - discounted
        return (put + .36 * call) @ positions

    def coordinates(self, z):
        if self.name == 'options_4d':
            return z @ self.cholesky.T
        return z[:, :1]

    def components(self, x):
        if self.name == 'student':
            return np.copysign(special.stdtrit(5, special.ndtr(-np.abs(x))) * -1, x)
        if self.name.startswith('options'):
            return self.option_value(x) - self.option_initial
        return x

    def aggregate(self, parts, z):
        if self.name == 'mixture':
            tail = z[:, 1] > MIX_THRESHOLD
            return np.where(tail, 2.8 + 1.7 * parts[:, 0], parts[:, 0])
        return parts[:, 0] if self.dim == 1 else parts.sum(axis=1)

    def value(self, z):
        return self.aggregate(self.components(self.coordinates(z)), z)

    def analytic(self):
        if self.name == 'normal':
            q = stats.norm.ppf(ALPHA)
            return q, stats.norm.pdf(q) / TAIL
        if self.name == 'student':
            q = stats.t.ppf(ALPHA, df=5)
            return q, (5 + q*q) / 4 * stats.t.pdf(q, df=5) / TAIL
        if self.name == 'mixture':
            p = np.array([1-MIX_P, MIX_P])
            mu = np.array([0, 2.8])
            sigma = np.array([1, 1.7])
            q = optimize.brentq(lambda x: p @ stats.norm.sf((x-mu)/sigma) - TAIL, -12, 25)
            u = (q-mu)/sigma
            return q, p @ (mu * stats.norm.sf(u) + sigma * stats.norm.pdf(u)) / TAIL
        return None


MODELS = {model.name: model for model in (
    Model('normal', 1, (2.,)),
    Model('student', 1, (2.,)),
    Model('mixture', 2, (1.8, .8)),
    Model('options_1d', 1, (2.,)),
    Model('options_4d', 4, (1.7, .65, .45, .20)),
)}
