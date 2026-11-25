"""Sampling, control variates and explicit accounting of exact calls."""
from dataclasses import dataclass
import warnings

import numpy as np
from numpy.polynomial import Chebyshev
from scipy.integrate import IntegrationWarning, quad
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.special import ndtr, ndtri, roots_hermitenorm, roots_legendre
from scipy.spatial.distance import cdist
from scipy.stats import qmc

from .problems import interval_rule


@dataclass(frozen=True)
class Method:
    sampler: str
    surrogate: str = ''
    fraction: float = .25
    audit: bool = False


BASELINES = {
    'MC': Method('mc'),
    'Stratifié': Method('strat'),
    'IS': Method('is'),
    'Antithétique': Method('anti'),
    'Sobol': Method('sobol'),
    'Contrôle polynomial': Method('mc','control'),
    'Quadrature': Method('quad'),
    'Gauss-Hermite': Method('hermite'),
    'Gauss-Kronrod adaptatif': Method('adaptive_quad'),
    'Gaussiennes scindées': Method('split'),
}
SURROGATES = {
    'Surrogate seul': Method('none','auto'),
    'Spline uniforme + MC': Method('mc','cubic'),
    'Spline quantiles + MC': Method('mc','quantile'),
    'PCHIP + MC': Method('mc','pchip'),
    'Chebyshev + MC': Method('mc','cheb'),
    'Adaptatif + MC': Method('mc','adaptive'),
    'Surrogate + MC': Method('mc','auto'),
    'Surrogate + strat.': Method('strat','auto'),
    'Surrogate + IS': Method('is','auto'),
    'Surrogate + Sobol': Method('sobol','auto',.5),
    'Surrogate + allocation': Method('mc','auto',audit=True),
}


class Evaluator:
    def __init__(self, problem):
        self.problem = problem
        self.count = 0

    def __call__(self, x):
        self.count += len(x)
        return self.problem.integrand(x)


def standard_normals(u):
    return ndtri(np.clip(u, 2.**-54, 1.-2.**-53))


def draws(kind, n, problem, rng):
    d = problem.dimension
    if kind == 'sobol':
        engine = qmc.Sobol(d, scramble=True, bits=53, rng=rng)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='The balance properties of Sobol')
            x = standard_normals(engine.random(n))
        return x, None
    if kind == 'strat':
        u = (rng.permuted(np.broadcast_to(np.arange(n)[:,None],(n,d)), axis=0)
             + rng.random((n,d))) / n
        return standard_normals(u), None
    if kind == 'anti':
        z = rng.standard_normal(((n+1)//2,d))
        return np.vstack((z,-z))[:n], None
    x = rng.standard_normal((n,d))
    if kind == 'split':
        x = (x + rng.standard_normal(x.shape)) / np.sqrt(2.)
    if kind != 'is' or not problem.shift:
        return x, None
    direction = np.ones(d) * problem.shift/np.sqrt(d)
    shift = rng.random(n) < .8
    x[shift] += direction
    log_q_over_p = np.logaddexp(np.log(.2), np.log(.8)+x@direction-.5*(direction@direction))
    return x, np.exp(-log_q_over_p)


def hermite(f, problem, n):
    order = n if problem.dimension==1 else int(np.sqrt(n))
    x,w = roots_hermitenorm(order)
    w /= np.sqrt(2*np.pi)
    if problem.dimension==1:
        return w @ f(x[:,None])
    a,b = np.meshgrid(x,x,indexing='ij')
    return w @ f(np.column_stack((a.ravel(),b.ravel()))).reshape(order,order) @ w


def adaptive_quadrature(f, problem, n):
    limit = (n + 21)//42

    def weighted_value(x):
        return np.exp(-.5*x*x)/np.sqrt(2*np.pi)*f(np.array([[x]]))[0]

    # A 21-point initial rule and 42 new calls per bisection.
    # The QUADPACK warning when the budget is exhausted is expected.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', IntegrationWarning)
        value, _ = quad(weighted_value, -9., 9., limit=limit, epsabs=1e-12, epsrel=1e-12)
    return value


def quadrature(f, problem, n):
    if problem.dimension == 1:
        bounds = (-9., *problem.breaks, 9.)
        orders = np.full(len(bounds)-1,n//(len(bounds)-1))
        orders[:n % len(orders)] += 1
        rules = [interval_rule(int(k),lo,hi) for k,lo,hi in zip(orders,bounds[:-1],bounds[1:])]
        x,w = (np.concatenate(s) for s in zip(*rules))
        return w @ f(x[:,None])
    order = int(np.sqrt(n))
    x,w = interval_rule(order,-9.,9.)
    a,b = np.meshgrid(x,x,indexing='ij')
    return w @ f(np.column_stack((a.ravel(),b.ravel()))).reshape(order,order) @ w


def fit_one_dimensional(f, kind, n):
    width = 4.5
    if kind == 'adaptive':
        x = np.linspace(-width,width,7)
        y = f(x[:,None])
        while len(x)<n:
            slopes = np.diff(y)/np.diff(x)
            local = np.abs(np.diff(slopes,prepend=slopes[0])) + np.abs(np.diff(slopes,append=slopes[-1]))
            scores = np.diff(x)**2*(.005+local)*np.exp(-.25*((x[:-1]+x[1:])/2)**2)
            idx = np.argsort(scores)[-min(4,n-len(x)):]
            mid = .5*(x[idx]+x[idx+1])
            order = np.argsort(np.r_[x,mid])
            x,y = np.r_[x,mid][order],np.r_[y,f(mid[:,None])][order]
    else:
        if kind == 'quantile':
            x = np.r_[-width, ndtri(np.arange(1,n-1)/(n-1)), width]
        elif kind == 'cheb':
            x = np.sort(width*np.cos(np.arange(n)*np.pi/(n-1)))
        else:
            x = np.linspace(-width,width,n)
        y = f(x[:,None])
    if kind == 'cheb':
        fit = Chebyshev.fit(x,y,min(n-1,24),domain=[-width,width])
    elif kind == 'pchip':
        # Negligible ordinates can overflow the reciprocal-slope formula.
        safe_y = np.where(np.abs(y)<1e-280,0.,y)
        fit = PchipInterpolator(x,safe_y)
    else:
        fit = CubicSpline(x,y)

    def predict(z):
        return fit(np.clip(z[:,0],-width,width))

    # For each smooth piece integrate fit times the normal density.
    # Clipping makes the mass beyond the training interval explicit.
    intervals = np.array([-width,width]) if kind=='cheb' else x
    points, weights = roots_legendre(48 if kind=='cheb' else 12)
    half = .5*np.diff(intervals)
    center = .5*(intervals[:-1]+intervals[1:])
    z = center[:,None]+half[:,None]*points
    mass = half[:,None]*weights*np.exp(-.5*z*z)/np.sqrt(2*np.pi)
    mean = np.sum(mass*fit(z)) + ndtr(-width)*(fit(-width)+fit(width))
    return predict,float(mean)


def fit_regression(f, kind, n, problem, rng):
    x,_ = draws('sobol',n,problem,rng)
    y = f(x)
    d = problem.dimension
    centers = x[:min(n//3,36)] if kind=='radial' else np.empty((0,d))
    scale2 = max(.7,d*.65)
    radial_mean = np.exp(-.5*np.sum(centers**2,axis=1)/(scale2+1))*(scale2/(1+scale2))**(.5*d)
    if kind=='radial' and d<=2:
        powers = np.array([(i,j) for i in range(6) for j in range(6-i) if 1<=i+j<=5])
        gaussian_moments = np.array([1,0,1,0,3,0])
    else:
        powers = np.empty((0,2), dtype=int)

    def features(z):
        # Features have zero target mean, so the intercept is the integral.
        radial = np.exp(-.5*cdist(z,centers,'sqeuclidean')/scale2) - radial_mean
        if len(powers):
            polynomials = np.prod(z[:,None,:]**powers[None,:,:], axis=2)
            polynomials -= np.prod(gaussian_moments[powers],axis=1)
            return np.column_stack((polynomials,radial))
        return np.column_stack((z,z*z-1,radial))

    h = features(x)
    design = np.column_stack((np.ones(n),h))
    gram = design.T@design
    penalty = np.r_[0., np.full(h.shape[1], .001*n if kind=='control' else .00003*n)]
    beta = np.linalg.solve(gram+np.diag(penalty),design.T@y)

    def predict(z):
        return beta[0] + features(z)@beta[1:]

    return predict,float(beta[0])


def fit(f, kind, n, problem, rng):
    if kind == 'control' or problem.dimension>1:
        return fit_regression(f,'control' if kind=='control' else 'radial',n,problem,rng)
    return fit_one_dimensional(f,'cubic' if kind=='auto' else kind,n)


def run(problem, method, budget, seed):
    f = Evaluator(problem)
    rng = np.random.default_rng(seed)
    training = audit = 0
    audit_rmse = None
    if method.sampler == 'quad':
        estimate = quadrature(f,problem,budget)
    elif method.sampler == 'adaptive_quad':
        estimate = adaptive_quadrature(f,problem,budget)
    elif method.sampler == 'hermite':
        estimate = hermite(f,problem,budget)
    elif method.surrogate:
        floor = min(budget//2,max(12,2*problem.dimension+4))
        training = max(floor, int(budget*method.fraction))
        if method.sampler == 'none':
            training = budget
        if method.audit:
            audit = max(8,budget//8)
            predict,mean = fit(f,method.surrogate,training,problem,rng)
            x,_ = draws('mc',audit,problem,rng)
            residual = f(x)-predict(x)
            audit_rmse = float(np.sqrt(np.mean(residual**2)))
            # Select a residual sampler from independent probes, then draw afresh.
            score = abs(np.corrcoef(residual,x[:,0])[0,1]) if np.std(residual)>0 else 0.
            if score<.45:
                correction_sampler = 'mc'
            else:
                correction_sampler = 'strat'
        else:
            correction_sampler = method.sampler
            predict,mean = fit(f,method.surrogate,training,problem,rng)
        estimate = mean
        if correction_sampler!='none':
            x,w = draws(correction_sampler,budget-training-audit,problem,rng)
            residual = f(x)-predict(x)
            estimate += np.mean(residual if w is None else w*residual)
    else:
        x,w = draws(method.sampler,budget,problem,rng)
        y = f(x)
        estimate = np.mean(y if w is None else w*y)
    return float(estimate),f.count,training,audit,audit_rmse
