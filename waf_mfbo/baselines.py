"""Distinct finite-pool adaptations of the paper's engineering comparison controls.

All objective/runtime arguments below contain completed observations only.
The adaptation conventions and primary algorithm sources are documented in README.
"""
import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.special import log_ndtr, logsumexp
from sklearn.neighbors import KernelDensity

from .models import MultiFidelityGP, rbf


METHODS = ['mfmes', 'async-mfbo', 'robust-mfbo', 'fabolas', 'hyperband', 'bohb']


class EnvironmentGP(MultiFidelityGP):
    """FABOLAS product kernel with [1,(1-s)^2] objective or [1,s] log-cost basis."""
    def __init__(self, target_fidelity, cost=False):
        super().__init__(target_fidelity)
        self.cost = cost

    def kernel(self, a, b, discrepancy=False):
        sa, sb = 2. ** (a[:, -1] - self.target), 2. ** (b[:, -1] - self.target)
        fa = np.c_[np.ones(len(a)), sa if self.cost else (1 - sa) ** 2]
        fb = np.c_[np.ones(len(b)), sb if self.cost else (1 - sb) ** 2]
        return rbf(a[:, :-1], b[:, :-1]) * (fa @ fb.T)


def draw_minima(model, grid, samples, rng):
    mean, cov = model.posterior(grid)
    cov = (cov + cov.T) / 2 + np.eye(len(grid)) * 1e-8
    return rng.multivariate_normal(mean, cov, size=samples).min(1)


def mf_mes(model, candidate, minima, quadrature=40):
    """Minimization MF-MES via the conditional Gaussian lower-truncation density."""
    target_point = candidate.copy()
    target_point[-1] = model.target
    mean, cov = model.posterior(np.stack([candidate, target_point]))
    var_y, var_f = max(float(cov[0, 0]), 1e-10), max(float(cov[1, 1]), 1e-10)
    if int(candidate[-1]) == model.target:
        gamma = (np.asarray(minima) - mean[0]) / np.sqrt(var_y)
        log_z = log_ndtr(-gamma)
        mills = np.exp(-.5 * gamma ** 2 - .5 * np.log(2 * np.pi) - log_z)
        return max(0., float(np.mean(-log_z - .5 * gamma * mills)))
    cross = float(cov[0, 1])
    conditional_var = max(var_f - cross * cross / var_y, 1e-12)
    nodes, weights = hermgauss(quadrature)
    y = mean[0] + np.sqrt(2 * var_y) * nodes
    log_normal = -.5 * np.log(2 * np.pi * var_y) - .5 * (y - mean[0]) ** 2 / var_y
    conditional_mean = mean[1] + cross / var_y * (y - mean[0])
    entropy = .5 * np.log(2 * np.pi * np.e * var_y)
    gains = []
    for minimum in minima:
        log_z = log_ndtr((mean[1] - minimum) / np.sqrt(var_f))
        log_ratio = log_ndtr((conditional_mean - minimum) / np.sqrt(conditional_var)) - log_z
        log_mass = np.log(weights) - .5 * np.log(np.pi) + log_ratio
        normalized = np.exp(log_mass - logsumexp(log_mass))
        conditional_entropy = -np.sum(normalized * (log_normal + log_ratio))
        gains.append(entropy - conditional_entropy)
    return max(0., float(np.mean(gains)))


def minimizer_entropy_gain(model, grid, candidate, samples, fantasies, rng):
    """FABOLAS entropy search on a finite target minimizer grid, not a KG proxy."""
    mean, cov = model.posterior(grid)
    y_mean, y_cov = model.posterior(candidate[None])
    _, cross = model.posterior(grid, candidate[None])
    var = max(float(y_cov[0, 0]) + model.noise * model.scale ** 2, 1e-10)
    conditional = (cov - cross @ cross.T / var)

    def factor(covariance):
        return np.linalg.cholesky((covariance + covariance.T) / 2 + np.eye(len(grid)) * 1e-8)

    def entropy(mu, cholesky):
        draws = mu + rng.normal(size=(samples, len(grid))) @ cholesky.T
        counts = np.bincount(draws.argmin(1), minlength=len(grid)) / samples
        return -np.sum(counts[counts > 0] * np.log(counts[counts > 0]))

    prior = entropy(mean, factor(cov))
    conditional_factor = factor(conditional)
    updated = [entropy(mean + cross[:, 0] / var * value, conditional_factor)
               for value in rng.normal(0, np.sqrt(var), fantasies)]
    return max(0., float(prior - np.mean(updated)))


def density_ratio(locations, losses, candidates, rng, random_fraction=1 / 3):
    """BOHB Gaussian KDE good/bad density ratio on the highest supported resource level."""
    if len(losses) < max(4, locations.shape[1] + 1) or rng.random() < random_fraction:
        return rng.random(len(candidates))
    ordered = np.argsort(losses)
    count = max(2, int(np.ceil(.15 * len(ordered))))
    count = min(count, len(ordered) - 2)
    scale = locations.std(0).clip(.05)
    good = locations[ordered[:count]] / scale
    bad = locations[ordered[count:]] / scale
    bandwidth = max(.05, len(locations) ** (-1 / (locations.shape[1] + 4)))
    low = KernelDensity(bandwidth=bandwidth).fit(good)
    high = KernelDensity(bandwidth=bandwidth).fit(bad)
    return low.score_samples(candidates / scale) - high.score_samples(candidates / scale)


class Halving:
    """Rotating Hyperband brackets; promotions wait for observed cohort outcomes."""
    def __init__(self, locations, eta=3, bohb=False):
        self.locations = locations
        self.levels = np.unique(locations[:, -1]).astype(int)
        self.eta, self.bohb = eta, bohb
        self.bracket = 0
        self.cohort, self.rung, self.promotions = [], None, []

    def select(self, candidates, observed_ids, observed_y, rng):
        observed = dict(zip(observed_ids, observed_y))
        candidate_set = set(candidates)
        for _ in range(len(self.levels) * 3):
            if self.promotions:
                rejected = {i for i in self.promotions if i not in candidate_set}
                self.promotions = [i for i in self.promotions if i in candidate_set]
                self.cohort = [i for i in self.cohort if i not in rejected]
                if self.promotions:
                    return self.promotions.pop(0)
            if self.cohort:
                if not all(i in observed for i in self.cohort):
                    return None
                good = [i for i in self.cohort if np.isfinite(observed[i])]
                good.sort(key=lambda i: observed[i])
                next_level = self.rung + 1
                next_cohort = []
                if next_level < len(self.levels):
                    for old in good[:max(1, len(good) // self.eta)]:
                        match = np.flatnonzero((self.locations[:, -1] == self.levels[next_level]) &
                                               np.all(self.locations[:, :-1] == self.locations[old, :-1], axis=1))
                        next_cohort.extend(int(i) for i in match if int(i) in candidate_set or int(i) in observed)
                if next_cohort:
                    self.cohort, self.rung = next_cohort, next_level
                    self.promotions = [i for i in next_cohort if i in candidate_set]
                    if self.promotions:
                        return self.promotions.pop(0)
                    continue
                self.cohort = []
                self.bracket += 1
            start = self.bracket % len(self.levels)
            level = self.levels[start]
            available = candidates[self.locations[candidates, -1] == level]
            if not len(available):
                self.bracket += 1
                continue
            if self.bohb:
                exact = np.array([i for i in observed_ids if np.isfinite(observed[i])], dtype=int)
                supported = [f for f in self.levels if np.sum(self.locations[exact, -1] == f) >= max(4, self.locations.shape[1])]
                ids = exact[self.locations[exact, -1] == max(supported)] if supported else exact
                scores = density_ratio(self.locations[ids, :-1], np.array([observed[i] for i in ids]), self.locations[available, :-1], rng)
                ordered = available[np.argsort(scores)[::-1]]
            else:
                ordered = rng.permutation(available)
            count = min(len(ordered), self.eta ** (len(self.levels) - start - 1))
            self.cohort = ordered[:count].tolist()
            self.rung = start
            self.promotions = self.cohort.copy()
            return self.promotions.pop(0)
        return None


class Selector:
    """History-only acquisition state, including rMFBO's distinct pseudo-SF trajectory."""
    def __init__(self, locations, method, args):
        self.locations, self.method, self.args = locations, method, args
        self.target = int(locations[:, -1].max())
        self.grid = np.c_[np.unique(locations[:, :-1], axis=0), np.full(len(np.unique(locations[:, :-1], axis=0)), self.target)]
        self.halving = Halving(locations, bohb=method == 'bohb')
        self.pseudo, self.proposed = [], {}
        self.diagnostic = {}

    def completed(self, task, value):
        point = self.proposed.pop(task, None)
        if point is not None and np.isfinite(value):
            self.pseudo.append(point)

    def select(self, candidates, completed_ids, values, runtimes, censored, pending, constraints, rng):
        locations, args = self.locations, self.args
        candidates = np.asarray(candidates, dtype=int)
        completed_ids = np.asarray(completed_ids, dtype=int)
        values, runtimes, censored = np.asarray(values), np.asarray(runtimes), np.asarray(censored, bool)
        for column in range(constraints.shape[1]):
            valid = np.isfinite(constraints[:, column]) & ~censored
            model = MultiFidelityGP(self.target).fit(locations[completed_ids[valid]], constraints[valid, column])
            from scipy.special import ndtr
            gm, gc = model.posterior(locations[candidates])
            candidates = candidates[ndtr(gm / np.sqrt(np.diag(gc).clip(1e-10))) <= args.violation_tolerance]
        if not len(candidates):
            return None
        if self.method in ['hyperband', 'bohb']:
            return self.halving.select(candidates, completed_ids, values, rng)
        full_candidates = candidates.copy()
        if len(candidates) > args.candidates:
            candidates = rng.choice(candidates, args.candidates, replace=False)
        valid = np.isfinite(values) & ~censored
        kind = EnvironmentGP if self.method == 'fabolas' else MultiFidelityGP
        model = kind(self.target).fit(locations[completed_ids[valid]], values[valid])
        exact = ~censored
        cost = EnvironmentGP(self.target, cost=True).fit(locations[completed_ids[exact]], np.log(runtimes[exact]))
        cost_mean, cost_cov = cost.posterior(locations[candidates])
        durations = np.exp(np.clip(cost_mean + .5 * np.diag(cost_cov), -10, 20))
        if not np.any(exact):
            durations = 2. ** locations[candidates, -1]
        minima = draw_minima(model, self.grid, args.samples, rng)
        if self.method == 'fabolas':
            gains = np.array([minimizer_entropy_gain(model, self.grid, locations[i], args.samples, args.fantasies, rng) for i in candidates])
        else:
            gains = np.array([mf_mes(model, locations[i], minima) for i in candidates])
        score = gains / durations.clip(1e-6)
        if self.method == 'async-mfbo' and len(pending):
            # Soft local penalization around busy target predictions, not a failure/deadline discount.
            grid_mean, grid_cov = model.posterior(self.grid)
            distance = np.linalg.norm(self.grid[:, None, :-1] - self.grid[None, :, :-1], axis=-1)
            slopes = abs(grid_mean[:, None] - grid_mean[None]) / np.maximum(distance, 1e-8)
            lipschitz = max(float(slopes.max()), model.scale / .3, 1e-6)
            lower = float(np.min(grid_mean - 2 * np.sqrt(np.diag(grid_cov).clip(1e-10))))
            from scipy.special import ndtr
            for point in pending:
                projected = point.copy()
                projected[-1] = self.target
                pm, pc = model.posterior(projected[None])
                distances = np.linalg.norm(locations[candidates, :-1] - projected[:-1], axis=1)
                score *= ndtr((lipschitz * distances - max(0., pm[0] - lower)) / np.sqrt(max(float(pc[0, 0]), 1e-10)))
        chosen = int(np.argmax(score))
        task = int(candidates[chosen])
        self.diagnostic = dict(information_gain=float(gains[chosen]), expected_seconds=float(durations[chosen]), score=float(score[chosen]))
        if self.method == 'robust-mfbo':
            target = completed_ids[valid & (locations[completed_ids, -1] == self.target)]
            observed = dict(zip(completed_ids, values))
            sf = MultiFidelityGP(self.target).fit(locations[target], np.array([observed[i] for i in target]))
            pseudo_y = []
            for point in self.pseudo:
                if len(target):
                    nearest = target[np.argmin(np.linalg.norm(locations[target, :-1] - point[:-1], axis=1))]
                    sm = sf.posterior(locations[nearest:nearest + 1])[0][0]
                    mm = model.posterior(locations[nearest:nearest + 1])[0][0]
                    source = sf if abs(sm - observed[nearest]) <= abs(mm - observed[nearest]) else model
                else:
                    source = model
                pseudo_y.append(source.posterior(point[None])[0][0])
            if self.pseudo:
                sf.fit(np.r_[locations[target], np.stack(self.pseudo)], np.r_[[observed[i] for i in target], pseudo_y])
            sf_candidates = full_candidates[locations[full_candidates, -1] == self.target]
            if not len(sf_candidates):
                return None
            sf_minima = draw_minima(sf, self.grid, args.samples, rng)
            sf_gain = [mf_mes(sf, locations[i], sf_minima) for i in sf_candidates]
            sf_task = int(sf_candidates[np.argmax(sf_gain)])
            _, uncertainty = model.posterior(locations[sf_task:sf_task + 1])
            gate = np.sqrt(max(float(uncertainty[0, 0]), 0)) <= args.robust_variance
            gate &= locations[task, -1] == self.target or score[chosen] >= args.robust_information
            if gate:
                self.proposed[task] = locations[sf_task].copy()
            else:
                task = sf_task
            self.diagnostic.update(pseudo_sf_task=sf_task, robust_multifidelity_gate=bool(gate))
        cm, cc = cost.posterior(locations[task:task + 1])
        self.diagnostic['runtime_quantiles'] = np.exp(np.clip(cm[0] + np.sqrt(max(float(cc[0, 0]), 1e-10)) *
                                                               np.array([-1.28155, 0., 1.28155]), -20, 30)).tolist()
        return task
