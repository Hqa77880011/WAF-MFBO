import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import log_ndtr
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF


def rbf(a, b, length=0.3):
    return np.exp(-0.5 * np.sum(((a[:, None] - b[None]) / length) ** 2, -1))


class MultiFidelityGP:
    def __init__(self, target_fidelity, noise=1e-4):
        self.target = target_fidelity
        self.noise = noise

    def kernel(self, a, b, discrepancy=False):
        base = rbf(a[:, :-1], b[:, :-1])
        same_auxiliary = (a[:, -1, None] == b[None, :, -1]) & (a[:, -1, None] != self.target)
        return base * same_auxiliary if discrepancy else base * (1 + same_auxiliary)

    def fit(self, x, y):
        self.x = np.asarray(x)
        self.mean = float(np.mean(y)) if len(y) else 0
        self.scale = max(float(np.std(y)), 1e-3) if len(y) else 1
        if len(y):
            self.factor = cho_factor(self.kernel(self.x, self.x) + np.eye(len(y)) * self.noise, lower=True)
            self.alpha = cho_solve(self.factor, (np.asarray(y) - self.mean) / self.scale)
        return self

    def posterior(self, a, b=None, discrepancy=False):
        a = np.atleast_2d(a)
        b = a if b is None else np.atleast_2d(b)
        prior = self.kernel(a, b, discrepancy)
        if not len(self.x):
            return np.full(len(a), 0 if discrepancy else self.mean), prior * self.scale ** 2
        ka = self.kernel(a, self.x, discrepancy)
        kb = self.kernel(b, self.x, discrepancy)
        mean = (0 if discrepancy else self.mean) + self.scale * (ka @ self.alpha)
        covariance = (prior - ka @ cho_solve(self.factor, kb.T)) * self.scale ** 2
        return mean, covariance


class RuntimeModel:
    def fit(self, features, times, censored, fidelities):
        self.x = np.asarray(features)
        self.noise = 0.45
        times, censored = np.asarray(times), np.asarray(censored, bool)
        self.prior = np.array([np.log(2 ** int(f)) for f in fidelities])
        if not len(features):
            self.latent = np.empty(0)
            return self
        self.covariance = rbf(self.x, self.x, 0.6) + np.eye(len(features)) * 1e-5
        inverse = np.linalg.inv(self.covariance)
        observed = np.log(np.maximum(times, 1e-8))
        variance = self.noise ** 2

        def objective(latent):
            residual = latent - self.prior
            z = (observed - latent) / self.noise
            exact = ~censored
            loss = 0.5 * residual @ inverse @ residual + 0.5 * np.sum(z[exact] ** 2)
            loss -= log_ndtr(-z[censored]).sum()
            gradient = inverse @ residual
            gradient[exact] += (latent[exact] - observed[exact]) / variance
            mills = np.exp(-0.5 * z[censored] ** 2 - 0.5 * np.log(2 * np.pi) - log_ndtr(-z[censored]))
            gradient[censored] -= mills / self.noise
            return float(loss), gradient

        fitted = minimize(objective, self.prior.copy(), jac=True, method="L-BFGS-B")
        if not np.isfinite(fitted.fun):
            raise ArithmeticError("Censored runtime model fit failed")
        self.latent = fitted.x
        z = (observed - self.latent) / self.noise
        mills = np.exp(-0.5 * z ** 2 - 0.5 * np.log(2 * np.pi) - log_ndtr(-z))
        precision = np.where(censored, mills * (mills - z) / variance, 1 / variance)
        self.posterior_inverse = np.linalg.inv(self.covariance + np.diag(1 / np.maximum(precision, 1e-6)))
        self.coefficients = inverse @ (self.latent - self.prior)
        return self

    def predict(self, features, fidelities):
        features = np.atleast_2d(features)
        prior = np.log(2.0) * np.asarray(fidelities)
        if not len(self.x):
            return prior, np.full(len(features), np.sqrt(1 + self.noise ** 2))
        cross = rbf(features, self.x, 0.6)
        mean = prior + cross @ self.coefficients
        variance = 1 - np.sum(cross * (cross @ self.posterior_inverse), axis=1) + self.noise ** 2
        return mean, np.sqrt(np.maximum(variance, 1e-6))

    def samples(self, features, fidelities, count, rng, age=None):
        mean, std = self.predict(features, fidelities)
        lower = np.zeros(len(mean)) if age is None else norm.cdf((np.log(np.maximum(age, 1e-12)) - mean) / std)
        uniform = lower[:, None] + (1 - lower[:, None]) * rng.uniform(1e-9, 1 - 1e-9, (len(mean), count))
        return np.exp(np.clip(mean[:, None] + std[:, None] * norm.ppf(np.clip(uniform, 1e-12, 1 - 1e-12)), -20, 30))


class FailureModel:
    def fit(self, features, failed):
        self.constant = (np.sum(failed) + 1) / (len(failed) + 2)
        self.model = None
        if len(np.unique(failed)) == 2:
            self.model = GaussianProcessClassifier(RBF(0.5), optimizer=None).fit(features, failed)
        return self

    def predict(self, features):
        return np.full(len(features), self.constant) if self.model is None else self.model.predict_proba(features)[:, 1]
