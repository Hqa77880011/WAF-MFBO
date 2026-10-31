"""Explicit engineering benchmark definitions and replayable operational processes."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.special import expit, logit
from scipy.stats import qmc
from sklearn.linear_model import LinearRegression, LogisticRegression


DIMENSIONS = dict(forrester=1, branin=2, park=4, sellar=3)


def response(problem, normalized, fidelity=2):
    """Canonical high source; explicitly declared intermediate/biased engineering sources."""
    x = np.asarray(normalized, dtype=float)
    if problem == 'forrester':
        value = (6 * x[0] - 2) ** 2 * np.sin(12 * x[0] - 4)
        if fidelity == 0:
            value = .5 * value + 10 * (x[0] - .5) - 5
        elif fidelity == 1:
            value = .8 * value + 3 * (x[0] - .5) - 1
        return float(value), [], True
    if problem == 'branin':
        a, b = -5 + 15 * x[0], 15 * x[1]
        value = (b - 5.1 * a ** 2 / (4 * np.pi ** 2) + 5 * a / np.pi - 6) ** 2 + 10 * (1 - 1 / (8 * np.pi)) * np.cos(a) + 10
        if fidelity == 0:
            value = .6 * value + 5 * np.sin(2 * np.pi * x[0]) + 2 * (x[1] - .5)
        elif fidelity == 1:
            value = .85 * value + 2 * np.sin(np.pi * x[0]) - .5 * x[1]
        return float(value), [], True
    if problem == 'park':
        a, b, c, d = x
        high = .5 * (np.sqrt(a ** 2 + (b + c ** 2) * d) - a) + (a + 3 * d) * np.exp(1 + np.sin(c))
        low = (1 + np.sin(a) / 10) * high - 2 * a + b ** 2 + c ** 2 + .5
        return float([low, .75 * high + .25 * low, high][fidelity]), [], True
    if problem == 'sellar':
        z1, z2, local = 10 * x
        coupling = [.12, .16, .2][fidelity]
        iterations = [12, 20, 100][fidelity]
        y1, y2 = 1., 1.
        residual = float('inf')
        for _ in range(iterations):
            first = z1 ** 2 + z2 + local - coupling * y2
            second = np.sqrt(abs(first)) + z1 + z2
            residual = max(abs(first - y1), abs(second - y2))
            y1, y2 = first, second
            if residual < 1e-10:
                break
        return float(local ** 2 + z2 + y1 + np.exp(-y2)), [3.16 - y1, y2 - 24], bool(residual < 1e-7)
    raise ValueError('Unknown benchmark')


def fit_pilot(path, dimensions):
    """Fit declared runtime/failure processes using supplied pilot rows only."""
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    observed = [r for r in rows if not int(r.get('censored') or 0)]
    if len(observed) < 3:
        raise ValueError('Need at least three exact-termination pilot records')
    features = np.array([[*[float(r[f'x{i}']) for i in range(dimensions)], *np.eye(3)[int(r['fidelity'])]] for r in observed])
    times = np.log([float(r['runtime_seconds']) for r in observed])
    runtime = LinearRegression().fit(features, times)
    sigma = max(.05, float(np.std(times - runtime.predict(features))))
    labels = np.array([int(r['failed']) for r in observed])
    failure = LogisticRegression(max_iter=1000).fit(features, labels) if len(np.unique(labels)) > 1 else None
    constant = float((labels.sum() + 1) / (len(labels) + 2))
    return runtime, sigma, failure, constant


def generate(problem, designs, seed, output, tail='lognormal', failure_rate=.1, sigma=.35, pilot=None):
    """Evaluate all three sources on one nested design pool; no future outcomes enter selection."""
    dim = DIMENSIONS[problem]
    x = qmc.LatinHypercube(dim, seed=seed).random(designs)
    rng = np.random.default_rng(seed + 100000)
    fitted = None if pilot is None else fit_pilot(pilot, dim)
    records = []
    for i, point in enumerate(x):
        for fidelity in range(3):
            y, constraints, solved = response(problem, point, fidelity)
            features = np.r_[point, np.eye(3)[fidelity]][None]
            if fitted is None:
                median = 2. ** fidelity * np.exp(.5 * point.mean())
                probability = 0. if failure_rate == 0 else float(expit(logit(failure_rate) + 3 * (point[0] - .5)))
                deviation = sigma
            else:
                runtime, deviation, failure, constant = fitted
                median = float(np.exp(runtime.predict(features)[0]))
                probability = float(failure.predict_proba(features)[0, 1]) if failure is not None else constant
            if tail == 'lognormal':
                duration = median * np.exp(deviation * rng.normal())
            elif tail == 'pareto':
                duration = median * (rng.pareto(1.8) + 1) / 2 ** (1 / 1.8)
            else:
                duration = median
            failed = not solved or rng.random() < probability
            record = dict(task_id=f'{problem}_{i}_f{fidelity}', **{f'x{k}': float(v) for k, v in enumerate(point)},
                          fidelity=fidelity, objective='' if failed else y, runtime_seconds=float(duration),
                          failed=int(failed), censored=0, recovery_seconds=float(.2 * median if failed else 0))
            for k, g in enumerate(constraints):
                record[f'g{k}'] = '' if failed else float(g)
            records.append(record)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'replay.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    metadata = dict(problem=problem, designs=designs, sources=3, seed=seed, runtime_tail=tail,
                    operational_source='declared_generated_process' if fitted is None else 'fitted_supplied_pilot',
                    failure_rate=None if fitted is not None else failure_rate, log_sigma=sigma if fitted is None else fitted[1],
                    fidelity_convention='0 low /1 explicitly constructed intermediate /2 canonical target; normalized coordinates')
    (out / 'definition.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--problem', choices=list(DIMENSIONS), required=True)
    p.add_argument('--designs', type=int, default=128)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--tail', choices=['lognormal', 'pareto', 'deterministic'], default='lognormal')
    p.add_argument('--failure-rate', type=float, default=.1)
    p.add_argument('--sigma', type=float, default=.35)
    p.add_argument('--pilot', help='Pilot runtime/failure CSV with normalized x coordinates; never generated test outcomes')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    generate(args.problem, args.designs, args.seed, args.output, args.tail, args.failure_rate, args.sigma, args.pilot)


if __name__ == '__main__':
    main()
