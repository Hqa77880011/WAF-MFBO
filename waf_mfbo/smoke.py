"""Exercise optimization with generated replay data."""
import argparse
import csv
import json
import pickle
import sys
from pathlib import Path
import numpy as np
from .cli import main as cli
from .models import MultiFidelityGP, RuntimeModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='runs/smoke')
    out = Path(p.parse_args().output)
    out.mkdir(parents=True, exist_ok=True)
    x = np.array([[0., 0.], [.5, 1.], [1., 1.]])
    model = MultiFidelityGP(1).fit(x, [1., .5, 0.])
    mean, covariance = model.posterior(x)
    assert np.max(abs(mean - np.array([1., .5, 0.]))) < .001
    assert np.linalg.eigvalsh(covariance).min() > -1e-9
    np.testing.assert_allclose(mean, pickle.loads(pickle.dumps(model)).posterior(x)[0])
    runtime = RuntimeModel().fit(x, [1., 2., 3.], [False, True, False], [0, 1, 1])
    samples = runtime.samples(x, [0, 1, 1], 16, np.random.default_rng(42), np.array([.5, 1., 2.]))
    assert np.isfinite(samples).all() and np.all(samples >= np.array([.5, 1., 2.])[:, None])
    with (out / 'source.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['task_id', 'x0', 'fidelity', 'objective', 'runtime_seconds', 'failed'])
        for i in range(12):
            w.writerow([f't{i}', i / 11, i % 2, '' if i == 3 else (i / 11 - .4) ** 2, .03 + .01 * (i % 3), int(i == 3)])
    sys.argv = ['waf_mfbo', 'prepare', '--csv', str(out / 'source.csv'), '--output', str(out / 'data')]
    cli()
    for method in ['full', 'information', 'random', 'no-runtime', 'no-pending']:
        sys.argv = ['waf_mfbo', 'run', '--data', str(out / 'data'), '--output', str(out / method),
                    '--method', method if not method.startswith('no-') else 'full', '--deadline', '1',
                    '--workers', '2', '--initial', '3', '--samples', '4', '--fantasies', '4',
                    '--candidates', '4', '--seed', '42', '--no-net-time']
        if method.startswith('no-'):
            sys.argv += ['--' + method]
        cli()
        m = json.loads((out / method / 'metrics.json').read_text())
        assert 0 <= m['worker_occupancy'] <= 1 + 1e-8
        assert all(e['time'] <= 1 for e in json.loads((out / method / 'events.json').read_text()))
        if method == 'random':
            assert m['simple_regret'] == 0
    print('GP fit/checkpoint, censored runtime, five policies and zero-regret example passed; seed=42')


if __name__ == '__main__':
    main()
