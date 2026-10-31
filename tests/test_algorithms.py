"""Algorithm identities and engineering-response replay, not score fabrication."""
import contextlib
import io
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from waf_mfbo.baselines import EnvironmentGP, Halving, Selector, density_ratio, mf_mes
from waf_mfbo.benchmarks import generate, response
from waf_mfbo.cli import main


class Independent:
    target = 1

    def posterior(self, points):
        return np.zeros(len(points)), np.eye(len(points))


class AlgorithmChecks(unittest.TestCase):
    def test_entropy_and_environment_kernel(self):
        self.assertAlmostEqual(mf_mes(Independent(), np.array([.5, 1]), [0.]), np.log(2))
        self.assertAlmostEqual(mf_mes(Independent(), np.array([.5, 0]), [-1., 0.]), 0., places=10)
        points = np.array([[.5, 0.], [.5, 2.]])
        k = EnvironmentGP(2).kernel(points, points)
        self.assertAlmostEqual(k[0, 1], 1.)
        self.assertAlmostEqual(k[0, 0], 1 + .75 ** 4)
        self.assertAlmostEqual(EnvironmentGP(2, cost=True).kernel(points, points)[0, 1], 1.25)

    def test_hyperband_promotes_only_observed_best(self):
        points = np.array([[i / 8, f] for i in range(9) for f in range(3)])
        hb = Halving(points)
        rng = np.random.default_rng(42)
        candidates = list(range(len(points)))
        launched = []
        for _ in range(9):
            task = hb.select(np.array(candidates), [], [], rng)
            self.assertEqual(points[task, -1], 0)
            launched.append(task)
            candidates.remove(task)
        self.assertIsNone(hb.select(np.array(candidates), [], [], rng))
        mid = []
        values = [points[i, 0] for i in launched]
        for _ in range(3):
            task = hb.select(np.array(candidates), launched, values, rng)
            mid.append(task)
            candidates.remove(task)
        self.assertEqual(set(mid), {1, 4, 7})
        self.assertIsNone(hb.select(np.array(candidates), launched, values, rng))
        task = hb.select(np.array(candidates), launched + mid, values + [points[i, 0] for i in mid], rng)
        self.assertEqual(task, 2)

    def test_bohb_density_and_robust_gate(self):
        x = np.arange(10)[:, None] / 10
        score = density_ratio(x, x[:, 0], np.array([[.05], [.95]]), np.random.default_rng(3), random_fraction=0.)
        self.assertGreater(score[0], score[1])
        points = np.array([[x, f] for x in [0., .5, 1.] for f in range(3)])
        args = SimpleNamespace(samples=8, fantasies=4, candidates=20, violation_tolerance=.05,
                               robust_variance=0., robust_information=0.)
        selector = Selector(points, 'robust-mfbo', args)
        observed = [2, 5]
        chosen = selector.select(np.array([i for i in range(9) if i not in observed]), observed,
                                 [1., .5], [1., 2.], [False, False], np.empty((0, 2)), np.empty((2, 0)), np.random.default_rng(4))
        self.assertEqual(points[chosen, -1], 2)
        self.assertFalse(selector.diagnostic['robust_multifidelity_gate'])

    def test_canonical_engineering_functions(self):
        self.assertAlmostEqual(response('forrester', [.7572487578])[0], -6.0207400558, places=6)
        self.assertAlmostEqual(response('branin', [(-np.pi + 5) / 15, 12.275 / 15])[0], .3978873577, places=7)
        value, constraints, solved = response('sellar', [.5, .2, .1])
        self.assertTrue(solved)
        self.assertAlmostEqual(value, 28.588308165, places=6)
        self.assertAlmostEqual(3.16 - constraints[0], 25.588302369, places=6)
        self.assertTrue(np.isfinite(response('park', [0., .5, .5, .5])[0]))

    def test_all_controls_on_actual_analytic_function_pool(self):
        out = Path('runs/algorithm-check')
        for problem in ['forrester', 'branin', 'park', 'sellar']:
            generate(problem, 6, 42, out / problem, tail='deterministic', failure_rate=0.)
        sys.argv = ['waf_mfbo', 'prepare', '--csv', str(out / 'forrester/replay.csv'), '--output', str(out / 'prepared')]
        main()
        initial = None
        for method in ['full', 'mfmes', 'async-mfbo', 'robust-mfbo', 'fabolas', 'hyperband', 'bohb']:
            sys.argv = ['waf_mfbo', 'run', '--data', str(out / 'prepared'), '--output', str(out / method),
                        '--method', method, '--initial', '3', '--workers', '2', '--deadline', '20',
                        '--samples', '4', '--fantasies', '3', '--candidates', '4', '--seed', '42', '--no-net-time']
            with contextlib.redirect_stdout(io.StringIO()):
                main()
            launches = json.loads((out / method / 'launches.json').read_text())
            first = [row['task'] for row in launches[:3]]
            if initial is None:
                initial = first
            self.assertEqual(first, initial)
            self.assertTrue(all(row['start'] < 20 for row in launches))
            metrics = json.loads((out / method / 'metrics.json').read_text())
            self.assertLessEqual(metrics['worker_occupancy'], 1. + 1e-8)
            self.assertIsNotNone(metrics['best_target_objective'])

    def test_actual_dispatch_overhead_is_charged(self):
        out = Path('runs/algorithm-check/net')
        generate('forrester', 3, 7, out, tail='deterministic', failure_rate=0.)
        sys.argv = ['waf_mfbo', 'prepare', '--csv', str(out / 'replay.csv'), '--output', str(out / 'prepared')]
        main()
        sys.argv = ['waf_mfbo', 'run', '--data', str(out / 'prepared'), '--output', str(out / 'run'),
                    '--method', 'random', '--initial', '1', '--workers', '1', '--deadline', '.2', '--seed', '7']
        main()
        metrics = json.loads((out / 'run/metrics.json').read_text())
        self.assertGreater(metrics['optimizer_seconds'], 0)
        launches = json.loads((out / 'run/launches.json').read_text())
        if launches:
            self.assertGreater(launches[0]['start'], 0)
            self.assertTrue(all(r['start'] < .2 for r in launches))


if __name__ == '__main__':
    unittest.main()
