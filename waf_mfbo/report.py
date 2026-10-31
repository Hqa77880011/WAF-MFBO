"""Paired problem/seed aggregates from actually completed common-budget replays."""
import argparse
import json
from pathlib import Path

import numpy as np


def aggregate(manifest, output):
    """Keep no-incumbent runs visible; never drop them to improve average regret."""
    source = Path(manifest)
    cases = json.loads(source.read_text(encoding='utf-8'))
    rows, shared, seen = [], {}, set()
    for case in cases:
        key = (case['problem'], case['seed'], case['method'])
        if key in seen:
            raise ValueError(f'Duplicate problem/seed/method {key}')
        seen.add(key)
        root = source.parent / case['run']
        settings = json.loads((root / 'settings.json').read_text(encoding='utf-8'))
        if settings['seed'] != case['seed'] or settings['method'] != case['method']:
            raise ValueError('Manifest seed/method disagrees with actual run settings')
        contract = {k: settings[k] for k in ['deadline', 'workers', 'initial', 'net_time']}
        paired = (case['problem'], case['seed'])
        if paired in shared and shared[paired] != contract:
            raise ValueError('Paired methods have different worker/deadline/initial/time accounting')
        shared[paired] = contract
        metrics = json.loads((root / 'metrics.json').read_text(encoding='utf-8'))
        rows.append(dict(problem=case['problem'], seed=case['seed'], method=case['method'], **metrics))
    summaries = []
    for problem, method in sorted({(r['problem'], r['method']) for r in rows}):
        group = [r for r in rows if (r['problem'], r['method']) == (problem, method)]
        regrets = [r['normalized_regret'] for r in group if r['normalized_regret'] is not None]
        complete = len(regrets) == len(group)
        summaries.append(dict(problem=problem, method=method, seeds=[r['seed'] for r in group], runs=len(group),
                              feasible_incumbent_rate=float(np.mean([r['feasible_incumbent'] for r in group])),
                              primary_regret_mean=float(np.mean(regrets)) if complete else None,
                              primary_regret_sample_sd=float(np.std(regrets, ddof=1)) if complete and len(regrets) > 1 else None,
                              conditional_regret_mean=float(np.mean(regrets)) if regrets else None,
                              failed_time_fraction=float(np.mean([r['failed_worker_time_fraction'] for r in group])),
                              worker_occupancy=float(np.mean([r['worker_occupancy'] for r in group])),
                              optimizer_seconds=float(np.mean([r['optimizer_seconds'] for r in group]))))
    macro = []
    for method in sorted({r['method'] for r in summaries}):
        group = [r for r in summaries if r['method'] == method]
        values = [r['primary_regret_mean'] for r in group]
        macro.append(dict(method=method, problems=[r['problem'] for r in group],
                          benchmark_macro_regret=float(np.mean(values)) if all(v is not None for v in values) else None,
                          benchmark_sample_sd=float(np.std(values, ddof=1)) if len(values) > 1 and all(v is not None for v in values) else None))
    Path(output).write_text(json.dumps(dict(problem_seed_runs=rows, per_problem=summaries, macro=macro), indent=2), encoding='utf-8')
    return summaries


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True, help='JSON entries: problem, seed, method, run relative to this manifest')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    aggregate(args.manifest, args.output)


if __name__ == '__main__':
    main()
