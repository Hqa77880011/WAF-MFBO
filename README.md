# WAF-MFBO

Implementation of **WAF-MFBO: Failure- and Runtime-Aware Asynchronous Multi-Fidelity Optimization under Wall-Clock Budgets**.

## Installation

Requires Python >=3.10.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5, NumPy 2.1.3, SciPy 1.15.3 and scikit-learn 1.6.1; neural entries use the installed PyTorch 2.7.1+cu118. Checks cover Python syntax, TOML/JSON and CLI options against parser definitions. Training and evaluation were not run.

## Data and preprocessing

Use measured solver replay observations or outcomes from a declared benchmark generator. CSV columns: `task_id`, numeric `x0,x1,...`, `fidelity`, `objective`, `runtime_seconds`, `failed`; optional `censored`, `recovery_seconds` and constraints `g0,g1,...` (feasible at <=0). Objective is blank on failure/censoring; duration is a censoring lower bound when marked. Include successful target-fidelity references and reuse one finite task pool across policies.

```bash
waf_mfbo prepare --csv data/solver_replay.csv --output data/prepared
```

## Run

```bash
waf_mfbo run --data data/prepared --deadline 300 --workers 4 --output runs/full --seed 42
```

## Ablations

Methods: `full`, `random`, `information`. Component controls: `--no-pending`, `--no-reliability`, `--no-failure`, `--queue-delay 0`, `--no-net-time`. Keep task pool, deadline, workers, seed and computation-time accounting matched.

```bash
waf_mfbo run --data data/prepared --deadline 300 --workers 4 --output runs/no_pending --no-pending --seed 42
```

## Outputs

Each run writes `metrics.json` (target regret, feasibility and runtime accounting), `events.json` (completions and target incumbent), `pending.json` (unfinished tasks) and `incumbent.png`. There is no separate evaluation command; replay outcomes are scored by `run`.
