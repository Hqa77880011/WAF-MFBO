# WAF-MFBO

Implementation of **WAF-MFBO: Failure- and Runtime-Aware Asynchronous Multi-Fidelity Optimization under Wall-Clock Budgets**.

## Installation and input

Python 3.10+; `python -m pip install -e .`.
Use measured solver replay observations or outcomes from a declared benchmark generator. CSV columns: unique `task_id`, numeric `x0,x1,...`, nonnegative integer `fidelity`, `objective`, positive `runtime_seconds`, and binary `failed`. Optional fields are `censored`, `recovery_seconds`, and constraints `g0,g1,...` (feasible at <=0). Objective is blank on failure/censoring. A censored duration is a lower bound; a terminated failure with measured termination time is exact unless marked censored. Include successful target-fidelity references and reuse the same finite task pool across policies.

```bash
waf_mfbo prepare --csv data/solver_replay.csv --output data/prepared
waf_mfbo run --data data/prepared --deadline 300 --workers 4 --output runs/full --seed 42
```

`prepare` scales design coordinates using the unlabeled finite pool bounds. Search sees design/fidelity coordinates in advance; objective, constraint, runtime and failure outcomes are released only at their completion event. Pending jobs contribute elapsed censoring bounds. Only completed feasible target observations can become the final incumbent.

## Method, metrics and controls

| Paper component | Code entry | Command / convention |
| --- | --- | --- |
| Shared target plus auxiliary discrepancy GP | `models.MultiFidelityGP` | target is maximum supplied fidelity |
| Log-time survival likelihood and failure GP | `RuntimeModel`, `FailureModel` | MAP plus diagonal Laplace; smoothed failure rate before two classes |
| Joint pending fantasies and deadline acquisition | `cli.information_gain`, `optimize` | `run`, fantasies 16 / runtime draws 32 |
| Deadline regret, feasibility, failure exposure, occupancy, quantile calibration | `optimize` | written by `run` to `metrics.json` |
| Incumbent trajectory | `optimize` | `events.json`, `incumbent.png` |

`simple_regret` is the completed feasible target incumbent minus the best feasible target in the fixed replay pool. Normalized regret divides by that pool's feasible target range (floor 1e-8). Worker occupancy counts execution clipped at the deadline; failed-time fraction includes completed failed execution and observed recovery clipped at the deadline. Unfinished results do not enter the incumbent. Optimizer fitting/acquisition time is charged by default; final file serialization/plotting is outside the decision budget.

Methods are `full`, `random` and `information`. Ablations: `--no-pending` removes pending information fantasies while retaining physical pending runtimes in makespan exposure; `--no-reliability`, `--no-failure`, `--no-runtime`, `--queue-delay 0`, and `--no-net-time`. The no-runtime control uses the declared unfitted log-time prior with median `2**fidelity` seconds. Keep pool, initial count, workers, deadline, seed and time convention matched.

```bash
waf_mfbo run --data data/prepared --deadline 300 --workers 4 --output runs/no_pending --no-pending --seed 42
waf_mfbo run --data data/prepared --deadline 300 --workers 4 --output runs/no_runtime --no-runtime --seed 42
```

Each run writes metrics, completion events, deadline-pending records and a plot. No separate evaluation command is needed. Use separate folders for seeds and settings.

## Implementation scope and checks

The existing finite-pool implementation approximates decision-risk reduction by a knowledge gradient over a feasible target grid, and uses a user-specified source scale (default 1), fixed RBF kernels and immediate/delayed starts. It is not a reproduction of the unavailable historical benchmark runner. FABOLAS, MF-MES, robust MFBO, Async-MFBO, Hyperband and BOHB adapters, pilot logs, and the paper's full four-benchmark generators are not supplied here. Supply actual paired replay CSVs to evaluate this implementation.

```bash
python -m waf_mfbo.smoke --output runs/smoke
```

The seed-42 generated check verifies GP fitting and positive posterior covariance, model save/load consistency, survival-conditioned runtime draws, CSV preparation, three policies plus no-runtime/no-pending, clipped occupancy/deadline events, and an exactly zero-regret enumerated example. Generated data verifies functionality only. Reference environment: Python 3.13.5, NumPy 2.1.3, SciPy 1.15.3, scikit-learn 1.6.1, CPU with one BLAS thread. Full benchmark runs and paper performance reproduction are not performed.
