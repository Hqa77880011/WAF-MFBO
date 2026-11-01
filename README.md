# WAF-MFBO

**Failure- and Runtime-Aware Asynchronous Multi-Fidelity Optimization under Wall-Clock Budgets** using a shared finite-pool asynchronous engine.

## Install and input

Python3.10+: `python -m pip install -e .`.
CSV fields: unique `task_id`, numeric `x0,x1,...`, nonnegative integer `fidelity`, `objective`, positive `runtime_seconds`, binary `failed`; optional `censored`, `recovery_seconds`, and constraints `g0,g1,...` (feasible<=0). Failed/censored objectives are blank. Supplied censored rows must represent a terminated timeout whose duration is a lower bound on successful runtime; still-running jobs are represented internally as pending, not prematurely released from a replay row. Supply at least one successful feasible target-fidelity reference.

```bash
waf_mfbo prepare --csv data/solver_replay.csv --output data/prepared
waf_mfbo run --data data/prepared --deadline 300 --workers 4 --method full --output runs/full --seed 42
```

Coordinates/fidelities in the task pool are known before search. Bounds normalization uses only these unlabeled coordinates. Objective, runtime, constraint and failure outcomes are released at completion events; pending elapsed time supplies a censoring bound. Only completed feasible target observations enter the incumbent. `launches.json` records actual decisions and available-observation counts.

## Distinct comparison mechanisms

| Method (`--method`) | Entry / actual mechanism | Primary definition |
| --- | --- | --- |
| full | `cli.optimize`: discrepancy GP, survival runtime, failure GP, joint pending fantasies, deadline-success value / incremental makespan+recovery | This paper |
| mfmes | `baselines.mf_mes`: Gaussian conditional truncated entropy; posterior target minima; Gauss-Hermite auxiliary-source integral / expected duration | [Takeno et al.](https://proceedings.mlr.press/v119/takeno20a.html) |
| async-mfbo | Same information-per-expected-time valuation plus soft busy-location Lipschitz penalization, without WAF failure/deadline factors | [Folch implementation](https://github.com/jpfolch/MFBoom) |
| robust-mfbo | Distinct pseudo single-fidelity trajectory; variance and information/cost gates; nearest-real-target choice refreshes pseudo outcomes; actual SF fallback | [rMFBO](https://github.com/AaltoPML/rMFBO) |
| fabolas | Product spatial/environment GP basis [1,(1-s)^2] and log-cost [1,s]; nested fantasy entropy of the target minimizer / expected time | [FABOLAS](https://github.com/automl/RoBO) |
| hyperband | Rotating resource brackets and observed top1/3 promotion; workers wait for cohort results | [Hyperband](https://arxiv.org/abs/1603.06560) |
| bohb | Same brackets, with highest-supported-resource good/bad Gaussian KDE proposal ranking and1/3 random exploration | [BOHB](https://proceedings.mlr.press/v80/falkner18a.html) |
| information / random | Unweighted conditional knowledge-gradient / random allocation | Explicit mechanism controls |

All methods share the initial random task permutation, finite pool, workers, deadline, feasibility rule and completion/recovery events. `--net-time` (default) charges actual model/acquisition computation, dispatch/scheduling work and the JSON serialization used for the launch ledger. Final report/plot file writes are outside the decision loop. `--no-net-time` is a declared overhead-excluded comparison.

RBF hyperparameters are fixed. FABOLAS maps fidelity to s=2**(f-target), models signed objectives and uses a finite minimizer grid. rMFBO defaults to `--robust-variance .1 --robust-information .001`; Async-MFBO uses information-based fidelity selection and a finite-grid Lipschitz estimate. Hyperband/BOHB require matched configurations across fidelities.

```bash
waf_mfbo run --data data/prepared --deadline 300 --workers 4 --method mfmes --output runs/mfmes --seed 42
waf_mfbo run --data data/prepared --deadline 300 --workers 4 --method robust-mfbo --output runs/robust --seed 42
waf_mfbo run --data data/prepared --deadline 300 --workers 4 --method bohb --output runs/bohb --seed 42
```

Full-method component controls: `--no-pending` removes pending information while retaining physical pending makespan; `--no-reliability`, `--no-failure`, `--no-runtime`, `--queue-delay 0`, `--no-net-time`. The no-runtime control uses the explicit prior median2**f seconds. Default seed42, fantasy16, runtime samples32 and candidate cap32 are saved in `settings.json`.

## Engineering functions and operational processes

```bash
python -m waf_mfbo.benchmarks --problem forrester --designs 128 --tail lognormal --failure-rate .1 --seed 42 --output data/forrester
waf_mfbo prepare --csv data/forrester/replay.csv --output data/forrester/prepared
```

`--problem` supports Forrester, Branin, Park and coupled Sellar, with three nested sources. Targets follow the [Forrester/Park library](https://www.sfu.ca/~ssurjano/optimization.html) and [OpenMDAO Sellar equations](https://openmdao.org/newdocs/versions/latest/basic_user_guide/multidisciplinary_optimization/sellar.html). Source formulas and operational settings are recorded in `definition.json`; Sellar nonconvergence is recorded as failure.

`--tail lognormal|pareto|deterministic` selects a generated completion process; Pareto is median-matched. `--pilot pilot.csv` fits log-runtime/failure models from exact-termination records using normalized coordinates, excluding censored records.

## Metrics and paired reporting

`metrics.json` contains target simple regret, pool-range-normalized regret, feasible-incumbent status, failed-time exposure including clipped observed recovery, clipped execution occupancy, unfinished jobs, runtime-quantile calibration and optimizer seconds. Failed/pending results never become objective penalties or incumbents. `events.json`, `pending.json`, `launches.json`, `settings.json` and `incumbent.png` retain decisions/outcomes.

```bash
python -m waf_mfbo.report --manifest comparison_runs.json --output comparison.json
```

Manifest entries are `{ "problem":"forrester", "seed":42, "method":"full", "run":"runs/full" }`, with run paths relative to the manifest. Reporting verifies worker/deadline/initial/time-accounting parity, computes per-problem seed means/sampleSD and an equal-problem macro mean. Missing feasible incumbents are retained: primary mean regret is null if any run has no incumbent, conditional mean and feasibility rate are reported separately. No failed run is silently dropped.

## Checks

```bash
python -m unittest discover -s tests -v
python -m waf_mfbo.smoke
```
