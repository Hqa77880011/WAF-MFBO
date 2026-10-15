# WAF-MFBO

Implementation of **WAF-MFBO: Failure- and Runtime-Aware Asynchronous Multi-Fidelity Optimization under Wall-Clock Budgets**. A target/discrepancy GP, right-censored runtime model, failure posterior and joint pending-job fantasies assign wall-clock value to candidate evaluations.

## Install and prepare replay observations

Use Python 3.10+ and activate a virtual environment:

```bash
python -m venv .venv
python -m pip install -e .
waf_mfbo prepare --csv data/solver_replay.csv --output data/prepared
waf_mfbo run --data data/prepared --deadline 300 --workers 4 --output runs/full
```

The replay CSV contains measured solver observations or a declared benchmark generator's outcomes. It uses these columns:

| Column | Meaning |
| --- | --- |
| `task_id` | Unique task/configuration identity |
| `x0,x1,...` | Numeric design coordinates |
| `fidelity` | Nonnegative source index; the largest index is the target |
| `objective` | Minimized objective, blank on failure or censoring |
| `runtime_seconds` | Observed duration or censoring lower bound |
| `failed` | `1` for a terminated failure, otherwise `0` |
| `censored` | Optional `1` when duration is only a right-censoring bound |
| `recovery_seconds` | Optional worker recovery time after failure |
| `g0,g1,...` | Optional continuous constraints, feasible when every value is ≤0 |

Provide the same finite task pool for every compared policy. Include successful target-fidelity reference observations so replay regret has a defined oracle. Failed and censored tasks never contribute invented objective values. The scheduler knows the available task designs/fidelities, but reads response, failure and duration observations only at their simulated completion events. Ground-truth replay outcomes are consulted after the deadline for regret accounting.

To evaluate Forrester, biased Branin, Park or Sellar workloads, export their actual source observations and pilot-fitted operational traces into this schema. Keep benchmark source definitions, runtime-tail and failure-generation settings fixed across paired methods. A task row represents one solver configuration; several configurations for one design/fidelity can support value-based retry comparisons without an automatic retry rule.

## Models and launch decisions

The response model has covariance `k_target(x,x') + 1[same auxiliary source]*k_discrepancy(x,x')`. Successful observations update the shared target and auxiliary discrepancies. The source reliability factor uses the posterior discrepancy mean/variance and `--source-scale`, expressed in objective units.

Runtime uses a log-time latent GP with Gaussian exact-duration likelihood and survival likelihood for right-censored observations. MAP fitting and a diagonal Laplace curvature approximation provide lognormal predictive samples. A running task contributes its elapsed lower bound only after it actually starts. Failure has a separate Bernoulli GP, with a Beta-smoothed constant probability until both outcomes are observed. A failure duration is exact unless its row explicitly marks censoring.

Pending and candidate outcomes are drawn jointly from the response posterior. For each fantasy, sampled successful/deadline-available pending observations update the decision baseline before the candidate's additional reduction is valued. Gain is one-step reduction in the posterior-minimum decision criterion over the finite target-design grid. This is a concrete knowledge-gradient approximation of the manuscript's decision-risk reduction.

Runtime samples determine deadline availability and incremental queue makespan. Failure recovery is an additional time cost. Candidate launch requires predicted constraint-violation probability below `--violation-tolerance`; initial pilot tasks form the declared common initialization before the learned constraint gate. Unknown feasibility can therefore stop subsequent launching. No usable target incumbent is represented by `null`, not an artificial objective penalty.

## Comparisons and parameters

```bash
waf_mfbo run --method random --output runs/random
waf_mfbo run --method information --output runs/information
waf_mfbo run --no-pending --output runs/no_pending
waf_mfbo run --no-reliability --output runs/no_reliability
waf_mfbo run --no-failure --output runs/no_failure
waf_mfbo run --queue-delay 0 --output runs/immediate_only
waf_mfbo run --no-net-time --output runs/scheduling_only
```

Random allocation selects uniformly from unused tasks. Information-only selection retains the same joint response fantasies while dropping source/deadline/cost discounts. Other flags remove individual WAF components. They are named for their implemented policies, not for external optimization packages.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--initial` | 8 | Seeded common pilot tasks |
| `--candidates` | 32 | Maximum sampled unused tasks per acquisition |
| `--samples` | 32 | Runtime Monte Carlo samples |
| `--fantasies` | 16 | Joint information-gain fantasies |
| `--queue-delay` | 1 s | Alternative reserved delayed start |
| `--recovery-prior` | 1 s | Recovery estimate before observing failures |
| `--source-scale` | 1 | Auxiliary discrepancy reliability scale |
| `--stop` | 1e-8 | Minimum launch value |
| `--violation-tolerance` | 0.05 | Candidate constraint-risk limit |
| `--seed` | 42 | Initial design, candidate and fantasy seed |

The default deadline is 300 seconds with four workers. `--net-time` is enabled: measured fitting and acquisition computation consume replay time while already running jobs continue. `--no-net-time` exposes the scheduling-only abstraction. All policies should use the same accounting setting. A reserved delayed start occupies its worker before execution; the delay itself is not counted as useful worker execution.

## Outputs

`metrics.json` includes target simple/normalized regret, feasible-incumbent status, completion and launch counts, failure fraction, failed worker-time fraction, worker occupancy, unfinished exposure, runtime quantile coverage and optimizer overhead. Run matched seeds and average feasible-incumbent indicators to obtain an incumbent rate. Runtime quantile coverage uses exact completed durations and records 0.1/0.5/0.9 forecasts made before launch.

`events.json` retains completion times and the target incumbent available at each event. `pending.json` records unfinished tasks as elapsed lower bounds at the hard deadline. `incumbent.png` plots only completed target observations. A successful low-fidelity result cannot become the final target incumbent. Normalized regret uses the successful feasible target replay range; use this same pool for all methods.

## Numerical conventions

The response kernel has fixed unit-domain length scale 0.3 and independent unit target/discrepancy amplitudes in the current response-standardized units. Runtime uses length scale 0.6 and log-time likelihood noise 0.45. Its initial median prior is `2**fidelity` seconds and is updated from operational observations. Finite candidate replay replaces continuous Sobol/trust-region search, and the objective reference grid is the supplied design pool. A canceled censored row frees its worker at the recorded bound but contributes only a runtime survival observation. These conventions expose the method's information, availability and redundancy components without importing future operational outcomes into their posteriors.
