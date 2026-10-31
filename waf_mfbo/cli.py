import argparse
import csv
import heapq
import json
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtr

from .models import FailureModel, MultiFidelityGP, RuntimeModel
from .baselines import METHODS, Selector


def prepare(args):
    with Path(args.csv).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or len({r['task_id'] for r in rows}) != len(rows):
        raise ValueError('Replay needs nonempty, unique task_id records')
    columns = sorted([k for k in rows[0] if k.startswith("x") and k[1:].isdigit()], key=lambda k: int(k[1:]))
    if not columns:
        raise ValueError("Design coordinates must be named x0, x1, ...")
    x = np.array([[float(r[c]) for c in columns] for r in rows])
    lower, upper = x.min(0), x.max(0)
    x = (x - lower) / np.maximum(upper - lower, 1e-8)
    fidelity = np.array([int(r["fidelity"]) for r in rows])
    failed = np.array([bool(int(r["failed"])) for r in rows])
    censored = np.array([bool(int(r.get("censored", 0))) for r in rows])
    y = np.array([float(r["objective"]) if r["objective"].strip() and not failed[i] and not censored[i] else np.nan for i, r in enumerate(rows)])
    runtime = np.array([float(r["runtime_seconds"]) for r in rows])
    recovery = np.array([float(r.get("recovery_seconds", 0)) for r in rows])
    constraint_columns = sorted([k for k in rows[0] if k.startswith("g") and k[1:].isdigit()], key=lambda k: int(k[1:]))
    constraints = np.array([[float(r[c]) if r[c].strip() else np.nan for c in constraint_columns] for r in rows])
    if np.any(runtime <= 0) or np.any(recovery < 0) or not np.isfinite(x).all():
        raise ValueError("Invalid design or operational duration")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "replay.npz", x=x, fidelity=fidelity, y=y, failed=failed, censored=censored,
                        runtime=runtime, recovery=recovery, lower=lower, upper=upper,
                        constraints=constraints, task=np.array([r["task_id"] for r in rows]))


def information_gain(model, grid, candidate, pending, pending_probability, fantasies, rng):
    observations = np.r_[pending, candidate[None]] if len(pending) else candidate[None]
    mean_grid, _ = model.posterior(grid)
    _, covariance = model.posterior(observations)
    covariance = (covariance + covariance.T) / 2 + np.eye(len(observations)) * 1e-6
    _, cross = model.posterior(grid, observations)
    draws = rng.multivariate_normal(np.zeros(len(observations)), covariance, size=fantasies)
    gains = []
    for draw in draws:
        available = np.flatnonzero(rng.random(len(pending)) < pending_probability)
        if len(available):
            part = covariance[np.ix_(available, available)]
            baseline = mean_grid + cross[:, available] @ np.linalg.solve(part, draw[available])
        else:
            baseline = mean_grid
        included = np.r_[available, len(observations) - 1]
        updated = mean_grid + cross[:, included] @ np.linalg.solve(covariance[np.ix_(included, included)], draw[included])
        gains.append(baseline.min() - updated.min())
    return max(0, float(np.mean(gains)))


def optimize(args):
    data = np.load(Path(args.data) / "replay.npz", allow_pickle=False)
    x, fidelity = data["x"], data["fidelity"]
    target = int(fidelity.max())
    locations = np.c_[x, fidelity]
    operational = np.c_[x, np.eye(target + 1)[fidelity]]
    grid_x = np.unique(x, axis=0)
    grid = np.c_[grid_x, np.full(len(grid_x), target)]
    rng = np.random.default_rng(args.seed)
    completed, used, pending, events = [], set(), {}, []
    launches, predicted_quantiles = {}, {}
    free = np.zeros(args.workers)
    clock, overhead, recovery_time = 0.0, 0.0, 0.0
    trace = []
    launch_log = []
    selector = Selector(locations, args.method, args) if args.method in METHODS else None
    initial = rng.permutation(len(x))[:args.initial].tolist()

    def charge_control(begin, already=0.):
        nonlocal clock, overhead
        extra = max(0., perf_counter() - begin - already)
        overhead += extra
        if args.net_time:
            clock = min(args.deadline, clock + extra)

    def process(until):
        while events and events[0][0] <= until:
            finish, worker, task = heapq.heappop(events)
            completed.append(task)
            pending.pop(task)
            if selector is not None:
                selector.completed(task, data['y'][task])
            free[worker] = finish + (data["recovery"][task] if data["failed"][task] else 0)
            eligible = [i for i in completed if fidelity[i] == target and np.isfinite(data["y"][i]) and np.all(data["constraints"][i] <= 0)]
            best = min((float(data["y"][i]) for i in eligible), default=None)
            trace.append({"time": float(finish), "task": str(data["task"][task]), "failed": bool(data["failed"][task]),
                          "censored": bool(data["censored"][task]), "incumbent": best})

    while clock < args.deadline:
        decision_begin, planning = perf_counter(), 0.
        process(clock)
        workers = np.flatnonzero(free <= clock)
        if not len(workers):
            future = [event[0] for event in events if event[0] > clock] + [float(v) for v in free if v > clock]
            charge_control(decision_begin)
            if not future:
                break
            clock = min(max(clock, min(future)), args.deadline)
            continue
        worker = int(workers[0])
        candidates = np.array([i for i in range(len(x)) if i not in used])
        if not len(candidates):
            charge_control(decision_begin)
            if not events:
                break
            clock = min(max(clock, events[0][0]), args.deadline)
            continue
        delay = 0.0
        quantile_forecast = np.exp(np.log(2.0) * fidelity[candidates[0]] + np.array([-1.28155, 0, 1.28155]))
        if initial:
            task = initial.pop(0)
            if task in used:
                continue
            quantile_forecast = np.exp(np.log(2.0) * fidelity[task] + np.array([-1.28155, 0, 1.28155]))
        elif args.method == "random":
            task = int(rng.choice(candidates))
            quantile_forecast = np.exp(np.log(2.0) * fidelity[task] + np.array([-1.28155, 0, 1.28155]))
        elif selector is not None:
            begin = perf_counter()
            task = selector.select(candidates, completed, data['y'][completed], data['runtime'][completed],
                                   data['censored'][completed], locations[list(pending)], data['constraints'][completed], rng)
            planning = perf_counter() - begin
            overhead += planning
            if args.net_time:
                clock = min(args.deadline, clock + planning)
            process(clock)
            if clock >= args.deadline:
                charge_control(decision_begin, planning)
                break
            if task is None:
                charge_control(decision_begin, planning)
                if events:
                    clock = min(max(clock, events[0][0]), args.deadline)
                    continue
                break
            quantile_forecast = np.asarray(selector.diagnostic.get('runtime_quantiles',
                                          np.exp(np.log(2.) * fidelity[task] + np.array([-1.28155, 0., 1.28155]))))
        else:
            begin = perf_counter()
            usable = [i for i in completed if np.isfinite(data["y"][i]) and not data["censored"][i]]
            objective = MultiFidelityGP(target).fit(locations[usable], data["y"][usable])
            constraint_models = []
            for c in range(data["constraints"].shape[1]):
                observed_constraint = [i for i in completed if np.isfinite(data["constraints"][i, c]) and not data["censored"][i]]
                constraint_models.append(MultiFidelityGP(target).fit(locations[observed_constraint], data["constraints"][observed_constraint, c]))
            grid_feasible = np.ones(len(grid), bool)
            for constraint_model in constraint_models:
                gm, gc = constraint_model.posterior(grid)
                grid_feasible &= ndtr(gm / np.sqrt(np.maximum(np.diag(gc), 1e-9))) <= args.violation_tolerance
            decision_grid = grid[grid_feasible]
            time_features, times, censored, fs = [], [], [], []
            for i in completed:
                time_features.append(operational[i])
                times.append(float(data["runtime"][i]))
                censored.append(bool(data["censored"][i]))
                fs.append(fidelity[i])
            for i, job in pending.items():
                age = clock - job["start"]
                if age > 0:
                    time_features.append(operational[i])
                    times.append(age)
                    censored.append(True)
                    fs.append(fidelity[i])
            if args.no_runtime:
                runtime = RuntimeModel().fit(np.empty((0, operational.shape[1])), [], [], [])
            else:
                runtime = RuntimeModel().fit(np.array(time_features).reshape(-1, operational.shape[1]), times, censored, fs)
            observed = [i for i in completed if not data["censored"][i]]
            failure = FailureModel().fit(operational[observed], data["failed"][observed])
            physical_pending = list(pending)
            pending_ids = physical_pending if not args.no_pending else []
            if physical_pending:
                ages = np.array([max(0, clock - pending[i]["start"]) for i in physical_pending])
                durations = runtime.samples(operational[physical_pending], fidelity[physical_pending], args.samples, rng, ages)
                remaining = np.maximum(durations - ages[:, None], 0)
                available = (remaining <= args.deadline - clock).mean(1) * (1 if args.no_failure else 1 - failure.predict(operational[physical_pending]))
                makespan = remaining.max(0)
                if args.no_pending:
                    available = np.empty(0)
            else:
                available, makespan = np.empty(0), np.zeros(args.samples)
            if len(candidates) > args.candidates:
                candidates = rng.choice(candidates, args.candidates, replace=False)
            best_score, task = -1.0, int(candidates[0])
            for i in candidates:
                if not len(decision_grid):
                    continue
                launchable = True
                for constraint_model in constraint_models:
                    gm, gc = constraint_model.posterior(locations[i:i + 1])
                    launchable &= ndtr(gm[0] / np.sqrt(max(float(gc[0, 0]), 1e-9))) <= args.violation_tolerance
                if not launchable:
                    continue
                gain = information_gain(objective, decision_grid, locations[i], locations[pending_ids], available, args.fantasies, rng)
                mu_delta, covariance_delta = objective.posterior(locations[i:i + 1], discrepancy=True)
                reliability = float(np.exp(-(mu_delta[0] ** 2 + max(float(covariance_delta[0, 0]), 0)) / (2 * args.source_scale ** 2))) if fidelity[i] != target and not args.no_reliability else 1.0
                p_fail = float(failure.predict(operational[i:i + 1])[0]) if not args.no_failure else 0.0
                duration = runtime.samples(operational[i:i + 1], fidelity[i:i + 1], args.samples, rng)[0]
                for start_delay in [0, args.queue_delay] if args.queue_delay else [0]:
                    probability = float(np.mean(duration + start_delay <= args.deadline - clock)) * (1 - p_fail)
                    exposure = np.mean(np.maximum(makespan, duration + start_delay) - makespan)
                    recovery = float(np.mean([data["recovery"][j] for j in completed if data["failed"][j]])) if any(data["failed"][j] for j in completed) else args.recovery_prior
                    if args.method == "information":
                        probability, exposure, reliability, p_fail = 1, 1, 1, 0
                    score = reliability * probability * gain / (exposure + p_fail * recovery + 1e-6)
                    if score > best_score:
                        best_score, task, delay = score, int(i), float(start_delay)
            log_mean, log_std = runtime.predict(operational[task:task + 1], fidelity[task:task + 1])
            quantile_forecast = np.exp(np.clip(log_mean[0] + log_std[0] * np.array([-1.28155, 0, 1.28155]), -20, 30))
            planning = perf_counter() - begin
            overhead += planning
            if args.net_time:
                clock = min(args.deadline, clock + planning)
            process(clock)
            if clock >= args.deadline or best_score <= args.stop:
                charge_control(decision_begin, planning)
                break
        charge_control(decision_begin, planning)
        process(clock)
        if clock >= args.deadline:
            break
        ledger_begin = perf_counter()
        start_time = clock + delay
        finish = start_time + float(data["runtime"][task])
        used.add(task)
        pending[task] = {"worker": worker, "start": start_time}
        launches[task] = start_time
        predicted_quantiles[task] = quantile_forecast
        free[worker] = finish
        heapq.heappush(events, (finish, worker, task))
        launch_log.append(json.dumps(dict(time=float(clock), start=float(start_time), worker=worker,
                               task=str(data['task'][task]), completed_observations=len(completed),
                               method=args.method, diagnostics=selector.diagnostic.copy() if selector is not None else {}), allow_nan=False))
        recovery_time += float(data["recovery"][task]) if data["failed"][task] else 0
        charge_control(ledger_begin)
    process(args.deadline)
    completed_target = [i for i in completed if fidelity[i] == target and np.isfinite(data["y"][i]) and not data["censored"][i] and np.all(data["constraints"][i] <= 0)]
    best = min((float(data["y"][i]) for i in completed_target), default=None)
    oracle = data["y"][(fidelity == target) & np.isfinite(data["y"]) & ~data["censored"] & np.all(data["constraints"] <= 0, axis=1)]
    if not len(oracle):
        raise ValueError("Replay has no successful target-fidelity reference")
    active = sum(max(0, args.deadline - max(job["start"], 0)) for job in pending.values())
    busy = sum(min(float(data["runtime"][i]), max(0, args.deadline - pending[i]["start"])) for i in pending)
    busy += sum(float(data["runtime"][i]) for i in completed)
    failure_waste = sum(float(data["runtime"][i]) + min(float(data["recovery"][i]), max(0, args.deadline - launches[i] - float(data["runtime"][i]))) for i in completed if data["failed"][i])
    exact_times = [i for i in completed if not data["censored"][i]]
    metrics = {"best_target_objective": best, "simple_regret": best - float(oracle.min()) if best is not None else None,
               "normalized_regret": (best - float(oracle.min())) / max(float(np.ptp(oracle)), 1e-8) if best is not None else None,
               "completed": len(completed), "target_completed": len(completed_target), "launched": len(used),
               "failure_fraction": float(np.mean(data["failed"][completed])) if completed else None,
               "failed_worker_time_fraction": failure_waste / (args.workers * args.deadline),
               "feasible_incumbent": bool(best is not None),
               "runtime_quantile_coverage": {str(q): float(np.mean([data["runtime"][i] <= predicted_quantiles[i][j] for i in exact_times])) if exact_times else None for j, q in enumerate([0.1, 0.5, 0.9])},
               "deadline_unfinished": len(pending), "unfinished_worker_seconds": float(active),
               "worker_occupancy": float(busy / (args.workers * args.deadline)),
               "optimizer_seconds": overhead, "recovery_seconds_committed": recovery_time}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
    (out / "events.json").write_text(json.dumps(trace, indent=2, allow_nan=False))
    (out / 'launches.json').write_text('[\n' + ',\n'.join(launch_log) + '\n]', encoding='utf-8')
    (out / 'settings.json').write_text(json.dumps({k: v for k, v in vars(args).items() if k != 'function'}, indent=2), encoding='utf-8')
    (out / "pending.json").write_text(json.dumps([{ "task": str(data["task"][i]), "elapsed_lower_bound": max(0, args.deadline - job["start"])} for i, job in pending.items()], indent=2))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.step([r["time"] for r in trace if r["incumbent"] is not None], [r["incumbent"] for r in trace if r["incumbent"] is not None], where="post")
    ax.set(xlabel="Wall-clock budget (s)", ylabel="Completed target incumbent")
    fig.tight_layout()
    fig.savefig(out / "incumbent.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--csv", required=True)
    p.add_argument("--output", default="data/prepared")
    p.set_defaults(function=prepare)
    p = sub.add_parser("run")
    p.add_argument("--data", default="data/prepared")
    p.add_argument("--output", default="runs/full")
    p.add_argument("--method", choices=["full", "information", "random", *METHODS], default="full")
    p.add_argument('--robust-variance', type=float, default=.1)
    p.add_argument('--robust-information', type=float, default=.001)
    for key, value in [("workers", 4), ("initial", 8), ("candidates", 32), ("samples", 32), ("fantasies", 16), ("seed", 42)]:
        p.add_argument("--" + key, type=int, default=value)
    for key, value in [("deadline", 300), ("queue-delay", 1), ("source-scale", 1), ("recovery-prior", 1), ("stop", 1e-8), ("violation-tolerance", 0.05)]:
        p.add_argument("--" + key, type=float, default=value)
    p.add_argument("--net-time", action=argparse.BooleanOptionalAction, default=True)
    for flag in ["no-pending", "no-reliability", "no-failure", "no-runtime"]:
        p.add_argument("--" + flag, action="store_true")
    p.set_defaults(function=optimize)
    args = parser.parse_args()
    args.function(args)
