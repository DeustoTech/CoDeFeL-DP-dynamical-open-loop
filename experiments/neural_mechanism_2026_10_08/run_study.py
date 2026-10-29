"""Public-feature neural fine-tuning: nonlinear versus analytic allocation."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] / "Control_Theoretic_DP_Numerics"
sys.path.insert(0, str(ROOT / "code"))
import fashion_mnist_mlp_schedule_compare as f
from experiment_utils import floor_aware_entropy_step, student_t_quantile


@dataclass(frozen=True)
class Setting:
    name: str
    horizon: int
    eta: float
    clip: float
    head_pretrain: int
    hidden: int = 12
    batch: int = 128


SETTINGS = {
    "head_training": Setting("head_training", 80, .3, 1., 0),
    "head_finetune": Setting("head_finetune", 120, .5, 1., 100),
    "head_finetune_clip2": Setting("head_finetune_clip2", 120, .5, 2., 100),
}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def allocate(weights, rho, floor):
    weights = np.asarray(weights)
    assert np.all(weights > 0) and len(weights) * floor < rho
    lo, hi = 0., rho / weights.min()
    for _ in range(90):
        mid = (lo + hi) / 2
        if np.maximum(floor, mid * weights).sum() < rho:
            lo = mid
        else:
            hi = mid
    return np.maximum(floor, (lo + hi) / 2 * weights)


def make_problem(setting, seed, rho):
    base = f.make_problem(ROOT / "data/fashion_mnist", seed=seed, scenario="lr_restart",
                          n_train=8000, n_test=3000, rho_max=rho)
    w, b, _, _ = f.unpack(base.theta_design_ref[0], base.shape)
    features = [f.softplus(x @ w[0] + b[0]) for x in (base.x_design, base.x_train, base.x_test)]
    mean, std = features[0].mean(0), features[0].std(0) + 1e-6
    xd, xt, xe = [(x - mean) / std for x in features]
    shape = f.MLPShape(32, setting.hidden, 10)
    rng = np.random.default_rng(seed + 80000)
    theta = f.init_theta(shape, rng)
    if setting.head_pretrain:
        theta = f.pretrain_theta(theta, xd, base.y_design, shape, 1e-4, setting.clip,
            steps=setting.head_pretrain, batch_size=setting.batch, eta=.2, seed=seed + 81000)
    batches = [rng.choice(len(xd), setting.batch, replace=False) for _ in range(setting.horizon)]
    eta = np.full(setting.horizon, setting.eta)
    masks = np.ones((setting.horizon, shape.p))
    rd = f.compute_reference(theta, xd, base.y_design, batches, masks, eta, shape, 1e-4, setting.clip)
    rt = f.compute_reference(theta, xt, base.y_train, batches, masks, eta, shape, 1e-4, setting.clip)
    return replace(base, x_design=xd, x_train=xt, x_test=xe, shape=shape,
        batches=batches, active_masks=masks, eta=eta, theta_design_ref=rd, theta_ref=rt,
        sensitivity=2 * setting.clip / setting.batch, clip=setting.clip,
        scenario=setting.name, phase_split=None)


def linear_coefficients(problem, probes=None, seed=0):
    """a_t = eta_t^2 Delta^2 ||A_(T-1)...A_(t+1)||_F^2 / 2."""
    p = problem.shape.p
    if probes is None:
        v = np.eye(p)
        scale = 1.
    else:
        v = np.random.default_rng(seed).choice([-1., 1.], (probes, p))
        scale = 1. / probes
    coefficients = np.zeros(problem.T)
    for t in range(problem.T - 1, -1, -1):
        coefficients[t] = .5 * (problem.sensitivity * problem.eta[t])**2 * np.sum(v*v) * scale
        idx = problem.batches[t]
        state = np.broadcast_to(problem.theta_design_ref[t], v.shape)
        v -= problem.eta[t] * f.mlp_grad_vjp(state, v, problem.x_design[idx],
            problem.y_design[idx], problem.shape, problem.l2, problem.clip)
    return coefficients


def train_pb(problem, xi, initial, iterations):
    q = initial.copy()
    theta0 = problem.theta_design_ref[0]
    initial_f = f.objective_only(problem, q, xi, theta0)
    # Objective normalization changes the step size, not the mirror update.
    alpha = .3 * problem.rho_max / (problem.T * initial_f)
    history, candidates = [], [q.copy()]
    max_feas = 0.
    for k in range(iterations):
        obj, grad = f.objective_and_grad(problem, q, xi, theta0)
        trial_alpha = alpha
        for attempt in range(25):
            trial, _, error = floor_aware_entropy_step(q, grad, trial_alpha, problem.rho_max, problem.q_min)
            trial_f = f.objective_only(problem, trial, xi, theta0)
            directional = float(grad @ (trial-q))
            if trial_f <= obj + .1 * directional + 1e-12 * max(1., obj):
                break
            trial_alpha *= .5
        else:
            raise RuntimeError("mirror line search failed")
        relative_gap = float(grad @ q - problem.q_min * grad.sum()
            - (problem.rho_max - problem.T * problem.q_min) * grad.min()) / max(obj, 1e-30)
        history.append({"k": k, "objective": obj, "next_objective": trial_f,
            "relative_variational_gap": relative_gap, "alpha": trial_alpha,
            "backtracks": attempt})
        max_feas = max(max_feas, error)
        q = trial
        alpha = min(trial_alpha * 1.1, problem.rho_max / (problem.T * initial_f))
        if (k+1) % 10 == 0 or k == iterations-1:
            candidates.append(q.copy())
    return candidates, history, max_feas


def evaluate(problem, schedules, seed, paths, public=False, metrics=True):
    names = list(schedules)
    qs = np.array(list(schedules.values()))
    data = problem.x_design if public else problem.x_train
    labels = problem.y_design if public else problem.y_train
    ref = problem.theta_design_ref if public else problem.theta_ref
    errors = np.zeros((len(names), paths))
    acc = np.zeros_like(errors)
    losses = np.zeros_like(errors)
    rng = np.random.default_rng(seed)
    for start in range(0, paths, 16):
        m = min(16, paths-start)
        theta = np.broadcast_to(ref[0], (len(names), m, problem.shape.p)).copy()
        for t, idx in enumerate(problem.batches):
            grad = f.mlp_grad(theta.reshape(-1, problem.shape.p), data[idx], labels[idx],
                problem.shape, problem.l2, problem.clip).reshape(theta.shape)
            noise = rng.normal(size=(m, problem.shape.p))
            amp = problem.sensitivity * problem.eta[t] / np.sqrt(2*qs[:, t])
            theta -= problem.eta[t] * grad + amp[:, None, None] * noise
        errors[:, start:start+m] = np.sum((theta-ref[-1])**2, axis=2)
        if metrics:
            loss, accuracy = f.mlp_loss_acc(theta.reshape(-1, problem.shape.p), problem.x_test,
                                           problem.y_test, problem.shape, problem.l2)
            losses[:, start:start+m] = loss.reshape(len(names), m)
            acc[:, start:start+m] = accuracy.reshape(len(names), m)
    result = {}
    for j, name in enumerate(names):
        result[name] = {"error": float(errors[j].mean()), "path_errors": errors[j].tolist()}
        if metrics:
            result[name].update(accuracy=float(acc[j].mean()), loss=float(losses[j].mean()))
    return result


def check_derivatives(problem, seed):
    rng = np.random.default_rng(seed + 82000)
    xi = rng.normal(size=(problem.T, 3, problem.shape.p))
    q = np.full(problem.T, problem.rho_max/problem.T)
    _, h = f.objective_and_grad(problem, q, xi, problem.theta_design_ref[0])
    checks = []
    for _ in range(3):
        d = rng.normal(size=problem.T)
        d -= d.mean()
        d /= np.linalg.norm(d)
        eps = 1e-5 * q.min()
        numerical = (f.objective_only(problem, q+eps*d, xi, problem.theta_design_ref[0])
                    - f.objective_only(problem, q-eps*d, xi, problem.theta_design_ref[0])) / (2*eps)
        exact = h @ d
        checks.append(float(abs(numerical-exact) / max(1., abs(numerical), abs(exact))))
    assert max(checks) < 2e-6, checks
    return checks


def run_case(setting, seed, rho, args):
    tag = f"{setting.name}_seed{seed}_rho{rho:g}"
    target = args.out / (tag + ".json")
    if target.exists():
        return json.loads(target.read_text())
    started = time.perf_counter()
    print("START " + tag, flush=True)
    problem = make_problem(setting, seed, rho)
    derivative_errors = check_derivatives(problem, seed)
    coefficients = linear_coefficients(problem, None if args.exact_linear else args.probes, seed+83000)
    lr = allocate(problem.eta, rho, problem.q_min)
    linear = allocate(np.sqrt(coefficients), rho, problem.q_min)
    xi = np.random.default_rng(seed+84000).normal(size=(problem.T, args.opt_paths, problem.shape.p))
    candidates, history, feas = train_pb(problem, xi, linear, args.iterations)
    choices = {f"k{k*10}": q for k, q in enumerate(candidates)}
    val = evaluate(problem, choices, seed+85000, args.val_paths, public=True, metrics=False)
    chosen = min(val, key=lambda key: val[key]["error"])
    schedules = {"LR": lr, "Linearized": linear, "PB-AMD": choices[chosen]}
    public = evaluate(problem, schedules, seed+86000, args.eval_paths, public=True, metrics=False)
    private = evaluate(problem, schedules, seed+87000, args.eval_paths)
    no_noise_loss, no_noise_acc = f.mlp_loss_acc(problem.theta_ref[-1], problem.x_test,
        problem.y_test, problem.shape, problem.l2)
    result = {"setting": asdict(setting), "seed": seed, "rho": rho,
        "parameters": problem.shape.p, "selected_checkpoint": chosen,
        "schedules": {k: v.tolist() for k, v in schedules.items()},
        "linear_coefficients": coefficients.tolist(), "linear_exact": args.exact_linear,
        "public": public, "private": private, "validation": {k:v["error"] for k,v in val.items()},
        "pb_history": history, "gradient_check_relative_errors": derivative_errors,
        "max_feasibility_error": feas, "no_noise_accuracy": float(no_noise_acc[0]),
        "no_noise_loss": float(no_noise_loss[0]), "seconds": time.perf_counter()-started}
    for scope in ("public", "private"):
        result[scope+"_gains"] = {baseline: 100*(1-result[scope]["PB-AMD"]["error"]/result[scope][baseline]["error"])
                                   for baseline in ("LR", "Linearized")}
    save(target, result)
    print(f"DONE {tag} gains={result['private_gains']} public={result['public_gains']} "
          f"acc={private['PB-AMD']['accuracy']:.4f} ref={no_noise_acc[0]:.4f} "
          f"time={result['seconds']:.1f}s", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", default=",".join(SETTINGS))
    parser.add_argument("--seeds", default="61001")
    parser.add_argument("--rhos", default="0.5")
    parser.add_argument("--opt-paths", type=int, default=64)
    parser.add_argument("--val-paths", type=int, default=128)
    parser.add_argument("--eval-paths", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--probes", type=int, default=256)
    parser.add_argument("--exact-linear", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--out", type=Path, default=HERE/"exploration")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
    manifest["settings_definitions"] = {k:asdict(v) for k,v in SETTINGS.items()}
    manifest["source_sha256"] = {str(p):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (Path(__file__), ROOT/"code/fashion_mnist_mlp_schedule_compare.py", ROOT/"code/experiment_utils.py")}
    manifest["numpy_version"] = np.__version__
    manifest["protocol"] = "public design/selection; disjoint private training; fresh Gaussian evaluation; all results retained"
    path = args.out/"manifest.json"
    if path.exists():
        assert json.loads(path.read_text()) == manifest, "incompatible resume"
    else:
        save(path, manifest)
    f.load_preprocessed_fashion_mnist(str((ROOT/"data/fashion_mnist").resolve()))
    jobs = [(SETTINGS[name], int(seed), float(rho)) for name in args.settings.split(",")
            for seed in args.seeds.split(",") for rho in args.rhos.split(",")]
    completed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_case, *job, args) for job in jobs]
        for future in as_completed(futures):
            completed.append(future.result())
            save(args.out/"completion.json", {"completed":len(completed), "expected":len(jobs),
                "status":"complete" if len(completed)==len(jobs) else "running"})


if __name__ == "__main__":
    main()
