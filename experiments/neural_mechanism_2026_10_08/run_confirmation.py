"""Locked validation with fresh seeds and a uniform PB-AMD initialization."""
import argparse
import hashlib
import json
from pathlib import Path
import run_regularized as r

def from_uniform(problem, xi, initial, iterations):
    q = r.s.np.full(problem.T, problem.rho_max/problem.T)
    theta0 = problem.theta_design_ref[0]
    initial_f = r.s.f.objective_only(problem, q, xi, theta0)
    # At a homogeneous quadratic optimum, |grad_t| = F/rho, not T*F/rho.
    alpha = .3 * problem.rho_max / initial_f
    history, candidates, max_feas = [], [q.copy()], 0.
    for k in range(iterations):
        obj, grad = r.s.f.objective_and_grad(problem, q, xi, theta0)
        trial_alpha = alpha
        for attempt in range(30):
            trial, _, error = r.s.floor_aware_entropy_step(q, grad, trial_alpha,
                problem.rho_max, problem.q_min)
            trial_f = r.s.f.objective_only(problem, trial, xi, theta0)
            directional = float(grad @ (trial-q))
            if trial_f <= obj + .1*directional + 1e-12*max(1.,obj):
                break
            trial_alpha *= .5
        else:
            raise RuntimeError('confirmation mirror line search failed')
        gap = float(grad @ q-problem.q_min*grad.sum()
            -(problem.rho_max-problem.T*problem.q_min)*grad.min()) / max(obj,1e-30)
        history.append({'k':k,'objective':obj,'next_objective':trial_f,
            'relative_variational_gap':gap,'alpha':trial_alpha,'backtracks':attempt})
        max_feas = max(max_feas,error)
        q = trial
        alpha = min(trial_alpha*1.1,2*problem.rho_max/initial_f)
        if (k+1) % 10 == 0 or k == iterations-1:
            candidates.append(q.copy())
    return candidates, history, max_feas


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--out', type=Path, required=True)
    args, _ = parser.parse_known_args()
    lock = {
        'entrypoint_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'initialization':'PB-AMD starts from uniform, not the linearized comparator',
        'checkpoint_selection':'public validation only; includes uniform checkpoint zero',
        'stepsize':'initial 0.3*rho/F(initial), Armijo 0.1, shrink 0.5, grow 1.1, cap 2*rho/F(initial)',
        'comparison':'LR and exact reference-linearized allocation',
        'primary_endpoint':'private terminal parameter tracking error',
        'secondary_endpoints':'test accuracy and cross-entropy, not used for selection',
        'inference':'paired seed-level t intervals; Holm adjustment over all six budget-by-baseline tests',
        'scope':'new seeds after exploration; results cannot establish a universal dominance claim',
    }
    path = args.out/'confirmation_lock.json'
    if path.exists():
        assert json.loads(path.read_text()) == lock
    else:
        r.s.save(path, lock)
    r.s.train_pb = from_uniform
    r.main()
