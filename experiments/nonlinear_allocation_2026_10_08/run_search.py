"""Public-only exploration followed by independent neural confirmation."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / 'neural_mechanism_2026_10_08'
sys.path.insert(0, str(PREVIOUS))
import run_study as s
from run_confirmation import from_uniform
np = s.np


@dataclass(frozen=True)
class Setting:
    name: str
    features: int = 8
    hidden: int = 4
    horizon: int = 120
    eta: float = .5
    batch: int = 128
    clip: float = 1.
    l2: float = 1e-4
    head_pretrain: int = 0


SETTINGS = {
    c.name: c for c in [
        Setting('train_small'),
        Setting('train_small_batch256', batch=256),
        Setting('train_small_long', horizon=240),
        Setting('train_small_fast', eta=1.),
        Setting('finetune_small', head_pretrain=100),
        Setting('finetune_small_batch256', batch=256, head_pretrain=100),
        Setting('train_wider', hidden=8),
        Setting('train_fullfeatures', features=32, hidden=4),
    ]
}
BASELINES = ('LR', 'Decay', 'Linearized', 'Tuned-geometric')
GEOMETRIC_GRID = np.linspace(-4., 4., 17)


@lru_cache(maxsize=12)
def public_features(seed):
    base = s.f.make_problem(s.ROOT/'data/fashion_mnist', seed=seed,
        scenario='lr_restart', n_train=8000, n_test=3000, rho_max=.5)
    w, b, _, _ = s.f.unpack(base.theta_design_ref[0], base.shape)
    features = [s.f.softplus(x @ w[0]+b[0])
        for x in (base.x_design, base.x_train, base.x_test)]
    mean = features[0].mean(0)
    centered = [x-mean for x in features]
    _, _, vt = np.linalg.svd(centered[0], full_matrices=False)
    # Fix PCA signs so reconstruction does not depend on SVD sign choices.
    for row in vt:
        j = int(np.argmax(np.abs(row)))
        if row[j] < 0:
            row *= -1
    return base, centered, vt


def make_problem(setting, seed, rho):
    base, centered, vt = public_features(seed)
    if setting.features < 32:
        features = [x @ vt[:setting.features].T for x in centered]
    else:
        features = centered
    std = features[0].std(0)+1e-6
    xd, xt, xe = [x/std for x in features]
    shape = s.f.MLPShape(setting.features, setting.hidden, 10)
    rng = np.random.default_rng(seed+180000)
    theta = s.f.init_theta(shape, rng)
    if setting.head_pretrain:
        theta = s.f.pretrain_theta(theta, xd, base.y_design, shape,
            setting.l2, setting.clip, steps=setting.head_pretrain,
            batch_size=setting.batch, eta=.2, seed=seed+181000)
    batches = [rng.choice(len(xd), setting.batch, replace=False)
        for _ in range(setting.horizon)]
    eta = np.full(setting.horizon, setting.eta)
    masks = np.ones((setting.horizon, shape.p))
    rd = s.f.compute_reference(theta, xd, base.y_design, batches, masks,
        eta, shape, setting.l2, setting.clip)
    rt = s.f.compute_reference(theta, xt, base.y_train, batches, masks,
        eta, shape, setting.l2, setting.clip)
    return replace(base, x_design=xd, x_train=xt, x_test=xe, shape=shape,
        batches=batches, active_masks=masks, eta=eta, theta_design_ref=rd,
        theta_ref=rt, sensitivity=2*setting.clip/setting.batch,
        clip=setting.clip, l2=setting.l2, scenario=setting.name,
        rho_max=rho, phase_split=None)


def analytic_schedules(p):
    a = s.linear_coefficients(p)
    factors = np.abs(1-p.eta*p.l2)
    assert np.all(factors > 0)
    suffix = np.exp(np.cumsum(np.log(factors)[::-1])[::-1]-np.log(factors))
    return {
        'LR': s.allocate(p.eta, p.rho_max, p.q_min),
        'Decay': s.allocate(p.eta*suffix, p.rho_max, p.q_min),
        'Linearized': s.allocate(np.sqrt(a), p.rho_max, p.q_min),
    }, a


def source_hashes():
    files = [Path(__file__), PREVIOUS/'run_study.py', PREVIOUS/'run_confirmation.py',
        PREVIOUS/'run_regularized.py', s.ROOT/'code/fashion_mnist_mlp_schedule_compare.py',
        s.ROOT/'code/experiment_utils.py']
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def run_case(setting, seed, rho, args):
    tag = f'{setting.name}_seed{seed}_rho{rho:g}'
    path = args.out/(tag+'.json')
    if path.exists():
        result = json.loads(path.read_text())
        assert result['source_sha256'] == source_hashes()
        return result
    start = time.perf_counter()
    print('START '+tag, flush=True)
    p = make_problem(setting, seed, rho)
    derivative_errors = s.check_derivatives(p, seed)
    schedules, coefficients = analytic_schedules(p)
    xi = np.random.default_rng(seed+184000).normal(size=(p.T,args.opt_paths,p.shape.p))
    candidates, history, feasibility = from_uniform(p, xi, schedules['LR'], args.iterations)
    labels = [0]+list(range(10,args.iterations+1,10))
    if args.iterations % 10:
        labels.append(args.iterations)
    choices = {f'k{k}':q for k,q in zip(labels,candidates,strict=True)}
    validation = s.evaluate(p, choices, seed+185000, args.val_paths, public=True, metrics=False)
    selected = min(validation, key=lambda key:validation[key]['error'])
    schedules['PB-AMD'] = choices[selected]
    geometric = {f'beta{beta:g}':s.allocate(p.eta*np.exp(beta*np.linspace(0,1,p.T)),rho,p.q_min)
        for beta in GEOMETRIC_GRID}
    gv = s.evaluate(p, geometric, seed+185000, args.val_paths, public=True, metrics=False)
    selected_geo = min(gv,key=lambda key:gv[key]['error'])
    schedules['Tuned-geometric'] = geometric[selected_geo]
    # Neither setting selection nor checkpoint selection uses private outcomes.
    public = s.evaluate(p, schedules, seed+186000,args.eval_paths,public=True,metrics=False)
    result = {'setting':asdict(setting),'seed':seed,'rho':rho,'phase':args.phase,
        'source_sha256':source_hashes(),'parameters':p.shape.p,
        'schedules':{k:v.tolist() for k,v in schedules.items()},
        'linear_coefficients':coefficients.tolist(),
        'selected_checkpoint':selected,'selected_geometric':selected_geo,
        'validation':{k:v['error'] for k,v in validation.items()},
        'geometric_validation':{k:v['error'] for k,v in gv.items()},
        'candidate_schedules':{k:v.tolist() for k,v in choices.items()},
        'pb_history':history,'public':public,'gradient_check_relative_errors':derivative_errors,
        'max_feasibility_error':feasibility}
    result['public_gains'] = {name:100*(1-public['PB-AMD']['error']/public[name]['error']) for name in BASELINES}
    if args.phase == 'confirmation':
        private = s.evaluate(p,schedules,seed+187000,args.eval_paths)
        loss,acc = s.f.mlp_loss_acc(p.theta_ref[-1],p.x_test,p.y_test,p.shape,p.l2)
        result.update(private=private,no_noise_accuracy=float(acc[0]),no_noise_loss=float(loss[0]))
        result['private_gains'] = {name:100*(1-private['PB-AMD']['error']/private[name]['error']) for name in BASELINES}
    result['seconds'] = time.perf_counter()-start
    s.save(path,result)
    print(f'DONE {tag} public={result["public_gains"]} private={result.get("private_gains")} '
          f'geo={selected_geo} time={result["seconds"]:.1f}s',flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings',default=','.join(SETTINGS))
    parser.add_argument('--seeds',default='94001')
    parser.add_argument('--rhos',default='0.5')
    parser.add_argument('--opt-paths',type=int,default=128)
    parser.add_argument('--val-paths',type=int,default=256)
    parser.add_argument('--eval-paths',type=int,default=512)
    parser.add_argument('--iterations',type=int,default=40)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--phase',choices=('exploration','confirmation'),default='exploration')
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    manifest = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    manifest.update(source_sha256=source_hashes(),
        settings_definitions={k:asdict(v) for k,v in SETTINGS.items()},
        geometric_grid=GEOMETRIC_GRID.tolist(), numpy_version=np.__version__,
        protocol='Unchanged squared parameter tracking objective. Public-only design and checkpoint/geometric selection. '
        'Frozen public encoder; public PCA and feature scaling. Exact linear baseline. '
        'Constant learning rate, weak L2; no private evaluation during exploration. '
        'All settings/seeds retained. Confirmation uses disjoint seed IDs. '
        'Primary inference is paired seed-level error reduction vs all four baselines, with Holm adjustment.')
    target = args.out/'manifest.json'
    if target.exists():
        assert json.loads(target.read_text()) == manifest,'incompatible resume'
    else:
        s.save(target,manifest)
    seeds = list(map(int,args.seeds.split(',')))
    # Serial cache warmup avoids redundant backbone training in worker threads.
    for seed in seeds:
        public_features(seed)
    jobs = [(SETTINGS[name],seed,float(rho)) for name in args.settings.split(',')
        for seed in seeds for rho in args.rhos.split(',')]
    n = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_case,*job,args) for job in jobs]
        for future in as_completed(futures):
            future.result()
            n += 1
            s.save(args.out/'completion.json',{'completed':n,'expected':len(jobs),
                'status':'complete' if n == len(jobs) else 'running'})


if __name__ == '__main__':
    main()
