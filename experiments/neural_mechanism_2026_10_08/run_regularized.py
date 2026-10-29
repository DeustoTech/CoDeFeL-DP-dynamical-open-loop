"""Regularized neural fine-tuning, retaining the quadratic tracking objective."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import run_study as s


@dataclass(frozen=True)
class RegularizedSetting(s.Setting):
    l2: float = .01


SETTINGS = {
    'regularized_001': RegularizedSetting('regularized_001',120,.5,1.,100,l2=.01),
    'regularized_003': RegularizedSetting('regularized_003',120,.5,1.,100,l2=.03),
}
original_make = s.make_problem


def make_regularized(setting, seed, rho):
    p = original_make(setting, seed, rho)
    theta0 = p.theta_design_ref[0]
    rd = s.f.compute_reference(theta0, p.x_design, p.y_design, p.batches,
        p.active_masks, p.eta, p.shape, setting.l2, p.clip)
    rt = s.f.compute_reference(theta0, p.x_train, p.y_train, p.batches,
        p.active_masks, p.eta, p.shape, setting.l2, p.clip)
    return replace(p, l2=setting.l2, theta_design_ref=rd, theta_ref=rt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings',default=','.join(SETTINGS))
    parser.add_argument('--seeds',default='62001')
    parser.add_argument('--rhos',default='0.5')
    parser.add_argument('--opt-paths',type=int,default=64)
    parser.add_argument('--val-paths',type=int,default=128)
    parser.add_argument('--eval-paths',type=int,default=256)
    parser.add_argument('--iterations',type=int,default=40)
    parser.add_argument('--probes',type=int,default=256)
    parser.add_argument('--exact-linear',action='store_true')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--out',type=Path,default=s.HERE/'regularized_exploration')
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    manifest = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    manifest['settings_definitions'] = {k:asdict(v) for k,v in SETTINGS.items()}
    manifest['source_sha256'] = {str(p):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (Path(__file__), s.HERE/'run_study.py',s.ROOT/'code/fashion_mnist_mlp_schedule_compare.py',
                  s.ROOT/'code/experiment_utils.py')}
    manifest['numpy_version'] = s.np.__version__
    manifest['protocol'] = ('Public pretraining uses original l2=1e-4. Private and public-design '
        'fine-tuning use the stated stronger l2; learning rate is constant. '
        'Quadratic tracking objective unchanged. All settings and results retained.')
    path = args.out/'manifest.json'
    if path.exists():
        assert json.loads(path.read_text()) == manifest, 'incompatible resume'
    else:
        s.save(path,manifest)
    s.make_problem = make_regularized
    s.f.load_preprocessed_fashion_mnist(str((s.ROOT/'data/fashion_mnist').resolve()))
    jobs = [(SETTINGS[name],int(seed),float(rho)) for name in args.settings.split(',')
            for seed in args.seeds.split(',') for rho in args.rhos.split(',')]
    completed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(s.run_case,*job,args) for job in jobs]
        for future in as_completed(futures):
            completed.append(future.result())
            s.save(args.out/'completion.json',{'completed':len(completed),'expected':len(jobs),
                'status':'complete' if len(completed)==len(jobs) else 'running'})


if __name__ == '__main__':
    main()
