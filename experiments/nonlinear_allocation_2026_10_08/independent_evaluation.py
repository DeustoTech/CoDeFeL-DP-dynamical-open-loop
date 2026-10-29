"""Locked final evaluation in RNG namespaces disjoint from every design stage."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
import run_fast_head as f
from summarize_results import summarize

r,s,np = f.r,f.r.s,f.r.np
DIRECTORY = r.HERE/'confirmation'
OUT = DIRECTORY/'independent_evaluation'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_case(path,manifest):
    target = OUT/path.name
    refined_path = DIRECTORY/'refined_geometric'/path.name
    if target.exists():
        result = json.loads(target.read_text())
        assert result['case_sha256']==sha(path) and result['refined_sha256']==sha(refined_path)
        assert result['evaluator_sha256']==sha(Path(__file__))
        return result
    c = json.loads(path.read_text())
    refined = json.loads(refined_path.read_text())
    assert c['source_sha256']==r.source_hashes()
    p = r.make_problem(r.Setting(**c['setting']),c['seed'],c['rho'])
    schedules = {name:np.asarray(q) for name,q in c['schedules'].items()}
    schedules['Refined-geometric'] = np.asarray(refined['schedule'])
    result = {'case_sha256':sha(path),'refined_sha256':sha(refined_path),
        'evaluator_sha256':sha(Path(__file__)),'source_sha256':r.source_hashes(),
        'seed':c['seed'],'rho':c['rho'],'paths':manifest['eval_paths'],
        'public_gaussian_seed':8_000_000+c['seed'],'private_gaussian_seed':9_000_000+c['seed']}
    max_error = 0.
    for scope,offset in [('public',8_000_000),('private',9_000_000)]:
        seed = offset+c['seed']
        public = scope=='public'
        values = s.evaluate(p,schedules,seed,manifest['eval_paths'],public=public,metrics=not public)
        # A separate simulator replays the first 16 paths, for every method.
        pp = p if public else replace(p,x_design=p.x_train,y_design=p.y_train,theta_design_ref=p.theta_ref)
        xi = np.random.default_rng(seed).normal(size=(p.T,16,p.shape.p))
        for name,q in schedules.items():
            expected = np.mean(values[name]['path_errors'][:16])
            actual = s.f.objective_only(pp,q,xi,pp.theta_design_ref[0])
            error = abs(expected-actual)/max(1.,abs(expected))
            assert error<1e-12
            max_error = max(max_error,error)
        result[scope] = values
    result['max_independent_replay_error'] = max_error
    result['private_gains'] = {name:100*(1-result['private']['PB-AMD']['error']/result['private'][name]['error'])
        for name in ('LR','Decay','Linearized','Refined-geometric')}
    s.save(target,result)
    print(f'INDEPENDENT {path.name}: {result["private_gains"]}',flush=True)
    return result


def main():
    # The code-only package has no historical screening manifests. Reuse the
    # fresh-run orchestration while preserving evaluate_case above unchanged.
    import argparse
    import sys
    parser = argparse.ArgumentParser(description='Evaluate one completed final cohort without historical screening files.')
    parser.add_argument('--cohort', choices=('confirmation','budget_extension'), default='confirmation')
    args = parser.parse_args()
    sys.path.insert(0, str(r.HERE.parents[1]))
    from run_head_final import COHORTS, evaluate
    for dirname, rho, seeds in COHORTS:
        if dirname == args.cohort:
            evaluate(r.HERE / dirname, seeds, rho)
            break
    print('INDEPENDENT FINAL EVALUATION COMPLETE',flush=True)


if __name__=='__main__':
    main()
