"""Refine the public-tuned geometric comparator without using test outcomes."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import run_search as r

np,s = r.np,r.s


def golden(function,left,right,iterations=18):
    ratio = (np.sqrt(5.)-1.)/2.
    x1,x2 = right-ratio*(right-left),left+ratio*(right-left)
    f1,f2 = function(x1),function(x2)
    for _ in range(iterations):
        if f1 < f2:
            right,x2,f2 = x2,x1,f1
            x1 = right-ratio*(right-left)
            f1 = function(x1)
        else:
            left,x1,f1 = x1,x2,f2
            x2 = left+ratio*(right-left)
            f2 = function(x2)
    return (x1,f1) if f1<f2 else (x2,f2)


def run_case(path,manifest):
    target = path.parent/'refined_geometric'/path.name
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if target.exists():
        old = json.loads(target.read_text())
        assert old['case_sha256'] == digest and old['script_sha256'] == script_hash
        return old
    c = json.loads(path.read_text())
    p = r.make_problem(r.Setting(**c['setting']),c['seed'],c['rho'])
    beta0 = float(c['selected_geometric'].removeprefix('beta'))
    trace = []
    def allocation(beta):
        return s.allocate(p.eta*np.exp(beta*np.linspace(0,1,p.T)),p.rho_max,p.q_min)
    def objective(beta):
        q = allocation(beta)
        value = s.evaluate(p,{'geo':q},c['seed']+185000,manifest['val_paths'],public=True,metrics=False)['geo']['error']
        trace.append({'beta':float(beta),'validation_error':value})
        return value
    initial = objective(beta0)
    np.testing.assert_allclose(initial,c['geometric_validation'][c['selected_geometric']],rtol=1e-13)
    left,right = max(-4.,beta0-.5),min(4.,beta0+.5)
    beta,value = golden(objective,left,right)
    if initial<value:
        beta,value = beta0,initial
    q = allocation(beta)
    result = {'case_sha256':digest,'script_sha256':script_hash,
        'source_sha256':r.source_hashes(),'seed':c['seed'],'rho':c['rho'],
        'setting':c['setting'],'beta':float(beta),'validation_error':value,
        'schedule':q.tolist(),'trace':trace}
    for scope,offset in [('public',186000)]+([('private',187000)] if c['phase']=='confirmation' else []):
        metrics = scope=='private'
        evaluation = s.evaluate(p,{'Refined-geometric':q},c['seed']+offset,
            manifest['eval_paths'],public=not metrics,metrics=metrics)['Refined-geometric']
        result[scope] = evaluation
        result[scope+'_pb_gain'] = 100*(1-c[scope]['PB-AMD']['error']/evaluation['error'])
    s.save(target,result)
    print(f'REFINED {path.name} beta={beta:.5f} public_gain={result["public_pb_gain"]:.6f} '
        f'private_gain={result.get("private_pb_gain")}',flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    parser.add_argument('--workers',type=int,default=4)
    args = parser.parse_args()
    manifest = json.loads((args.directory/'manifest.json').read_text())
    assert manifest['source_sha256'] == r.source_hashes()
    paths = sorted(args.directory.glob('*_seed*_rho*.json'))
    assert paths
    lock = {'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'rule':'Refine the best public-validation grid point in its neighboring bracket with 18 golden-section iterations; retain the grid candidate if better. No private outcomes used.',
        'primary_comparator':'Refined-geometric replaces the coarse grid comparator for final claims.'}
    target = args.directory/'refined_geometric'/'protocol.json'
    if target.exists():
        assert json.loads(target.read_text()) == lock
    else:
        s.save(target,lock)
    for seed in sorted({json.loads(path.read_text())['seed'] for path in paths}):
        r.public_features(seed)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_case,path,manifest) for path in paths]
        for future in as_completed(futures):
            future.result()


if __name__ == '__main__':
    main()
