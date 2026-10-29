"""Seed-level comparisons retaining every budget and every strong baseline."""
import argparse
import hashlib
import json
from pathlib import Path
import run_search as r
from summarize import interval
from experiment_utils import holm_adjusted_pvalues

np,s = r.np,r.s


def summarize(directory,independent=False):
    manifest = json.loads((directory/'manifest.json').read_text())
    if any(Path(path).name=='run_fast_head.py' for path in manifest['source_sha256']):
        import run_fast_head
    elif any(Path(path).name=='run_extended.py' for path in manifest['source_sha256']):
        import run_extended
    assert manifest['source_sha256'] == r.source_hashes()
    paths = sorted(directory.glob('*_seed*_rho*.json'))
    expected = {f'{name}_seed{seed}_rho{float(rho):g}.json'
        for name in manifest['settings'].split(',') for seed in manifest['seeds'].split(',')
        for rho in manifest['rhos'].split(',')}
    assert {p.name for p in paths} == expected
    cases = []
    for path in paths:
        c = json.loads(path.read_text())
        refined = json.loads((directory/'refined_geometric'/path.name).read_text())
        assert refined['case_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
        for scope in ['public']+(['private'] if 'private' in c else []):
            c[scope]['Refined-geometric'] = refined[scope]
        if independent:
            evaluation = json.loads((directory/'independent_evaluation'/path.name).read_text())
            assert evaluation['case_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
            assert evaluation['refined_sha256'] == hashlib.sha256((directory/'refined_geometric'/path.name).read_bytes()).hexdigest()
            c['public'],c['private'] = evaluation['public'],evaluation['private']
        cases.append(c)
    scope = 'private' if manifest['phase']=='confirmation' else 'public'
    rows = []
    for setting,rho in sorted({(c['setting']['name'],c['rho']) for c in cases}):
        group = [c for c in cases if (c['setting']['name'],c['rho'])==(setting,rho)]
        for baseline in ('LR','Decay','Linearized','Refined-geometric'):
            gains = [100*(1-c[scope]['PB-AMD']['error']/c[scope][baseline]['error']) for c in group]
            mean,low,high,pvalue = interval(gains)
            row = {'setting':setting,'rho':rho,'baseline':baseline,'scope':scope,
                'seeds':len(group),'error_reduction_percent':mean,'CI95_low':low,'CI95_high':high,
                'pvalue':pvalue,'wins':sum(g>0 for g in gains),'seed_level_gains':gains,
                'PB_error_mean':float(np.mean([c[scope]['PB-AMD']['error'] for c in group])),
                'baseline_error_mean':float(np.mean([c[scope][baseline]['error'] for c in group]))}
            if scope=='private':
                accuracy = [100*(c[scope]['PB-AMD']['accuracy']-c[scope][baseline]['accuracy']) for c in group]
                am,al,ah,_ = interval(accuracy)
                row.update(accuracy_difference_pp=am,accuracy_CI95_low=al,accuracy_CI95_high=ah,
                    PB_accuracy_percent=float(100*np.mean([c[scope]['PB-AMD']['accuracy'] for c in group])),
                    baseline_accuracy_percent=float(100*np.mean([c[scope][baseline]['accuracy'] for c in group])),
                    reference_accuracy_percent=float(100*np.mean([c['no_noise_accuracy'] for c in group])))
            rows.append(row)
    if all(row['pvalue'] is not None for row in rows):
        adjusted = holm_adjusted_pvalues([row['pvalue'] for row in rows])
        for row,value in zip(rows,adjusted,strict=True):
            row['holm_pvalue'] = value
    report = {'phase':manifest['phase'],'scope':scope,'cases':len(cases),'independent_evaluation':independent,
        'comparisons':rows,'source_sha256':r.source_hashes(),
        'summary_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'statistical_dependency_sha256':{str(path):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (r.PREVIOUS/'summarize.py',s.ROOT/'code/experiment_utils.py')},
        'case_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    if independent:
        report['evaluation_sha256'] = {p.name:hashlib.sha256((directory/'independent_evaluation'/p.name).read_bytes()).hexdigest() for p in paths}
    s.save(directory/('independent_summary.json' if independent else 'summary.json'),report)
    print(json.dumps(rows,indent=2))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    parser.add_argument('--independent',action='store_true')
    args = parser.parse_args()
    summarize(args.directory,args.independent)
