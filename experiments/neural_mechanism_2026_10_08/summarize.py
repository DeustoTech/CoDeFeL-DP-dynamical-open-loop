"""Verify completed case records and report paired, seed-level comparisons."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import run_study as s
from experiment_utils import student_t_quantile, student_t_two_sided_pvalue, holm_adjusted_pvalues


def interval(values):
    values = np.asarray(values)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, None, None, None
    se = float(values.std(ddof=1)/np.sqrt(len(values)))
    critical = student_t_quantile(.975, len(values)-1)
    if len(values) == 10:
        np.testing.assert_allclose(critical,2.262157162798205,rtol=1e-13,atol=0.)
    half = critical*se
    p = student_t_two_sided_pvalue(mean/se, len(values)-1) if se else float(mean == 0)
    return mean, mean-half, mean+half, p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.directory/'manifest.json').read_text())
    for path, digest in manifest['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    lock_path = args.directory/'confirmation_lock.json'
    if lock_path.exists():
        lock = json.loads(lock_path.read_text())
        assert hashlib.sha256((s.HERE/'run_confirmation.py').read_bytes()).hexdigest() == lock['entrypoint_sha256']
    cases = [json.loads(p.read_text()) for p in sorted(args.directory.glob('*_seed*_rho*.json'))]
    expected = {(name, int(seed), float(rho)) for name in manifest['settings'].split(',')
        for seed in manifest['seeds'].split(',') for rho in manifest['rhos'].split(',')}
    keys = [(c['setting']['name'], c['seed'], c['rho']) for c in cases]
    assert len(set(keys)) == len(keys) and set(keys) == expected
    max_budget_error = 0.
    for c in cases:
        for q in c['schedules'].values():
            q = np.array(q)
            error = abs(q.sum()-c['rho'])
            max_budget_error = max(max_budget_error, error)
            assert error < 1e-12 and q.min() >= 1e-5
        lr = np.asarray(c['schedules']['LR'])
        np.testing.assert_allclose(lr, np.full(len(lr),c['rho']/len(lr)),rtol=1e-13,atol=1e-16)
        a = np.array(c['linear_coefficients'])
        q = np.array(c['schedules']['Linearized'])
        free = q > 1e-5 * (1+1e-8)
        slope = a/q**2
        np.testing.assert_allclose(slope[free], np.full(free.sum(), slope[free].mean()), rtol=1e-10)
        assert np.all(slope[~free] <= slope[free].mean()*(1+1e-9))
        selected = c['selected_checkpoint']
        assert c['validation'][selected] == min(c['validation'].values())
        assert max(c['gradient_check_relative_errors']) < 2e-6
        assert all(h['next_objective'] <= h['objective'] + 1e-9 * max(1., h['objective'])
                   for h in c['pb_history'])
        for scope in ('public', 'private'):
            for name, result in c[scope].items():
                assert len(result['path_errors']) == manifest['eval_paths']
                np.testing.assert_allclose(result['error'], np.mean(result['path_errors']), rtol=1e-14)
    for setting, seed in {(c['setting']['name'],c['seed']) for c in cases}:
        matched = [c for c in cases if (c['setting']['name'],c['seed']) == (setting,seed)]
        for c in matched[1:]:
            np.testing.assert_array_equal(c['linear_coefficients'],matched[0]['linear_coefficients'])
            assert c['no_noise_accuracy'] == matched[0]['no_noise_accuracy']
    rows = []
    for setting, rho in sorted({(c['setting']['name'],c['rho']) for c in cases}):
        group = [c for c in cases if (c['setting']['name'],c['rho']) == (setting,rho)]
        for baseline in ('LR','Linearized'):
            gains = [100*(1-c['private']['PB-AMD']['error']/c['private'][baseline]['error']) for c in group]
            accuracy = [100*(c['private']['PB-AMD']['accuracy']-c['private'][baseline]['accuracy']) for c in group]
            mean, lo, hi, pvalue = interval(gains)
            am, al, ah, _ = interval(accuracy)
            rows.append({'setting':setting, 'rho':rho, 'baseline':baseline, 'n_seeds':len(group),
                'error_reduction_percent':mean, 'CI95_low':lo, 'CI95_high':hi,
                'pvalue':pvalue, 'accuracy_difference_pp':am, 'accuracy_CI95_low':al,
                'accuracy_CI95_high':ah,
                'PB_accuracy_percent':100*np.mean([c['private']['PB-AMD']['accuracy'] for c in group]),
                'baseline_accuracy_percent':100*np.mean([c['private'][baseline]['accuracy'] for c in group]),
                'reference_accuracy_percent':100*np.mean([c['no_noise_accuracy'] for c in group])})
    if all(r['pvalue'] is not None for r in rows):
        adjusted = holm_adjusted_pvalues([r['pvalue'] for r in rows])
        for row, p in zip(rows, adjusted):
            row['holm_pvalue'] = p
    with (args.directory/'summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {'status':'verified', 'cases':len(cases), 'max_budget_error':max_budget_error,
        'all_linear_baselines_exact':all(c['linear_exact'] for c in cases),
        'summary_sha256':hashlib.sha256((args.directory/'summary.csv').read_bytes()).hexdigest(),
        'checks':['source identity','case completeness','allocation feasibility',
                  'linear baseline KKT','constant-LR uniformity','cross-budget reference consistency','public-only checkpoint selection',
                  'directional derivatives','SAA descent','path means']}
    s.save(args.directory/'verification.json', report)
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
