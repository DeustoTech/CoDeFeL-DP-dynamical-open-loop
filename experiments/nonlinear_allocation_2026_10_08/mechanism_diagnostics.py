"""Compare linear-model predictions with saved nonlinear public outcomes."""
import argparse
import hashlib
import json
from pathlib import Path
import run_search as r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    parser.add_argument('--independent',action='store_true')
    args = parser.parse_args()
    rows = []
    for path in sorted(args.directory.glob('*_seed*_rho*.json')):
        c = json.loads(path.read_text())
        if args.independent:
            evaluation = json.loads((args.directory/'independent_evaluation'/path.name).read_text())
            assert evaluation['case_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
            c['public'] = evaluation['public']
            c['public_gains']['Linearized'] = 100*(1-c['public']['PB-AMD']['error']/c['public']['Linearized']['error'])
        a = r.np.array(c['linear_coefficients'])
        linear_predictions = {name:float(r.np.sum(a/r.np.array(q))) for name,q in c['schedules'].items()}
        best_linear = linear_predictions['Linearized']
        assert all(v>=best_linear-1e-10 for v in linear_predictions.values())
        row = {'case':path.name,'case_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'linear_model_error_predictions':linear_predictions,
            'nonlinear_public_errors':{name:v['error'] for name,v in c['public'].items()},
            'linear_model_PB_excess_percent':100*(linear_predictions['PB-AMD']/best_linear-1),
            'nonlinear_PB_reduction_vs_linear_percent':c['public_gains']['Linearized'],
            'PB_budget_shares_by_third':[float(r.np.sum(x)/c['rho']) for x in r.np.array_split(c['schedules']['PB-AMD'],3)],
            'linear_budget_shares_by_third':[float(r.np.sum(x)/c['rho']) for x in r.np.array_split(c['schedules']['Linearized'],3)],
            'pure_regularization_first_to_last_weight_ratio':float((1-c['setting']['eta']*c['setting']['l2'])**(c['setting']['horizon']-1))}
        rows.append(row)
    assert rows
    r.s.save(args.directory/('independent_mechanism_diagnostics.json' if args.independent else 'mechanism_diagnostics.json'),{
        'scope':'The quadratic reference-linearized model versus fresh nonlinear public evaluation; no new simulation.',
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'cases':rows})
    print(json.dumps([{k:row[k] for k in ('case','linear_model_PB_excess_percent',
        'nonlinear_PB_reduction_vs_linear_percent','PB_budget_shares_by_third')} for row in rows],indent=2))


if __name__=='__main__':
    main()
