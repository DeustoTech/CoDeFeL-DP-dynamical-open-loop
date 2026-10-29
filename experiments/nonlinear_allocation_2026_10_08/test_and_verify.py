"""Independent finite-difference tests and recorded-case replay checks."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import run_search as r
import refine_geometric as g
from experiment_utils import student_t_cdf, student_t_quantile, holm_adjusted_pvalues

np, s = r.np, r.s


def unit_tests():
    for value in (-5.,-1.,0.,1.,5.):
        np.testing.assert_allclose(student_t_cdf(value,1),.5+math.atan(value)/math.pi,atol=2e-14)
        np.testing.assert_allclose(student_t_cdf(value,2),.5+value/(2*math.sqrt(value*value+2)),atol=2e-14)
    np.testing.assert_allclose(student_t_quantile(.975,9),2.262157162798205,rtol=1e-13)
    np.testing.assert_allclose(holm_adjusted_pvalues([.01,.04,.03,.002]),[.03,.06,.06,.008])
    minimizer,value = g.golden(lambda x:(x-.237)**2,-1.,1.)
    assert abs(minimizer-.237)<1e-4 and value<1e-8
    rng = np.random.default_rng(910901)
    shape = s.f.MLPShape(3, 3, 2)
    x, y = rng.normal(size=(15,3)), rng.integers(0,2,15)
    eta = np.array([.1,.23,.17,.31])
    batches = [rng.choice(15,7,replace=False) for _ in eta]
    theta0 = s.f.init_theta(shape,rng)
    masks = np.ones((len(eta),shape.p))
    ref = s.f.compute_reference(theta0,x,y,batches,masks,eta,shape,1e-4,.8)
    p = SimpleNamespace(T=len(eta),shape=shape,x_design=x,y_design=y,
        x_train=x,y_train=y,x_test=x,y_test=y,batches=batches,
        active_masks=masks,theta_design_ref=ref,theta_ref=ref,eta=eta,
        l2=1e-4,clip=.8,sensitivity=1.6/7,rho_max=.5,q_min=1e-5)
    matrices = []
    eps = 1e-5
    for t,idx in enumerate(batches):
        columns = []
        for e in eps*np.eye(shape.p):
            up = s.f.mlp_grad(ref[t]+e,x[idx],y[idx],shape,p.l2,p.clip)[0]
            down = s.f.mlp_grad(ref[t]-e,x[idx],y[idx],shape,p.l2,p.clip)[0]
            columns.append((up-down)/(2*eps))
        jac = np.column_stack(columns)
        v = rng.normal(size=(3,shape.p))
        vjp = s.f.mlp_grad_vjp(np.broadcast_to(ref[t],v.shape),v,x[idx],y[idx],shape,p.l2,p.clip)
        np.testing.assert_allclose(vjp,v@jac,atol=3e-9,rtol=3e-7)
        matrices.append(np.eye(shape.p)-eta[t]*jac)
    coefficients = []
    for t in range(p.T):
        product = np.eye(shape.p)
        for j in range(t+1,p.T):
            product = matrices[j]@product
        coefficients.append(.5*(p.sensitivity*eta[t])**2*np.sum(product**2))
    schedules,a = r.analytic_schedules(p)
    np.testing.assert_allclose(a,coefficients,rtol=3e-9,atol=1e-11)
    independent_decay = np.array([eta[t]*np.prod(1-eta[t+1:]*p.l2) for t in range(p.T)])
    np.testing.assert_allclose(schedules['Decay'],s.allocate(independent_decay,.5,1e-5),rtol=1e-13)
    np.testing.assert_allclose(s.allocate(np.array([.001,1.,2.]),1.,.1),[.1,.3,.6])
    derivative_errors = s.check_derivatives(p,910901)
    xi = rng.normal(size=(p.T,32,shape.p))
    candidates,history,feas = r.from_uniform(p,xi,schedules['LR'],20)
    assert all(h['next_objective'] <= h['objective']+1e-10 for h in history)
    for q in candidates:
        assert abs(q.sum()-.5)<1e-12 and q.min()>=p.q_min
    for q in [schedules['LR'],schedules['Decay'],schedules['Linearized'],candidates[-1]]:
        xi_eval = np.random.default_rng(900).normal(size=(p.T,9,shape.p))
        expected = s.f.objective_only(p,q,xi_eval,theta0)
        actual = s.evaluate(p,{'q':q},900,9,public=True,metrics=False)['q']['error']
        np.testing.assert_allclose(actual,expected,rtol=1e-13,atol=1e-14)
    return {'status':'passed','tests':['dense finite-difference VJP',
        'chronological full-matrix coefficients','independent regularization-product allocation',
        'active-floor normalization','adjoint finite differences','Armijo mirror feasibility and descent',
        'independent evaluator agreement','analytic Student-t checks and Holm correction',
        'golden-section minimization'], 'derivative_errors':derivative_errors,
        'max_budget_error':feas}


def verify_case(path,manifest,replay):
    c = json.loads(path.read_text())
    rho = c['rho']
    for q in c['schedules'].values():
        assert abs(np.sum(q)-rho)<1e-12 and min(q)>=1e-5
    assert c['source_sha256'] == manifest['source_sha256']
    assert c['validation'][c['selected_checkpoint']] == min(c['validation'].values())
    assert c['geometric_validation'][c['selected_geometric']] == min(c['geometric_validation'].values())
    assert set(c['geometric_validation']) == {f'beta{b:g}' for b in manifest['geometric_grid']}
    assert all(h['next_objective'] <= h['objective']+1e-9*max(1.,h['objective']) for h in c['pb_history'])
    np.testing.assert_array_equal(c['schedules']['PB-AMD'],c['candidate_schedules'][c['selected_checkpoint']])
    for scope in ['public']+(['private'] if c['phase']=='confirmation' else []):
        for value in c[scope].values():
            assert len(value['path_errors']) == manifest['eval_paths']
            np.testing.assert_allclose(value['error'],np.mean(value['path_errors']),rtol=1e-14)
        for name in r.BASELINES:
            expected = 100*(1-c[scope]['PB-AMD']['error']/c[scope][name]['error'])
            np.testing.assert_allclose(c[scope+'_gains'][name],expected,atol=1e-12,rtol=1e-12)
    assert max(c['gradient_check_relative_errors']) < 2e-6
    result = {'case':path.name,'status':'passed'}
    refined_path = path.parent/'refined_geometric'/path.name
    refined = json.loads(refined_path.read_text()) if refined_path.exists() else None
    if refined:
        assert refined['case_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert refined['script_sha256'] == hashlib.sha256(Path(g.__file__).read_bytes()).hexdigest()
        assert refined['source_sha256'] == r.source_hashes()
        assert abs(sum(refined['schedule'])-rho)<1e-12 and min(refined['schedule'])>=1e-5
        assert refined['validation_error']<=min(c['geometric_validation'].values())+1e-12
        assert any(t['beta']==refined['beta'] and t['validation_error']==refined['validation_error'] for t in refined['trace'])
        for scope in ['public']+(['private'] if 'private' in c else []):
            np.testing.assert_allclose(refined[scope]['error'],np.mean(refined[scope]['path_errors']),rtol=1e-14)
            np.testing.assert_allclose(refined[scope+'_pb_gain'],100*(1-c[scope]['PB-AMD']['error']/refined[scope]['error']),atol=1e-12)
        result['refined_geometric_checked'] = True
    if replay:
        p = r.make_problem(r.Setting(**c['setting']),c['seed'],rho)
        schedules,a = r.analytic_schedules(p)
        np.testing.assert_array_equal(a,c['linear_coefficients'])
        for name,q in schedules.items():
            np.testing.assert_array_equal(q,c['schedules'][name])
        beta = float(c['selected_geometric'].removeprefix('beta'))
        expected = s.allocate(p.eta*np.exp(beta*np.linspace(0,1,p.T)),rho,p.q_min)
        np.testing.assert_array_equal(expected,c['schedules']['Tuned-geometric'])
        replay_errors = []
        for scope,offset in [('public',186000)]+([('private',187000)] if 'private' in c else []):
            pp = p if scope=='public' else replace(p,x_design=p.x_train,y_design=p.y_train,theta_design_ref=p.theta_ref)
            xi = np.random.default_rng(c['seed']+offset).normal(size=(p.T,16,p.shape.p))
            for name,q in c['schedules'].items():
                value = s.f.objective_only(pp,np.array(q),xi,pp.theta_design_ref[0])
                expected = np.mean(c[scope][name]['path_errors'][:16])
                err = abs(value-expected)/max(1.,abs(expected))
                assert err<1e-12
                replay_errors.append(err)
            if refined:
                expected_q = s.allocate(p.eta*np.exp(refined['beta']*np.linspace(0,1,p.T)),rho,p.q_min)
                np.testing.assert_array_equal(expected_q,refined['schedule'])
                actual = s.f.objective_only(pp,expected_q,xi,pp.theta_design_ref[0])
                expected = np.mean(refined[scope]['path_errors'][:16])
                err = abs(actual-expected)/max(1.,abs(expected))
                assert err<1e-12
                replay_errors.append(err)
        xi = np.random.default_rng(c['seed']+184000).normal(size=(p.T,manifest['opt_paths'],p.shape.p))
        q = np.array(c['schedules']['PB-AMD'])
        value,grad = s.f.objective_and_grad(p,q,xi,p.theta_design_ref[0])
        k = int(c['selected_checkpoint'][1:])
        expected = c['pb_history'][k]['objective'] if k<len(c['pb_history']) else c['pb_history'][-1]['next_objective']
        np.testing.assert_allclose(value,expected,rtol=1e-12,atol=1e-12)
        result.update(max_replay_error=max(replay_errors),
            selected_relative_gap=float((grad@q-p.q_min*grad.sum()-(rho-p.T*p.q_min)*grad.min())/value))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory',type=Path,nargs='?')
    parser.add_argument('--replay',action='store_true')
    parser.add_argument('--workers',type=int,default=4)
    args = parser.parse_args()
    report = unit_tests()
    report['test_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if args.directory:
        manifest = json.loads((args.directory/'manifest.json').read_text())
        if any(Path(path).name=='run_fast_head.py' for path in manifest['source_sha256']):
            import run_fast_head
        elif any(Path(path).name=='run_extended.py' for path in manifest['source_sha256']):
            import run_extended
        assert manifest['source_sha256'] == r.source_hashes()
        paths = sorted(args.directory.glob('*_seed*_rho*.json'))
        expected = {f'{name}_seed{seed}_rho{float(rho):g}.json'
            for name in manifest['settings'].split(',') for seed in manifest['seeds'].split(',')
            for rho in manifest['rhos'].split(',')}
        assert {p.name for p in paths} == expected
        rows = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(verify_case,p,manifest,args.replay) for p in paths]
            for future in as_completed(futures):
                result = future.result()
                rows.append(result)
                print(result,flush=True)
        report.update(cases=len(rows),checks=sorted(rows,key=lambda row:row['case']),replay=args.replay)
    s.save((args.directory or r.HERE)/'verification.json',report)
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
