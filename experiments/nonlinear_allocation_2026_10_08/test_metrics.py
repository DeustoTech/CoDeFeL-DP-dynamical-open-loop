"""Independent forward metrics and multi-chunk Monte Carlo evaluator tests."""
import hashlib
from pathlib import Path
from types import SimpleNamespace
import run_search as r

s,np = r.s,r.np


def direct_metrics(theta,x,y,shape,l2):
    d,h,c = shape.d,shape.h,shape.c
    w1 = theta[:d*h].reshape(d,h)
    b1 = theta[d*h:d*h+h]
    w2 = theta[d*h+h:d*h+h+h*c].reshape(h,c)
    b2 = theta[-c:]
    logits = np.logaddexp(0.,x@w1+b1)@w2+b2
    loss = np.mean(np.logaddexp.reduce(logits,axis=1)-logits[np.arange(len(y)),y])
    return loss+.5*l2*np.sum(theta**2),np.mean(logits.argmax(1)==y)


def main():
    rng = np.random.default_rng(902102)
    shape = s.f.MLPShape(3,4,2)
    x,y = rng.normal(size=(17,3)),rng.integers(0,2,17)
    parameters = rng.normal(size=(3,shape.p))
    loss,acc = s.f.mlp_loss_acc(parameters,x,y,shape,.013,batch=5)
    expected = np.array([direct_metrics(theta,x,y,shape,.013) for theta in parameters])
    np.testing.assert_allclose(loss,expected[:,0],rtol=2e-14,atol=2e-14)
    np.testing.assert_array_equal(acc,expected[:,1])
    eta = np.array([.1,.2,.13,.15])
    batches = [rng.choice(17,6,replace=False) for _ in eta]
    theta0 = parameters[0]
    masks = np.ones((len(eta),shape.p))
    ref = s.f.compute_reference(theta0,x,y,batches,masks,eta,shape,.013,1.)
    p = SimpleNamespace(shape=shape,T=4,eta=eta,batches=batches,active_masks=masks,
        theta_design_ref=ref,theta_ref=ref,x_design=x,y_design=y,x_train=x,y_train=y,
        x_test=x,y_test=y,l2=.013,clip=1.,sensitivity=1./3,rho_max=.5,q_min=1e-5)
    schedules = {'a':np.full(4,.125),'b':np.array([.08,.1,.14,.18])}
    actual = s.evaluate(p,schedules,92919,19,public=False,metrics=True)
    rng = np.random.default_rng(92919)
    draws = [rng.normal(size=(4,m,shape.p)) for m in (16,3)]
    max_error = 0.
    for name,q in schedules.items():
        errors,losses,accuracies = [],[],[]
        for xi in draws:
            for j in range(xi.shape[1]):
                theta = theta0.copy()
                for t,idx in enumerate(batches):
                    grad = s.f.mlp_grad(theta,x[idx],y[idx],shape,p.l2,p.clip)[0]
                    theta -= eta[t]*grad+p.sensitivity*eta[t]/np.sqrt(2*q[t])*xi[t,j]
                errors.append(np.sum((theta-ref[-1])**2))
                l,a = direct_metrics(theta,x,y,shape,p.l2)
                losses.append(l)
                accuracies.append(a)
        np.testing.assert_allclose(actual[name]['path_errors'],errors,atol=2e-13,rtol=2e-13)
        np.testing.assert_allclose(actual[name]['loss'],np.mean(losses),atol=2e-13,rtol=2e-13)
        np.testing.assert_allclose(actual[name]['accuracy'],np.mean(accuracies),atol=2e-15,rtol=0.)
        max_error = max(max_error,float(np.max(np.abs(np.array(actual[name]['path_errors'])-errors))))
    report = {'status':'passed','source_sha256':r.source_hashes(),
        'test_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'tests':['manual forward pass, cross-entropy and per-model accuracy',
            'nondivisible test minibatches','independent one-path simulation across Gaussian chunks',
            'mean individual-model accuracy without ensembling','method and path indexing'],
        'max_path_error_discrepancy':max_error}
    s.save(r.HERE/'metrics_verification.json',report)
    print(report)


if __name__=='__main__':
    main()
