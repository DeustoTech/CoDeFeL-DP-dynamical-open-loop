"""Check prediction metrics independently and match the original evaluator."""
from types import SimpleNamespace

import run_study as s
import numpy as np


def direct_metrics(theta, x, y, shape):
    # Explicit single-network forward pass, independent of the batched metric code.
    d, h, c = shape.d, shape.h, shape.c
    a = d * h
    w1, b1 = theta[:a].reshape(d, h), theta[a:a+h]
    w2, b2 = theta[a+h:-c].reshape(h, c), theta[-c:]
    logits = np.logaddexp(0., x @ w1 + b1) @ w2 + b2
    z = logits - np.max(logits, axis=1, keepdims=True)
    ce = np.mean(np.log(np.exp(z).sum(axis=1)) - z[np.arange(len(y)), y])
    acc = np.mean(logits.argmax(axis=1) == y)
    return float(ce), float(acc)


def main():
    rng = np.random.default_rng(81191)
    shape = s.f.MLPShape(3, 4, 2)
    x = rng.normal(size=(13, 3))
    y = rng.integers(0, 2, size=13)
    theta = rng.normal(scale=.2, size=(5, shape.p))
    ce, acc = s.f.mlp_loss_acc(theta, x, y, shape, 0., batch=4)
    independent = np.array([direct_metrics(t, x, y, shape) for t in theta])
    np.testing.assert_allclose(ce, independent[:, 0], atol=1e-14, rtol=0.)
    np.testing.assert_array_equal(acc, independent[:, 1])
    reg, _ = s.f.mlp_loss_acc(theta, x, y, shape, .17)
    np.testing.assert_allclose(reg-ce, .085 * (theta**2).sum(axis=1), atol=1e-14, rtol=0.)
    batches = [np.array([0, 1, 2]), np.array([3, 4, 5]), np.array([6, 7, 8])]
    mask = np.ones((3, shape.p))
    mask[1, :2] = 0.
    eta = np.array([.04, .06, .03])
    reference = s.f.compute_reference(theta[0], x, y, batches, mask, eta, shape, .01, 1.)
    p = SimpleNamespace(shape=shape, T=3, x_train=x, y_train=y, x_test=x,
                        y_test=y, batches=batches, active_masks=mask, l2=.01,
                        sensitivity=.2, clip=1., eta=eta,
                        theta_design_ref=reference, theta_ref=reference)
    schedules = {"first": [.1, .2, .3], "second": [.3, .2, .1]}
    new = s.evaluate(p, schedules, 981, paths=7, chunk=3)
    old = s.f.simulate_eval_many(p, list(schedules.values()), theta[0], 981, paths=7, chunk=3)
    for method, archived in zip(schedules, old):
        for k, a in (("accuracy", "test_acc"), ("regularized_loss", "test_loss"), ("tracking", "terminal_mean")):
            np.testing.assert_allclose(np.mean(new[method][k]), archived[a], atol=1e-13, rtol=0.)
        assert len(new[method]["accuracy"]) == 7
    report = {"status": "passed", "tests": ["independent single-network forward pass",
        "unregularized cross-entropy", "exact regularization removal", "old evaluator agreement",
        "partial chunk", "multiple schedules and masked dynamics", "per-model rather than ensemble metrics"]}
    s.save(s.HERE / "unit_tests.json", report)
    print(report)


if __name__ == "__main__":
    main()
