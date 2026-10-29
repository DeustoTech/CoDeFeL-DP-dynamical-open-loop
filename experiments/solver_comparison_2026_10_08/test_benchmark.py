"""Tests for oracle equivalence, derivatives, diagnostics and update bookkeeping."""
import json
import time
import unittest

import benchmark as b

np = b.np


class UnitTests(unittest.TestCase):
    def test_gap_matches_explicit_vertices(self):
        rng = np.random.default_rng(7001)
        for n in (3, 17, 120):
            rho, floor = .5, 1e-5
            q = b.utils.project_lower_bounded_simplex(rng.normal(size=n), rho, floor)
            g = rng.normal(size=n)
            vertices = np.full((n, n), floor) + (rho - n * floor) * np.eye(n)
            direct = max(float(g @ (q - z)) for z in vertices)
            self.assertAlmostEqual(b.utils.variational_gap(q, g, rho, floor), direct, places=12)

    def test_scaled_initial_displacement(self):
        n, rho, floor, alpha = 30, .5, 1e-5, 1e-7
        q = np.full(n, rho / n)
        gradient = np.linspace(-1., 1., n)
        pb = b.update(q, gradient, "PB-AMD", alpha, rho, floor)
        pg = b.update(q, gradient, "PGD", alpha * rho / n, rho, floor)
        # The production entropic normalizer has absolute budget tolerance 1e-13.
        np.testing.assert_allclose(pb, pg, atol=2e-13, rtol=0.)

    def test_lq_oracle_bookkeeping(self):
        class Problem:
            T, rho_max, q_min = 8, .5, 1e-5
        class Context:
            problem, task, seed = Problem(), "unit", 0
            xi = np.zeros((8, 1, 1))
            uniform = np.full(8, .5 / 8)
            def fg(self, q):
                a = np.linspace(1., 3., 8)
                return float(np.sum(a / q)), -a / q**2
        for method in b.METHODS:
            result = b.solve(Context(), method, .3, 100, tolerance=1e-6)
            trace = result["trace"]
            self.assertEqual(len(trace), result["updates"] + 1)
            self.assertLessEqual(trace[-1]["relative_gap"], 1e-6)
            for old, new in zip(trace, trace[1:]):
                expected = b.update(np.array(old["q"]), np.array(old["gradient"]),
                                    method, result["alpha"], .5, 1e-5)
                np.testing.assert_allclose(expected, new["q"], atol=1e-14, rtol=0.)
                self.assertGreaterEqual(new["seconds"], old["seconds"])
                self.assertEqual(new["gradient_evaluations"], old["gradient_evaluations"] + 1)


def check_oracles():
    records = []
    for task in b.TASKS:
        ctx = b.make_context(task, 224001 + 1000 * b.TASKS.index(task), paths=8)
        q = ctx.uniform
        f, g = ctx.fg(q)
        fs, gs = ctx.serial_fg(q)
        np.testing.assert_allclose(f, fs, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(g, gs, rtol=2e-10, atol=1e-11)
        np.testing.assert_allclose(f, ctx.objective(q), rtol=1e-12, atol=1e-12)
        rng = np.random.default_rng(8001)
        errors = []
        for _ in range(2):
            direction = rng.normal(size=len(q))
            direction -= direction.mean()
            direction /= np.linalg.norm(direction)
            epsilon = 1e-3 * q.min()
            fd = (ctx.objective(q + epsilon * direction) - ctx.objective(q - epsilon * direction)) / (2 * epsilon)
            analytic = float(g @ direction)
            error = abs(fd - analytic) / max(1., abs(fd), abs(analytic))
            assert error < 3e-5, (task, error)
            errors.append(error)
        records.append({"task": task, "path_workers_vs_serial_gradient_max_error": float(np.max(np.abs(g - gs))),
                        "directional_relative_errors": errors})
        print(task, records[-1], flush=True)
    b.save(b.HERE / "unit_test_report.json", {"status": "passed", "tests": records,
                                             "source_sha256": b.sources()})


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(UnitTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    assert result.wasSuccessful()
    check_oracles()
