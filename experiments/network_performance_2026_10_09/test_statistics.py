"""Independent exact trigonometric check of the nine-d.f. Student tails."""
import math

import numpy as np
import summarize as s


def tail_df9(x):
    # With t=3 tan(theta), the Student density is proportional to cos(theta)^8.
    theta = math.atan(abs(x)/3)
    integral = theta
    for n in (2, 4, 6, 8):
        integral = math.sin(theta)*math.cos(theta)**(n-1)/n + (n-1)*integral/n
    normalization = math.gamma(5)/(math.sqrt(math.pi)*math.gamma(4.5))
    return max(0., 1-2*normalization*integral)


def main():
    lower, upper = 2., 3.
    for _ in range(60):
        middle = (lower+upper)/2
        if tail_df9(middle) > .05:
            lower = middle
        else:
            upper = middle
    critical = (lower+upper)/2
    np.testing.assert_allclose(s.student_t_quantile(.975,9), critical,
                               rtol=0, atol=2e-13)
    for x in (0., .1, 1., critical, 5., 10.):
        np.testing.assert_allclose(s.student_t_two_sided_pvalue(x,9), tail_df9(x),
                                   rtol=0, atol=2e-14)
    a = np.array([.1,.2,.12,.18,.3,.25,.11,.16,.19,.23])
    r = s.interval(a)
    se = math.sqrt(sum((a-a.mean())**2)/9)/math.sqrt(10)
    np.testing.assert_allclose([r["mean"],r["ci_low"],r["ci_high"],r["pvalue"]],
        [a.mean(),a.mean()-critical*se,a.mean()+critical*se,
         tail_df9(a.mean()/se)], rtol=0, atol=2e-13)
    np.testing.assert_allclose(s.holm_adjusted_pvalues([.01,.04,.02]), [.03,.04,.04])
    s.s.save(s.s.HERE / "statistics_tests.json", {"status":"passed",
        "tests":["nine-d.f. quantile", "exact trigonometric Student tail",
                 "ten-seed standard error and interval", "Holm adjustment"]})
    print("Independent seed-level statistical checks passed")


if __name__ == "__main__":
    main()
