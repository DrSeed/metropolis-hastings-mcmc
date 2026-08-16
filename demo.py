#!/usr/bin/env python3
# Self-contained demo: Bayesian linear regression by Metropolis-Hastings MCMC,
# from scratch. Instead of a single best-fit line, MCMC gives a full posterior
# distribution over the parameters -- and therefore honest uncertainty.
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mcmc import make_log_posterior, metropolis

RNG = np.random.default_rng(0)
TRUE_A, TRUE_B, TRUE_SIG = 2.0, -1.5, 1.0
BURN = 4000


def main():
    os.makedirs("figures", exist_ok=True); os.makedirs("results", exist_ok=True)
    x = RNG.uniform(-3, 3, 80)
    y = TRUE_A + TRUE_B * x + RNG.normal(0, TRUE_SIG, 80)

    chain, acc = metropolis(make_log_posterior(x, y), [0, 0, 0], [0.15, 0.08, 0.08], 20000)
    post = chain[BURN:]
    a_s, b_s = post[:, 0], post[:, 1]
    # OLS for comparison.
    X = np.column_stack([np.ones_like(x), x]); ols = np.linalg.lstsq(X, y, rcond=None)[0]

    fig, ax = plt.subplots(2, 2, figsize=(12, 9))

    # Panel 1: data with posterior-mean line and a credible band.
    a = ax[0, 0]
    a.scatter(x, y, s=14, color="#4C72B0", alpha=0.7)
    gx = np.linspace(x.min(), x.max(), 100)
    draws = post[RNG.integers(0, len(post), 200)]
    lines = draws[:, 0][:, None] + draws[:, 1][:, None] * gx[None, :]
    a.fill_between(gx, np.percentile(lines, 2.5, 0), np.percentile(lines, 97.5, 0),
                   color="#C44E52", alpha=0.25, label="95% credible band")
    a.plot(gx, a_s.mean() + b_s.mean() * gx, color="#C44E52", lw=2, label="posterior mean")
    a.set_xlabel("x"); a.set_ylabel("y"); a.set_title("Bayesian fit with uncertainty"); a.legend(fontsize=8)

    # Panel 2: trace plots (mixing) for the two coefficients.
    a = ax[0, 1]
    a.plot(a_s, lw=0.5, color="#4C72B0", label="intercept")
    a.plot(b_s, lw=0.5, color="#DD8452", label="slope")
    a.axhline(TRUE_A, color="#4C72B0", ls="--", lw=1); a.axhline(TRUE_B, color="#DD8452", ls="--", lw=1)
    a.set_xlabel("MCMC iteration (post burn-in)"); a.set_ylabel("value")
    a.set_title(f"Trace plots (acceptance = {acc:.2f})"); a.legend(fontsize=8)

    # Panel 3: marginal posteriors with truth and OLS marked.
    a = ax[1, 0]
    a.hist(a_s, bins=40, alpha=0.6, color="#4C72B0", label="intercept posterior", density=True)
    a.hist(b_s, bins=40, alpha=0.6, color="#DD8452", label="slope posterior", density=True)
    a.axvline(TRUE_A, color="#4C72B0", ls="--"); a.axvline(TRUE_B, color="#DD8452", ls="--")
    a.axvline(ols[0], color="k", ls=":", lw=1); a.axvline(ols[1], color="k", ls=":", lw=1, label="OLS")
    a.set_xlabel("parameter value"); a.set_ylabel("density")
    a.set_title("Marginal posteriors (dashed = truth)"); a.legend(fontsize=8)

    # Panel 4: joint posterior of the two coefficients.
    a = ax[1, 1]
    a.scatter(a_s[::5], b_s[::5], s=3, alpha=0.2, color="#8172B3")
    a.axvline(TRUE_A, color="grey", ls="--"); a.axhline(TRUE_B, color="grey", ls="--")
    a.set_xlabel("intercept"); a.set_ylabel("slope")
    r = np.corrcoef(a_s, b_s)[0, 1]
    a.set_title(f"Joint posterior (corr = {r:.2f})")

    fig.suptitle("Bayesian linear regression via Metropolis-Hastings MCMC (synthetic data)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig("figures/demo.png", dpi=120)

    pd.DataFrame([{
        "param": "intercept", "true": TRUE_A, "post_mean": a_s.mean(),
        "post_sd": a_s.std(), "ci_low": np.percentile(a_s, 2.5), "ci_high": np.percentile(a_s, 97.5), "ols": ols[0]},
        {"param": "slope", "true": TRUE_B, "post_mean": b_s.mean(),
         "post_sd": b_s.std(), "ci_low": np.percentile(b_s, 2.5), "ci_high": np.percentile(b_s, 97.5), "ols": ols[1]}
    ]).to_csv("results/summary.csv", index=False)

    print(f"acceptance={acc:.2f}")
    print(f"intercept: post mean {a_s.mean():.2f} (true {TRUE_A}), 95% CI [{np.percentile(a_s,2.5):.2f},{np.percentile(a_s,97.5):.2f}]")
    print(f"slope:     post mean {b_s.mean():.2f} (true {TRUE_B})")
    print("Wrote figures/demo.png and results/summary.csv")


if __name__ == "__main__":
    main()
