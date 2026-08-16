#!/usr/bin/env python3
# Metropolis-Hastings MCMC from scratch, applied to Bayesian linear regression.
# numpy only. Samples the posterior of (intercept, slope, log-sigma).
import numpy as np


def make_log_posterior(x, y, prior_sd=10.0):
    x = np.asarray(x, float); y = np.asarray(y, float); n = x.size
    def log_post(p):
        a, b, ls = p
        sigma = np.exp(ls)
        pred = a + b * x
        ll = -0.5 * np.sum((y - pred) ** 2) / sigma**2 - n * ls
        log_prior = -0.5 * (a**2 + b**2) / prior_sd**2 - 0.5 * ls**2 / 4.0
        return ll + log_prior
    return log_post


def metropolis(log_post, init, step, iters=20000, seed=0):
    # Random-walk Metropolis-Hastings. Returns the chain and acceptance rate.
    rng = np.random.default_rng(seed)
    cur = np.array(init, float); lp = log_post(cur)
    d = cur.size; chain = np.zeros((iters, d)); accepted = 0
    step = np.atleast_1d(step)
    for i in range(iters):
        prop = cur + rng.normal(0, step, size=d)
        lpp = log_post(prop)
        if np.log(rng.random()) < lpp - lp:
            cur, lp = prop, lpp; accepted += 1
        chain[i] = cur
    return chain, accepted / iters


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.uniform(-3, 3, 80); y = 2.0 - 1.5 * x + rng.normal(0, 1, 80)
    lp = make_log_posterior(x, y)
    chain, acc = metropolis(lp, [0, 0, 0], [0.15, 0.08, 0.08], 15000)
    post = chain[3000:]
    print("posterior means (a,b,sigma):", post[:, 0].mean().round(2),
          post[:, 1].mean().round(2), np.exp(post[:, 2]).mean().round(2), "acc:", round(acc, 2))
