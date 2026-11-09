# Self-check of the weight-conditioned sampler and the stratification algebra of
# src/strat_test.py. Run: .venv/bin/python tests/test_strat_sampler.py

import math
import torch
from scipy import stats  # type: ignore[import-untyped]

from sbnd.data import ebno_to_sigma
from sbnd.strat_test import binom_pmf, hard_error_prob, sample_w_errors


def check_counts_and_marginals(n: int, w: int, ebno_dB: float, bs: int) -> None:
    sigma = ebno_to_sigma(45 / 63, ebno_dB)
    y, e = sample_w_errors(n, w, sigma, bs)
    assert torch.equal(e.sum(1), torch.full((bs,), w)), "not exactly w negatives"
    assert torch.equal(e, (y.float() < 0).to(torch.int8)), "sign lost in float32"
    z = ((y - 1) / sigma).numpy()
    err = e.bool().numpy()
    a = 1 / sigma
    # truncated standard Gaussians: (-inf, -a) on errors, (-a, inf) elsewhere
    p_err = stats.kstest(z[err], stats.truncnorm(-math.inf, -a).cdf).pvalue
    p_ok = stats.kstest(z[~err][:200_000], stats.truncnorm(-a, math.inf).cdf).pvalue
    # positions uniform: each one is an error w/n of the time
    hits = err.sum(0)
    chi2 = stats.chisquare(hits).pvalue
    print(
        f"{ebno_dB:4.1f} dB w={w}: KS p(err)={p_err:.3f} p(ok)={p_ok:.3f}, "
        f"positions chi2 p={chi2:.3f}"
    )
    assert min(p_err, p_ok, chi2) > 1e-3


def check_mixture_reproduces_channel(ebno_dB: float) -> None:
    """Sum_w P_w x (law of y | w) must be the unconditioned channel: compare the
    marginal of y_0 with N(1, sigma^2) (all-zero codeword)."""
    n, bs = 63, 20_000
    sigma = ebno_to_sigma(45 / 63, ebno_dB)
    P = binom_pmf(n, hard_error_prob(sigma))
    counts = torch.distributions.Multinomial(bs, torch.tensor(P)).sample().long()
    ys = [sample_w_errors(n, w, sigma, int(c))[0] for w, c in enumerate(counts) if c]
    y0 = torch.cat(ys)[:, 0].numpy()
    p = stats.kstest(y0, stats.norm(1, sigma).cdf).pvalue
    print(f"{ebno_dB:4.1f} dB mixture vs channel: KS p={p:.3f}")
    assert p > 1e-3


def check_pmf() -> None:
    P = binom_pmf(63, 1e-4)
    assert abs(sum(P) - 1) < 1e-12
    ref = stats.binom(63, 1e-4).pmf(range(64))
    assert all(abs(a - b) <= 1e-12 * max(b, 1e-300) for a, b in zip(P, ref) if b > 0)


if __name__ == "__main__":
    torch.manual_seed(0)
    check_pmf()
    for snr in (4.0, 9.0, 12.0):
        for w in (1, 3, 7):
            check_counts_and_marginals(63, w, snr, 100_000)
    for snr in (2.0, 5.0):
        check_mixture_reproduces_channel(snr)
    print("ok")
