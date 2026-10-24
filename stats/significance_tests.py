"""Statistical Significance Suite for Governance Evaluation.

Provides:
- Bootstrap 95% Confidence Intervals (BCa method for small samples)
- Paired McNemar's Exact Test
- Cohen's Kappa for Inter-Annotator Agreement
- Effect Size metrics (Odds Ratio, Cohen's g)
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from scipy import stats


def compute_bootstrap_ci(
    outcomes: List[int],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    random_seed: int = 42,
) -> Dict[str, float]:
    """
    Computes empirical bootstrap confidence interval for a binary outcome series
    (1=correct, 0=incorrect).

    Uses BCa (bias-corrected and accelerated) approximation for improved coverage
    on small samples (N < 30). Falls back to percentile method if BCa computation
    encounters numerical issues.
    """
    if not outcomes:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "std_err": 0.0}

    arr = np.array(outcomes, dtype=float)
    n = len(arr)
    observed_mean = float(np.mean(arr))
    rng = np.random.default_rng(random_seed)

    # Generate bootstrap resamples
    boot_means = np.array([
        np.mean(rng.choice(arr, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])

    alpha = (1.0 - ci) / 2.0

    # Attempt BCa correction
    try:
        # Bias correction factor z0
        prop_below = np.mean(boot_means < observed_mean)
        if prop_below == 0.0:
            prop_below = 1.0 / (2 * n_bootstrap)
        elif prop_below == 1.0:
            prop_below = 1.0 - 1.0 / (2 * n_bootstrap)
        z0 = stats.norm.ppf(prop_below)

        # Acceleration factor a (jackknife estimate)
        jackknife_means = np.array([
            np.mean(np.delete(arr, i)) for i in range(n)
        ])
        jk_mean = np.mean(jackknife_means)
        jk_diff = jk_mean - jackknife_means
        numerator = np.sum(jk_diff ** 3)
        denominator = 6.0 * (np.sum(jk_diff ** 2) ** 1.5)

        if abs(denominator) > 1e-12:
            a = numerator / denominator
        else:
            a = 0.0

        # BCa adjusted percentiles
        z_alpha_lower = stats.norm.ppf(alpha)
        z_alpha_upper = stats.norm.ppf(1.0 - alpha)

        adj_lower = stats.norm.cdf(z0 + (z0 + z_alpha_lower) / (1 - a * (z0 + z_alpha_lower)))
        adj_upper = stats.norm.cdf(z0 + (z0 + z_alpha_upper) / (1 - a * (z0 + z_alpha_upper)))

        ci_lower = float(np.percentile(boot_means, adj_lower * 100))
        ci_upper = float(np.percentile(boot_means, adj_upper * 100))

        # Sanity check — BCa can produce inverted intervals on degenerate data
        if ci_lower > ci_upper:
            ci_lower, ci_upper = ci_upper, ci_lower

    except (ValueError, ZeroDivisionError, FloatingPointError):
        # Fallback to standard percentile method
        ci_lower = float(np.percentile(boot_means, alpha * 100))
        ci_upper = float(np.percentile(boot_means, (1.0 - alpha) * 100))

    std_err = float(np.std(boot_means))

    return {
        "mean": round(observed_mean, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "std_err": round(std_err, 4),
    }


def mcnemar_test(outcomes_a: List[int], outcomes_b: List[int]) -> Dict[str, Any]:
    """
    Computes McNemar's test for paired nominal data (System A vs System B
    on identical test cases).

    Contingency table:
                    System B Correct (1) | System B Incorrect (0)
    System A Correct (1)   |       a (both 1)    |       b (A=1, B=0)
    System A Incorrect (0) |       c (A=0, B=1)  |       d (both 0)

    Uses exact binomial test for small discordant counts (b+c < 25),
    Edwards continuity-corrected chi-square otherwise.
    """
    assert len(outcomes_a) == len(outcomes_b), "Outcomes lists must be of equal length."

    a = sum(1 for x, y in zip(outcomes_a, outcomes_b) if x == 1 and y == 1)
    b = sum(1 for x, y in zip(outcomes_a, outcomes_b) if x == 1 and y == 0)
    c = sum(1 for x, y in zip(outcomes_a, outcomes_b) if x == 0 and y == 1)
    d = sum(1 for x, y in zip(outcomes_a, outcomes_b) if x == 0 and y == 0)

    discordant = b + c
    if discordant == 0:
        p_value = 1.0
        statistic = 0.0
    elif discordant < 25:
        # Exact two-tailed binomial test under H0: p = 0.5
        k = min(b, c)
        p_value = 2.0 * stats.binom.cdf(k, discordant, 0.5)
        p_value = min(1.0, float(p_value))
        statistic = float(abs(b - c))
    else:
        # Edwards continuity-corrected McNemar Chi-Square
        statistic = float((abs(b - c) - 1.0) ** 2 / (b + c))
        p_value = float(stats.chi2.sf(statistic, df=1))

    is_significant = p_value < 0.05

    return {
        "contingency_table": {"both_correct": a, "a_only": b, "b_only": c, "both_incorrect": d},
        "b_discordant_a_wins": b,
        "c_discordant_b_wins": c,
        "statistic": round(statistic, 4),
        "p_value": round(p_value, 5),
        "is_statistically_significant": is_significant,
    }


def compute_cohens_kappa(rater1: List[int], rater2: List[int]) -> Dict[str, float]:
    """
    Computes Cohen's Kappa for inter-annotator agreement between human and
    automated judge.

    Returns kappa, observed agreement, and expected agreement.
    Handles degenerate cases (all-same labels) gracefully.
    """
    assert len(rater1) == len(rater2), "Rater arrays must be of equal length."
    n = len(rater1)
    if n == 0:
        return {"kappa": 1.0, "observed_agreement": 1.0, "expected_agreement": 1.0}

    # Observed agreement
    po = sum(1 for x, y in zip(rater1, rater2) if x == y) / n

    # Expected agreement by chance
    p1_pos = sum(rater1) / n
    p1_neg = 1.0 - p1_pos
    p2_pos = sum(rater2) / n
    p2_neg = 1.0 - p2_pos

    pe = (p1_pos * p2_pos) + (p1_neg * p2_neg)

    if pe == 1.0:
        kappa = 1.0 if po == 1.0 else 0.0
    else:
        kappa = (po - pe) / (1.0 - pe)

    return {
        "kappa": round(kappa, 4),
        "observed_agreement": round(po, 4),
        "expected_agreement": round(pe, 4),
    }
