"""Trust layer: calibrated confidence, refuse-to-guess, per-camera trust table.

* Temperature scaling (Guo et al. 2017) of the ordinal logits, fitted on val only.
* Class-conditional (Mondrian) split conformal prediction for referable vs non-referable:
  for each true class the prediction set contains it with probability >= 1 - alpha,
  so the miss rate on referable eyes is controlled, not just the average error.
  If the set is {non-ref, ref} (or empty) the system refuses to guess -> human review.
* Camera-domain trust table: per-domain miss rate at the grade 1/2 boundary, abstention,
  calibration error. Domains are the dataset/camera proxies from the manifests.
"""
import numpy as np
from scipy.optimize import minimize_scalar


def _sig(z):
    return 1 / (1 + np.exp(-z))


def fit_temperature(logits, y, K=5):
    """Single temperature for all ordinal thresholds, minimising the ordinal NLL on val."""
    T = (np.asarray(y).reshape(-1, 1) > np.arange(K - 1)).astype(float)

    def nll(logt):
        p = np.clip(_sig(logits / np.exp(logt)), 1e-7, 1 - 1e-7)
        return -(T * np.log(p) + (1 - T) * np.log(1 - p)).mean()

    r = minimize_scalar(nll, bounds=(-3, 3), method="bounded")
    return float(np.exp(r.x))


def calibrated_probs(logits, temperature):
    P = _sig(logits / temperature)
    return np.minimum.accumulate(P, axis=1)            # monotone P(>0) >= P(>1) >= ...


def conformal_fit(p_ref, y_ref, alpha):
    """Class-conditional thresholds on the nonconformity 1 - p(true class)."""
    p_ref, y_ref = np.asarray(p_ref, float), np.asarray(y_ref, bool)
    q = {}
    for cls, p_true in ((1, p_ref[y_ref]), (0, 1 - p_ref[~y_ref])):
        s = np.sort(1 - p_true)
        n = len(s)
        k = min(n - 1, int(np.ceil((n + 1) * (1 - alpha))) - 1)
        q[cls] = float(s[k])
    return q


def conformal_predict(p_ref, q):
    """Returns decision per eye: 'referable', 'non-referable' or 'refer-to-human'."""
    p_ref = np.asarray(p_ref, float)
    in1 = (1 - p_ref) <= q[1]
    in0 = p_ref <= q[0]
    out = np.full(len(p_ref), "refer-to-human", dtype=object)
    out[in1 & ~in0] = "referable"
    out[in0 & ~in1] = "non-referable"
    return out


def decision_metrics(decision, y_ref):
    y_ref = np.asarray(y_ref, bool)
    auto = decision != "refer-to-human"
    pred = decision == "referable"
    missed = (~pred & y_ref & auto).sum()          # referable eyes auto-cleared as healthy
    return {"n": int(len(y_ref)), "abstain_rate": float(1 - auto.mean()),
            "auto_sensitivity": float((pred & y_ref & auto).sum() / max((y_ref & auto).sum(), 1)),
            "auto_specificity": float((~pred & ~y_ref & auto).sum() / max((~y_ref & auto).sum(), 1)),
            "missed_referable_rate": float(missed / max(y_ref.sum(), 1)),
            "system_sensitivity": float(1 - missed / max(y_ref.sum(), 1))}  # humans review abstentions


def ece(y, p, bins=10):
    y, p = np.asarray(y, float), np.asarray(p, float)
    idx = np.clip((p * bins).astype(int), 0, bins - 1)
    return float(sum((idx == k).mean() * abs(y[idx == k].mean() - p[idx == k].mean())
                     for k in range(bins) if (idx == k).any()))


def domain_table(domains, y, p_ref, decision, thr):
    """Per camera-domain trust figures on held-out data."""
    domains, y = np.asarray(domains), np.asarray(y).astype(int)
    rows = []
    for d in np.unique(domains):
        m = domains == d
        yr = y[m] >= 2
        bnd = m & np.isin(y, [1, 2])                   # the clinically hardest boundary
        rows.append({
            "domain": d, "n": int(m.sum()), "referable_prevalence": float(yr.mean()),
            "miss_rate_at_threshold": float(((p_ref[m] <= thr) & yr).sum() / max(yr.sum(), 1)),
            "miss_rate_grade1v2_boundary": float(((p_ref[bnd] <= thr) & (y[bnd] == 2)).sum() / max((y[bnd] == 2).sum(), 1)),
            "abstain_rate": float((decision[m] == "refer-to-human").mean()),
            "ece": ece(yr, p_ref[m]),
        })
    return rows
