"""Certification with many independent votes (rows of ladder_run2.py).  Truth |dw|; direction-free counts.
Phase votes: v6, v6 flip-y, v6 flip-x, frozen_8c, v4, v6 on two parallel lines.  Other evidence: official grad_mag integral (calibrated on the other folds), lasagna cos peaks, surface-prediction crossings.
Rules compared on one-wrap pairs (the community certificate type), precision against coverage, 95% Wilson intervals:
  - k of the 7 phase votes equal the grad_mag count (the certified count is the grad_mag count)
  - the mode of the 7 phase votes is supported by >= k votes (alone, and together with grad_mag)
  - learned certifier: logistic regression on the vote agreement pattern, leave-fold-out
  python ladder_certify2.py ROWS... (env EXCL=1 drops the training z range)"""
import os, sys
import numpy as np

R = np.concatenate([np.load(f)["rows"] for f in sys.argv[1:] if np.load(f)["rows"].size])
if os.environ.get("EXCL"): R = R[~((R[:, 15] >= 9984) & (R[:, 15] < 10500))]
ri, dw = R[:, 0].astype(int), R[:, 1]; V = np.rint(np.abs(R[:, 2:9])).astype(int); cosc = np.abs(R[:, 9]).astype(int); eg = R[:, 10]; spc = R[:, 11]; minc = R[:, 12]; meanc = R[:, 13]; L = R[:, 14]
truth = np.abs(dw).astype(int); fold = ri % 5; names = ["v6", "v6 flip-y", "v6 flip-x", "frozen_8c", "v4", "v6 line +3", "v6 line -3"]


def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return ((c - h) / d * 100, (c + h) / d * 100)


gmf = np.zeros(len(R));
for f in range(5):
    tr = fold != f; k = np.sum(eg[tr] * truth[tr]) / np.sum(eg[tr] ** 2); gmf[fold == f] = np.abs(eg[fold == f] * k)
gmc = np.rint(gmf).astype(int)
one = truth == 1; print(f"{len(R)} pairs, {len(np.unique(ri))} collections, one-wrap pairs {int(one.sum())}")
print("single votes, one-wrap pairs: " + ", ".join(f"{n} {np.mean(V[one, i] == 1) * 100:.1f}%" for i, n in enumerate(names)) + f", grad_mag {np.mean(gmc[one] == 1) * 100:.1f}%, cos peaks {np.mean(cosc[one] == 1) * 100:.1f}%")
nagree_gm = (V == gmc[:, None]).sum(1)                                                       # how many phase votes equal the grad_mag count
mode = np.array([np.bincount(v, minlength=12).argmax() for v in V]); nmode = np.array([np.bincount(v, minlength=12).max() for v in V])
print("\nrule A: certified count = grad_mag count, accepted when >= k of the 7 phase votes equal it")
for k in range(1, 8):
    m = one & (nagree_gm >= k); ok = (gmc[m] == 1); lo, hi = wilson(int(ok.sum()), int(m.sum()))
    print(f"  k >= {k}: coverage {m.sum() / one.sum() * 100:5.1f}%, precision {ok.mean() * 100 if m.any() else float('nan'):5.1f}% (n={int(m.sum())}, interval {lo:.1f}-{hi:.1f})")
print("\nrule B: certified count = mode of the phase votes, accepted when the mode has >= k votes AND equals the grad_mag count")
for k in range(2, 8):
    m = one & (nmode >= k) & (mode == gmc); ok = (mode[m] == 1); lo, hi = wilson(int(ok.sum()), int(m.sum()))
    print(f"  k >= {k}: coverage {m.sum() / one.sum() * 100:5.1f}%, precision {ok.mean() * 100 if m.any() else float('nan'):5.1f}% (n={int(m.sum())}, interval {lo:.1f}-{hi:.1f})")
print("\nrule C: mode of the phase votes with >= k votes, grad_mag not required")
for k in range(3, 8):
    m = one & (nmode >= k); ok = (mode[m] == 1); lo, hi = wilson(int(ok.sum()), int(m.sum()))
    print(f"  k >= {k}: coverage {m.sum() / one.sum() * 100:5.1f}%, precision {ok.mean() * 100 if m.any() else float('nan'):5.1f}% (n={int(m.sum())}, interval {lo:.1f}-{hi:.1f})")
try:
    from sklearn.linear_model import LogisticRegression
    X = np.c_[nmode, nagree_gm, (mode == gmc), (mode == cosc), np.abs(gmf - gmc), minc, meanc, np.log1p(L), np.abs(spc - mode), np.abs(R[:, 2:9]).std(1), (V == mode[:, None]).sum(1)]
    est = np.where(nagree_gm >= nmode, gmc, mode); y = (est == truth).astype(int); p = np.zeros(len(R))
    for f in range(5):
        tr = (fold != f) & one; te = fold == f
        if tr.sum() > 50: p[te] = LogisticRegression(max_iter=3000, C=1.0).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    print("\nrule D: learned certifier (leave-fold-out), one-wrap pairs, ranked by predicted correctness")
    o = np.argsort(-p[one]); yy = y[one][o]
    for cov in (0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5):
        k = int(cov * len(yy)); lo, hi = wilson(int(yy[:k].sum()), k); print(f"  coverage {cov * 100:3.0f}% -> precision {yy[:k].mean() * 100:5.1f}% (n={k}, interval {lo:.1f}-{hi:.1f})")
    cums = np.cumsum(yy) / np.arange(1, len(yy) + 1)
    for target in (0.99, 0.995):
        idx = np.nonzero(cums >= target)[0]; print(f"  highest coverage with precision >= {target * 100:.1f}%: {(idx.max() + 1) / len(yy) * 100 if len(idx) else 0:.1f}%")
except ImportError:
    print("sklearn missing")
print("\nall pair types (|dw| 2 and 3) with rule A, k >= 4: ", end="")
for t in (2, 3):
    m = (truth == t) & (nagree_gm >= 4); print(f"|dw|={t}: coverage {m.sum() / (truth == t).sum() * 100:.1f}%, precision {(gmc[m] == t).mean() * 100 if m.any() else float('nan'):.1f}%   ", end="")
print()
