"""Certification with all votes: rows of ladder_run2 (7 phase votes, grad_mag, cos, surface prediction) joined with ladder_run3 (two minimum-cost-path phase counts, surface crossings on the path,
CT profile peaks).  The rows of both runs have the same order (same pair enumeration); the join is checked on (region, winding difference).
Truth |dw|.  Reports single votes, rule A with the extended vote set, and a learned certifier (gradient boosting, leave-fold-out by collection) with precision against coverage on one-wrap pairs,
the coverage at 99.0 / 99.5%, the harder bands, and the dangerous error (a two-wrap pair certified as one wrap).
  python ladder_certify3.py ROWS2_0 ROWS2_1 -- ROWS3_0 ROWS3_1   (the training z range is dropped by default; env EXCL=0 keeps it)"""
import os, sys
import numpy as np

sep = sys.argv.index("--"); f2 = sys.argv[1:sep]; f3 = sys.argv[sep + 1:]
R2 = np.concatenate([np.load(f)["rows"] for f in f2 if np.load(f)["rows"].size]); R3 = np.concatenate([np.load(f)["rows"] for f in f3 if np.load(f)["rows"].size])
# the third run may be partial; join on the sequence index inside each region
def key(R): return R[:, 0].astype(int) * 100000 + np.cumsum(np.r_[0, (np.diff(R[:, 0]) == 0)] * 1) * 0
idx2 = {}; cnt = {}
for i, (ri, dw) in enumerate(R2[:, :2]):
    ri = int(ri); k = cnt.get(ri, 0); cnt[ri] = k + 1; idx2[(ri, k)] = i
cnt = {}; sel2 = []; sel3 = []
for j, (ri, dw) in enumerate(R3[:, :2]):
    ri = int(ri); k = cnt.get(ri, 0); cnt[ri] = k + 1
    i = idx2.get((ri, k))
    if i is not None and abs(R2[i, 1] - dw) < 1e-9: sel2.append(i); sel3.append(j)
R2 = R2[sel2]; R3 = R3[sel3]; print(f"joined rows: {len(R2)}")
if os.environ.get("EXCL", "1") != "0":
    m = ~((R2[:, 15] >= 9984) & (R2[:, 15] < 10500)); R2 = R2[m]; R3 = R3[m]
ri = R2[:, 0].astype(int); truth = np.abs(R2[:, 1]).astype(int); fold = ri % 5
P = np.abs(np.c_[R2[:, 2:9], R3[:, 2:4]]); V = np.rint(P).astype(int)                                          # 9 phase votes
cosc = np.abs(R2[:, 9]).astype(int); gm1 = R2[:, 10]; gm2 = R3[:, 9]; spl = R2[:, 11]; spp = R3[:, 6]; ctp = R3[:, 7]; ctm = R3[:, 8]; minc = R2[:, 12]; meanc = R2[:, 13]; L = R2[:, 14]; z = R2[:, 15]; pmin = R3[:, 4]


def wilson(k, n, zz=1.96):
    if n == 0: return (float("nan"), float("nan"))
    p = k / n; d = 1 + zz * zz / n; c = p + zz * zz / (2 * n); h = zz * np.sqrt(p * (1 - p) / n + zz * zz / (4 * n * n)); return ((c - h) / d * 100, (c + h) / d * 100)


def calib(x):
    out = np.zeros(len(x)); ok = np.isfinite(x)
    for f in range(5):
        tr = (fold != f) & ok; k = np.sum(x[tr] * truth[tr]) / np.sum(x[tr] ** 2); out[fold == f] = np.abs(x[fold == f] * k)
    return out


g1 = calib(gm1); g2 = calib(np.where(np.isfinite(gm2), gm2, 0)); g2[~np.isfinite(gm2)] = np.nan; gc1 = np.rint(g1).astype(int)
one = truth == 1
names = ["v6", "v6 flip-y", "v6 flip-x", "frozen_8c", "v4", "v6 line +3", "v6 line -3", "v6 min-cost path (1/conf^2)", "v6 min-cost path (1/conf^4)"]
print("single votes, one-wrap pairs: " + "; ".join(f"{n} {np.mean(V[one, i] == 1) * 100:.1f}%" for i, n in enumerate(names)) + f"; grad_mag straight {np.mean(gc1[one] == 1) * 100:.1f}%; grad_mag path {np.nanmean(np.rint(g2[one]) == 1) * 100:.1f}%; cos {np.mean(cosc[one] == 1) * 100:.1f}%; CT peaks {np.mean(ctp[one] == 1) * 100:.1f}%")
valid_path = np.isfinite(P[:, 7])
for lab, cols in (("7 straight-line votes", list(range(7))), ("9 votes (with the two path votes)", list(range(9))), ("the two path votes only", [7, 8])):
    nag = (V[:, cols] == gc1[:, None]).sum(1)
    for k in sorted({1, 2, 3, max(1, len(cols) - 1), len(cols)}):
        m = one & (nag >= k); ok = gc1[m] == 1; lo, hi = wilson(int(ok.sum()), int(m.sum()))
        print(f"  rule A [{lab}], >= {k} votes equal the grad_mag count: coverage {m.sum() / one.sum() * 100:5.1f}%, precision {ok.mean() * 100:5.1f}% (n={int(m.sum())}, {lo:.1f}-{hi:.1f})")
try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    Vv = np.where(np.isfinite(P), V, -1); mode = np.array([np.bincount(v[v >= 0], minlength=12).argmax() if (v >= 0).any() else 0 for v in Vv])
    cand = np.where((V == gc1[:, None]).sum(1) >= (Vv == mode[:, None]).sum(1), gc1, mode)                                       # candidate count: grad_mag count unless the phase mode has more votes
    X = np.c_[(Vv == cand[:, None]).sum(1), (Vv >= 0).sum(1), (Vv == gc1[:, None]).sum(1), (cosc == cand), (np.rint(g2) == cand), np.abs(g1 - gc1), np.abs(g1 - cand), np.nan_to_num(np.abs(g2 - cand), nan=-1),
              (ctp == cand), (spl == cand), (spp == cand), np.abs(ctp - cand), ctm, minc, meanc, np.nan_to_num(pmin, nan=-1), np.log1p(L), np.abs(P[:, :7] - cand[:, None]).mean(1), np.nan_to_num(np.abs(P[:, 7] - cand), nan=-1), cand, z]
    y = (cand == truth).astype(int); p = np.zeros(len(R2)); typ = (truth >= 1) & (truth <= 4)                               # train and evaluate on 1- to 4-wrap pairs together
    for f in range(5):
        tr = (fold != f) & typ; te = fold == f
        p[te] = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, max_leaf_nodes=15, l2_regularization=1.0, random_state=0).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    print("\nlearned certifier trained and evaluated on 1- to 4-wrap pairs together (leave-fold-out); ONE threshold for all pair types")
    order = np.argsort(-p[typ]); yy = y[typ][order]; cums = np.cumsum(yy) / np.arange(1, len(yy) + 1); ps = p[typ][order]
    for target in (0.98, 0.99, 0.995):
        ii = np.nonzero(cums >= target)[0]
        if not len(ii): print(f"  precision >= {target * 100:.1f}%: not reached"); continue
        k = ii.max() + 1; tau = ps[k - 1]; cert = typ & (p >= tau); lo, hi = wilson(int(y[cert].sum()), int(cert.sum()))
        print(f"  overall precision >= {target * 100:.1f}%: coverage {cert.sum() / typ.sum() * 100:.1f}% of all 1-4-wrap pairs, precision {y[cert].mean() * 100:.2f}% (n={int(cert.sum())}, {lo:.1f}-{hi:.1f})")
        for t in (1, 2, 3, 4):
            m = truth == t; c = m & (p >= tau); lo, hi = wilson(int(y[c].sum()), int(c.sum()))
            under = int((cand[c] < t).sum()); print(f"      {t}-wrap pairs: coverage {c.sum() / m.sum() * 100:5.1f}%, certified count right {y[c].mean() * 100:5.1f}% ({lo:.1f}-{hi:.1f}), undercounts (a sheet missed) {under} = {under / max(c.sum(), 1) * 100:.2f}% of certified, certified as ONE wrap when truth is more: {int(((cand[c] == 1) & (t > 1)).sum())}")
        if target == 0.99:
            print("      by z band, certified at this threshold, one-wrap pairs:")
            for lo_z, hi_z in ((0, 8000), (8000, 10000), (10000, 12000), (11000, 12000), (12000, 14000), (14000, 16000), (16000, 99999)):
                m = (truth == 1) & (z >= lo_z) & (z < hi_z); c = m & (p >= tau)
                if m.sum() >= 20: lo, hi = wilson(int(y[c].sum()), int(c.sum())); print(f"        z {lo_z}-{hi_z}: n={int(m.sum())}, coverage {c.sum() / m.sum() * 100:.0f}%, precision {y[c].mean() * 100:.1f}% ({lo:.1f}-{hi:.1f})")
except ImportError:
    print("sklearn missing")
