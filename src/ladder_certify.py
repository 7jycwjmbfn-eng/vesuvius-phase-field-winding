"""Certify winding counts along human-ladder pairs and report precision against coverage (PCU recipe: several independent counts, accept only when they agree).
Reads the row files of ladder_run.py: ri, dw, est_psi, cnt_cos, est_gm, minc, meanc, cnt_sp, L, zmid.  The truth is |dw| (direction-free count).  Primary metric: one-wrap pairs (|dw| = 1),
as in the community certificate; |dw| = 2, 3 are reported separately.  The grad_mag calibration and the learned certifier are fitted on collections of one fold and evaluated on the others.
  python ladder_certify.py ROWS1.npz [ROWS2.npz ...]"""
import sys, glob
import numpy as np

import os
rows = np.concatenate([np.load(f)["rows"] for f in sys.argv[1:] if np.load(f)["rows"].size])
if os.environ.get("EXCL"): rows = rows[~((rows[:, 9] >= 9984) & (rows[:, 9] < 10500))]                         # EXCL=1: drop pairs inside the training z range of the network
ri, dw, ep, cc, eg, minc, meanc, csp, L, zm = rows.T
def wilson(k, n, z=1.96):
    if n == 0: return (float('nan'), float('nan'))
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return ((c - h) / d * 100, (c + h) / d * 100)
truth = np.abs(dw).astype(int); psi_cnt = np.rint(np.abs(ep)).astype(int); cos_cnt = np.abs(cc).astype(int)
print(f"{len(rows)} pairs from {len(np.unique(ri))} collections; |dw| = 1: {int((truth == 1).sum())}, 2: {int((truth == 2).sum())}, 3: {int((truth == 3).sum())}, >= 4: {int((truth >= 4).sum())}")
fold = (ri.astype(int) % 5)                                                                  # 5 folds by collection
gm_cnt = np.zeros(len(rows), int); gmf = np.zeros(len(rows))
for f in range(5):                                                                           # grad_mag slope fitted on the other folds (one slope, least squares through the origin)
    tr = fold != f; k = np.sum(eg[tr] * truth[tr]) / np.sum(eg[tr] ** 2); gmf[fold == f] = np.abs(eg[fold == f] * k); gm_cnt[fold == f] = np.rint(gmf[fold == f]).astype(int)
ok_psi = psi_cnt == truth; ok_gm = gm_cnt == truth; ok_cos = cos_cnt == truth
for name, sel in (("|dw| = 1", truth == 1), ("|dw| = 2", truth == 2), ("|dw| = 3", truth == 3), ("all", truth > 0)):
    print(f"  {name:9s} n={int(sel.sum()):6d}: complex phase {ok_psi[sel].mean() * 100:5.1f}%  | official cos peaks {ok_cos[sel].mean() * 100:5.1f}%  | official grad_mag integral {ok_gm[sel].mean() * 100:5.1f}%")
one = truth == 1
agree3 = (psi_cnt == gm_cnt) & (psi_cnt == cos_cnt); agree2 = psi_cnt == gm_cnt
print("\nagreement certificates, one-wrap pairs (precision of the agreed count against the ladder):")
for lab, m in (("phase + grad_mag agree", agree2), ("phase + cos agree", psi_cnt == cos_cnt), ("all three agree", agree3), ("phase + grad_mag agree, min conf >= 0.3", agree2 & (minc >= 0.3)), ("all three agree, min conf >= 0.3", agree3 & (minc >= 0.3))):
    mm = m & one; ok = (psi_cnt[mm] == truth[mm]).mean() * 100 if mm.any() else float("nan")
    lo_, hi_ = wilson(int((psi_cnt[mm] == truth[mm]).sum()), int(mm.sum())); print(f"  {lab:42s}: coverage {mm.sum() / one.sum() * 100:5.1f}%, precision {ok:5.1f}% (n={int(mm.sum())}, 95% interval {lo_:.1f}-{hi_:.1f})")
# learned certifier: logistic regression, leave-fold-out; features describe how consistent the evidence is
try:
    from sklearn.linear_model import LogisticRegression
    X = np.c_[np.abs(np.abs(ep) - psi_cnt), np.abs(gmf - gm_cnt), (psi_cnt == gm_cnt), (psi_cnt == cos_cnt), (gm_cnt == cos_cnt), minc, meanc, np.log1p(L), np.abs(np.abs(ep) - gmf), np.abs(csp - psi_cnt), csp]
    y = ok_psi.astype(int); p = np.zeros(len(rows))
    for f in range(5):
        tr = (fold != f) & (truth == 1); te = fold == f
        if tr.sum() > 50: p[te] = LogisticRegression(max_iter=2000, C=1.0).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    print("\nlearned certifier (leave-fold-out), one-wrap pairs; ranked by predicted correctness; the estimate is the phase count:")
    o = np.argsort(-p[one]); yy = y[one][o]
    for cov in (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3):
        k = int(cov * len(yy)); lo_, hi_ = wilson(int(yy[:k].sum()), k); print(f"  coverage {cov * 100:3.0f}% -> precision {yy[:k].mean() * 100:5.1f}% (n={k}, 95% interval {lo_:.1f}-{hi_:.1f})")
except ImportError:
    print("sklearn not installed")
zb = np.digitize(zm, [8000, 10000, 12000, 14000, 16000])
print("\nby z band (L2), one-wrap pairs: phase / grad_mag / agree-certified precision (coverage)")
for b, lab in enumerate(("< 8000", "8000-10000", "10000-12000", "12000-14000", "14000-16000", ">= 16000")):
    sel = one & (zb == b)
    if sel.sum() >= 10:
        c = sel & agree2; print(f"  {lab:12s} n={int(sel.sum()):5d}: phase {ok_psi[sel].mean() * 100:5.1f}%, grad_mag {ok_gm[sel].mean() * 100:5.1f}%, certified {ok_psi[c].mean() * 100 if c.any() else float('nan'):5.1f}% ({c.sum() / sel.sum() * 100:.0f}%)")
