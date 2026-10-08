"""Long-range winding counts (k = 1..10 wraps, rays up to 210 voxels) on truth pairs from the W2 held-out slab.
(a) certifier trained on human ladders only (1-4-wrap pairs), thresholds from the ladder out-of-fold predictions, applied unchanged to the long-range pairs, precision and coverage by k
(b) certifier trained on the long-range pairs themselves (k up to 10), 4 z-band folds
  python truth_longrange.py TRUTH_B.npz [TRUTH_A.npz] -- ROWS2_0 ROWS2_1 -- ROWS3_0 ROWS3_1"""
import sys
import numpy as np
import certlib as C

segs = []; cur = []
for a in sys.argv[1:]:
    if a == "--": segs.append(cur); cur = []
    else: cur.append(a)
segs.append(cur); ft, f2, f3 = segs
T = np.concatenate([np.load(f)["rows"] for f in ft]); tru = C.rows_to_dict(T); K = tru["truth"]; print(f"long-range truth pairs {len(T)}; k counts " + " ".join(f"{k}:{int((K == k).sum())}" for k in range(1, 11)))
lad = C.load_ladder(f2, f3); s1, s2 = C.gm_slopes(lad); Xl, cl, _ = C.features(lad, s1, s2); yl = (cl == lad["truth"]).astype(int); typl = (lad["truth"] >= 1) & (lad["truth"] <= 4)
Xt, ct_, g1t = C.features(tru, s1, s2); yt = (ct_ == K).astype(int)
for k in range(1, 11):
    m = K == k
    if m.sum() >= 5: print(f"k={k:2d}: n={int(m.sum())}, v6 {np.mean(np.rint(tru['V'][m, 0]) == k) * 100:.1f}%, path {np.mean(np.rint(tru['V'][m, 4]) == k) * 100:.1f}%, gm straight {np.mean(np.rint(g1t[m]) == k) * 100:.1f}%, candidate {np.mean(ct_[m] == k) * 100:.1f}%, mean length {tru['L'][m].mean():.0f} voxels")


def wilson(k, n, zz=1.96):
    if n == 0: return (float("nan"), float("nan"))
    p = k / n; d = 1 + zz * zz / n; c = p + zz * zz / (2 * n); h = zz * np.sqrt(p * (1 - p) / n + zz * zz / (4 * n * n)); return ((c - h) / d * 100, (c + h) / d * 100)


def report(name, p, tau):
    cert = p >= tau; lo, hi = wilson(int(yt[cert].sum()), int(cert.sum()))
    print(f"  [{name}] threshold {tau:.3f}: certified {int(cert.sum())} of {len(yt)} = {cert.mean() * 100:.1f}%, precision {yt[cert].mean() * 100:.2f}% ({lo:.1f}-{hi:.1f}), undercounts {int((ct_[cert] < K[cert]).sum())}")
    line = []
    for lo_k, hi_k in ((1, 1), (2, 2), (3, 4), (5, 6), (7, 8), (9, 10)):
        m = (K >= lo_k) & (K <= hi_k); c = m & cert
        if m.sum(): line.append(f"k{lo_k}-{hi_k}: n={int(m.sum())} cov {c.sum() / m.sum() * 100:.0f}% prec {yt[c].mean() * 100 if c.any() else float('nan'):.1f}%")
    print("      " + "; ".join(line))


pl = np.zeros(len(yl))
for f in range(5):
    tr = (lad["fold"] != f) & typl; te = lad["fold"] == f; pl[te] = C.fit(Xl[tr], yl[tr]).predict_proba(Xl[te])[:, 1]
final = C.fit(Xl[typl], yl[typl]); pt = final.predict_proba(Xt)[:, 1]
order = np.argsort(-pl[typl]); cum = np.cumsum(yl[typl][order]) / np.arange(1, typl.sum() + 1); ps = pl[typl][order]
print("\n(a) trained on human ladders (1-4 wraps), thresholds from ladder out-of-fold predictions")
for target in (0.98, 0.99, 0.995):
    ii = np.nonzero(cum >= target)[0]
    if len(ii): report(f"ladder->long-range, ladder precision {target * 100:.1f}%", pt, ps[ii.max()])
zb = np.digitize(T[:, 17], np.quantile(T[:, 17], [0.25, 0.5, 0.75])); p1 = np.zeros(len(yt))
for f in range(4):
    tr = zb != f; te = zb == f; p1[te] = C.fit(Xt[tr], yt[tr]).predict_proba(Xt[te])[:, 1]
order = np.argsort(-p1); cum = np.cumsum(yt[order]) / np.arange(1, len(yt) + 1); ps = p1[order]
print("\n(b) trained within the long-range pairs (k up to 10), 4 z-band folds")
for target in (0.98, 0.99, 0.995):
    ii = np.nonzero(cum >= target)[0]
    if len(ii): report(f"in-set CV precision {target * 100:.1f}%", p1, ps[ii.max()])
