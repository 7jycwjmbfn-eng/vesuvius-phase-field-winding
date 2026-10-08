"""Evaluate the winding-count certificate on pairs drawn from the dense truth (truth_pairs.py rows), with the learned certifier in two ways:
 (1) within the truth pairs, leave-one-spatial-fold-out (4 bands of the start x position)
 (2) trained ONLY on the human-ladder rows (ladder_run2 + ladder_run3), threshold fixed on the ladder side at an overall precision of 99.0 / 98.0 / 99.5 %, then applied unchanged to the truth pairs
Only the votes that exist in both datasets are used: v6, frozen_8c, v6 line +3, v6 line -3, min-cost path (two costs), grad_mag integral (straight, path), lasagna cos peaks, surface-prediction crossings, CT peaks, confidences, length.
  python truth_certify.py TRUTH_ROWS.npz [TRUTH_ROWS2.npz ...] -- ROWS2_0 ROWS2_1 -- ROWS3_0 ROWS3_1"""
import os, sys
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

segs = []; cur = []
for a in sys.argv[1:]:
    if a == "--": segs.append(cur); cur = []
    else: cur.append(a)
segs.append(cur); ft, f2, f3 = segs
T = np.concatenate([np.load(f)["rows"] for f in ft if np.load(f)["rows"].size]); print(f"truth pairs {len(T)}; k counts {[int((T[:, 0] == k).sum()) for k in (1, 2, 3, 4)]}")


def wilson(k, n, zz=1.96):
    if n == 0: return (float("nan"), float("nan"))
    p = k / n; d = 1 + zz * zz / n; c = p + zz * zz / (2 * n); h = zz * np.sqrt(p * (1 - p) / n + zz * zz / (4 * n * n)); return ((c - h) / d * 100, (c + h) / d * 100)


# ---- ladder rows -> common feature dict
R2 = np.concatenate([np.load(f)["rows"] for f in f2 if np.load(f)["rows"].size]); R3 = np.concatenate([np.load(f)["rows"] for f in f3 if np.load(f)["rows"].size])
idx2 = {}; cnt = {}
for i, (ri, dw) in enumerate(R2[:, :2]):
    ri = int(ri); k = cnt.get(ri, 0); cnt[ri] = k + 1; idx2[(ri, k)] = i
cnt = {}; s2 = []; s3 = []
for j, (ri, dw) in enumerate(R3[:, :2]):
    ri = int(ri); k = cnt.get(ri, 0); cnt[ri] = k + 1; i = idx2.get((ri, k))
    if i is not None and abs(R2[i, 1] - dw) < 1e-9: s2.append(i); s3.append(j)
R2 = R2[s2]; R3 = R3[s3]; m = ~((R2[:, 15] >= 9984) & (R2[:, 15] < 10500)) if os.environ.get("EXCL", "1") != "0" else np.ones(len(R2), bool); R2 = R2[m]; R3 = R3[m]
if os.environ.get("EXCL_BOX", "1") != "0":                                                                                         # drop every ladder region that touches the truth box (z 9984-11008, y 2304-3840, x 3584-5120)
    import json
    regs = json.load(open("E:/vesuvius_ladder_regions.json")); tb_lo = np.array([9984, 2304, 3584]); tb_hi = tb_lo + np.array([1024, 1536, 1536])
    bad = {i for i, r in enumerate(regs) if np.all(np.array(r["lo"]) < tb_hi) and np.all(np.array(r["hi"]) > tb_lo)}; m = ~np.isin(R2[:, 0].astype(int), list(bad)); R2 = R2[m]; R3 = R3[m]; print(f"dropped {len(bad)} ladder regions touching the truth box")
lad = dict(truth=np.abs(R2[:, 1]).astype(int), fold=R2[:, 0].astype(int) % 5, V=np.abs(np.c_[R2[:, 2], R2[:, 5], R2[:, 7], R2[:, 8], R3[:, 2], R3[:, 3]]), gm1=R2[:, 10], gm2=R3[:, 9], cos=np.abs(R2[:, 9]), sp1=R2[:, 11], sp2=R3[:, 6],
           ct=R3[:, 7], minc=R2[:, 12], meanc=R2[:, 13], pmin=R3[:, 4], L=R2[:, 14])
tru = dict(truth=T[:, 0].astype(int), fold=np.minimum((np.arange(len(T)) * 0 + 0), 0).astype(int), V=np.abs(T[:, [1, 4, 2, 3, 5, 6]]), gm1=T[:, 7], gm2=T[:, 8] * 2.0, cos=T[:, 9], sp1=T[:, 10], sp2=T[:, 11], ct=T[:, 12], minc=T[:, 13], meanc=T[:, 14], pmin=T[:, 15], L=T[:, 16])
print(f"ladder pairs {len(R2)} (types {[int((lad['truth'] == k).sum()) for k in (1, 2, 3, 4)]})")


def features(D, slope1, slope2):
    V = np.rint(D["V"]).astype(int); Vv = np.where(np.isfinite(D["V"]), V, -1); g1 = np.abs(D["gm1"] * slope1); g2 = np.abs(D["gm2"] * slope2); gc1 = np.rint(g1).astype(int)
    mode = np.array([np.bincount(v[v >= 0], minlength=12).argmax() if (v >= 0).any() else 0 for v in Vv]); cand = np.where((V == gc1[:, None]).sum(1) >= (Vv == mode[:, None]).sum(1), gc1, mode)
    X = np.c_[(Vv == cand[:, None]).sum(1), (Vv >= 0).sum(1), (Vv == gc1[:, None]).sum(1), (D["cos"] == cand), (np.rint(np.nan_to_num(g2, nan=-9)) == cand), np.abs(g1 - gc1), np.abs(g1 - cand), np.nan_to_num(np.abs(g2 - cand), nan=-1),
              (D["ct"] == cand), (D["sp1"] == cand), (D["sp2"] == cand), np.abs(D["ct"] - cand), D["minc"], D["meanc"], np.nan_to_num(D["pmin"], nan=-1), np.log1p(D["L"]), np.abs(D["V"][:, :4] - cand[:, None]).mean(1), np.nan_to_num(np.abs(D["V"][:, 4] - cand), nan=-1), cand]
    return X, cand, g1


def fit(X, y): return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, max_leaf_nodes=15, l2_regularization=1.0, random_state=0).fit(X, y)


def table(name, p, tau, cand, truth, nboot=0):
    typ = (truth >= 1) & (truth <= 4); y = (cand == truth).astype(int); cert = typ & (p >= tau); lo, hi = wilson(int(y[cert].sum()), int(cert.sum()))
    print(f"  [{name}] threshold {tau:.3f}: certified {cert.sum()} of {typ.sum()} = {cert.sum() / typ.sum() * 100:.1f}%, precision {y[cert].mean() * 100:.2f}% ({lo:.1f}-{hi:.1f}), undercounts {int((cand[cert] < truth[cert]).sum())}")
    for t in (1, 2, 3, 4):
        mm = truth == t; c = mm & (p >= tau)
        if mm.sum():
            lo, hi = wilson(int(y[c].sum()), int(c.sum())); print(f"      {t}-wrap: n={int(mm.sum())}, coverage {c.sum() / mm.sum() * 100:5.1f}%, certified right {y[c].mean() * 100 if c.any() else float('nan'):5.1f}% ({lo:.1f}-{hi:.1f}), certified as one wrap when truth is more: {int(((cand[c] == 1) & (t > 1)).sum())}")


# calibration slope of grad_mag integral on ladder rows (single slope through the origin, L2-voxel units), used for both datasets
ok = np.isfinite(lad["gm1"]); s1 = float(np.sum(lad["gm1"][ok] * lad["truth"][ok]) / np.sum(lad["gm1"][ok] ** 2)); ok2 = np.isfinite(lad["gm2"]); s2_ = float(np.sum(lad["gm2"][ok2] * lad["truth"][ok2]) / np.sum(lad["gm2"][ok2] ** 2))
print(f"grad_mag slopes fitted on ladders: straight {s1:.4f}, path {s2_:.4f}")
Xl, cl, g1l = features(lad, s1, s2_); Xt, ct_, g1t = features(tru, s1, s2_); yl = (cl == lad["truth"]).astype(int); yt = (ct_ == tru["truth"]).astype(int); typl = (lad["truth"] >= 1) & (lad["truth"] <= 4); typt = (tru["truth"] >= 1) & (tru["truth"] <= 4)
for k in (1, 2, 3, 4):
    mm = tru["truth"] == k
    if mm.sum(): print(f"truth pairs k={k}: single votes " + ", ".join(f"{n} {np.mean(np.rint(tru['V'][mm, i]) == k) * 100:.1f}%" for i, n in enumerate(["v6", "frozen", "+3", "-3", "path2", "path4"])) + f", gm straight {np.mean(np.rint(g1t[mm]) == k) * 100:.1f}%, cos {np.mean(tru['cos'][mm] == k) * 100:.1f}%, ct {np.mean(tru['ct'][mm] == k) * 100:.1f}%, candidate {np.mean(ct_[mm] == k) * 100:.1f}%")

# (2) ladder-trained, threshold fixed on ladder out-of-fold predictions, applied to the truth pairs
pl = np.zeros(len(yl))
for f in range(5):
    tr = (lad["fold"] != f) & typl; te = lad["fold"] == f; pl[te] = fit(Xl[tr], yl[tr]).predict_proba(Xl[te])[:, 1]
final = fit(Xl[typl], yl[typl]); pt = final.predict_proba(Xt)[:, 1]
order = np.argsort(-pl[typl]); cum = np.cumsum(yl[typl][order]) / np.arange(1, typl.sum() + 1); ps = pl[typl][order]
print("\n(2) trained on human ladders only; threshold chosen on ladder out-of-fold predictions; applied unchanged to the truth pairs")
for target in (0.98, 0.99, 0.995):
    ii = np.nonzero(cum >= target)[0]
    if not len(ii): continue
    tau = ps[ii.max()]; print(f"ladder side, overall precision >= {target * 100:.1f}%:"); table("ladder OOF", pl, tau, cl, lad["truth"]); table("TRUTH pairs", pt, tau, ct_, tru["truth"])

# (1) within truth pairs, spatial folds by start position (index modulo is not spatial; rows have z only, so use z bands plus order blocks)
zb = np.digitize(T[:, 17], np.quantile(T[:, 17], [0.25, 0.5, 0.75])); pt1 = np.zeros(len(yt))
for f in range(4):
    tr = (zb != f) & typt; te = zb == f; pt1[te] = fit(Xt[tr], yt[tr]).predict_proba(Xt[te])[:, 1]
order = np.argsort(-pt1[typt]); cum = np.cumsum(yt[typt][order]) / np.arange(1, typt.sum() + 1); ps = pt1[typt][order]
print("\n(1) trained and evaluated within truth pairs (4 z-band folds)")
for target in (0.98, 0.99, 0.995):
    ii = np.nonzero(cum >= target)[0]
    if not len(ii): print(f"  target {target}: not reached"); continue
    tau = ps[ii.max()]; table(f"truth CV >= {target * 100:.1f}%", pt1, tau, ct_, tru["truth"])
