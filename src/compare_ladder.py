"""Counter network against the gradient-boosting certifier on the human ladders, same pairs, same protocol.
Pairs: the ladder pairs with a profile (length <= 255 voxels) that survive the certlib exclusions (training z range, regions touching the truth box), true count 1..4 for the scored tables.
Both methods give a count c and a confidence p per pair.  Threshold protocol: 5 folds by collection (ri % 5); for fold f the threshold is the lowest confidence whose cumulative precision (over pairs of the other four folds with true count 1..4, sorted by confidence)
stays >= the overall-precision target (at least 20 pairs); it is applied to fold f.  The gradient-boosting certifier uses grad_mag slopes fitted inside the training folds.  Intervals: bootstrap over collections (thresholds fixed).
  python compare_ladder.py PRED_LAD0.npz PRED_LAD1.npz LAD0.npz LAD1.npz [NBOOT]      (pred files from counter_train --ladder, profile files for zmid)"""
import sys
import numpy as np
import certlib as C

predf = sys.argv[1:3]; ladf = sys.argv[3:5]; NB = int(sys.argv[5]) if len(sys.argv) > 5 else 300
cc, pp, kk, ri, sq = [], [], [], [], []
for f in predf:
    d = np.load(f); cc.append(d["c"]); pp.append(d["p"]); kk.append(d["k"]); ri.append(d["ri"]); sq.append(d["seq"])
cc = np.concatenate(cc); pp = np.concatenate(pp); kk = np.concatenate(kk).astype(int); ri = np.concatenate(ri).astype(int); sq = np.concatenate(sq).astype(int)
f2 = ["E:/vesuvius_ladder2_rows_0.npz", "E:/vesuvius_ladder2_rows_1.npz"]; f3 = ["E:/vesuvius_ladder3_rows_0.npz", "E:/vesuvius_ladder3_rows_1.npz"]
lad = C.load_ladder(f2, f3); n = len(lad["truth"]); typ = (lad["truth"] >= 1) & (lad["truth"] <= 4); fold = lad["fold"]
s1, s2 = C.gm_slopes(lad); pg = np.zeros(n); cg = np.zeros(n, int)
for f in range(5):
    trm = (fold != f) & typ; sub = {k: (v[trm] if hasattr(v, "__len__") and len(v) == n else v) for k, v in lad.items()}; a1, a2 = C.gm_slopes(sub); X, cand, _ = C.features(lad, a1, a2); y = (cand == lad["truth"]).astype(int); te = fold == f
    pg[te] = C.fit(X[trm], y[trm]).predict_proba(X[te])[:, 1]; cg[te] = cand[te]
idx = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(lad["ri"], lad["seq"]))}; sel_g = []; sel_c = []
for j, key in enumerate(zip(ri, sq)):
    i = idx.get((int(key[0]), int(key[1])))
    if i is not None and lad["truth"][i] == kk[j]: sel_g.append(i); sel_c.append(j)
sel_g = np.array(sel_g); sel_c = np.array(sel_c); print(f"common pairs {len(sel_g)} of {n} certlib pairs and {len(cc)} profile pairs")
truth = lad["truth"][sel_g]; rr = lad["ri"][sel_g]; fo = fold[sel_g]; M = {"counter": (cc[sel_c], pp[sel_c]), "GBM certifier": (cg[sel_g], pg[sel_g])}
tt = (truth >= 1) & (truth <= 4)


def thr(c, p, mask, target):
    o = np.argsort(-p[mask]); ok = (c[mask][o] == truth[mask][o]).astype(float); cum = np.cumsum(ok) / np.arange(1, len(ok) + 1); ii = np.nonzero((cum >= target) & (np.arange(1, len(ok) + 1) >= 20))[0]
    return float(p[mask][o][ii.max()]) if len(ii) else np.inf


for target in (0.98, 0.99, 0.995):
    print(f"\n=== overall precision target {target * 100:.1f}% over pairs of 1 to 4 windings (thresholds from the other four folds)")
    for name, (c, p) in M.items():
        cert = np.zeros(len(truth), bool)
        for f in range(5): cert[fo == f] = p[fo == f] >= thr(c, p, tt & (fo != f), target)
        cert &= tt; right = (c == truth); cov = cert.sum() / tt.sum(); prec = right[cert].mean()
        rng = np.random.default_rng(0); colls = np.unique(rr); idxc = {u: np.nonzero(rr == u)[0] for u in colls}; bs = []
        for b in range(NB):
            pick = rng.choice(colls, len(colls), replace=True); ii = np.concatenate([idxc[u] for u in pick]); t_ = tt[ii]; ce = cert[ii]; bs.append((ce.sum() / t_.sum(), right[ii][ce].mean() if ce.any() else np.nan))
        bs = np.array(bs); per = "; ".join(f"{k}-wrap {np.mean(cert[truth == k]) * 100:.1f}% / {right[cert & (truth == k)].mean() * 100 if (cert & (truth == k)).any() else float('nan'):.1f}% (missed sheet {int(((c < truth) & cert & (truth == k)).sum())})" for k in (1, 2, 3, 4))
        print(f"  {name:14s}: coverage {cov * 100:.1f}% ({np.percentile(bs[:, 0], 2.5) * 100:.1f}-{np.percentile(bs[:, 0], 97.5) * 100:.1f}), precision {prec * 100:.2f}% ({np.nanpercentile(bs[:, 1], 2.5) * 100:.1f}-{np.nanpercentile(bs[:, 1], 97.5) * 100:.1f}); {per}")
