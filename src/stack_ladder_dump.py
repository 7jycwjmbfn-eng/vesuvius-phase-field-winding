"""Combine the counter network and the gradient-boosting certifier on the human ladders (same pairs and folds as compare_ladder.py).
Rules compared (thresholds from the other four folds, 5 folds by collection):
  agree : certified when both give the same count; confidence = min(p_nn, p_gbm)
  stack : gradient boosting on [p_nn, p_gbm, counts, agreement, NN probability of the GBM count, NN entropy, pair length] trained out-of-fold; predicts whether the NN count is right
  python stack_ladder.py PRED_LAD0.npz PRED_LAD1.npz [NBOOT]"""
import sys
import numpy as np
import certlib as C

import os
predf = sys.argv[1:]; NB = int(os.environ.get("NB", 200))
d = [np.load(f) for f in predf]; cc = np.concatenate([x["c"] for x in d]); pp = np.concatenate([x["p"] for x in d]); kk = np.concatenate([x["k"] for x in d]).astype(int); ri = np.concatenate([x["ri"] for x in d]).astype(int); sq = np.concatenate([x["seq"] for x in d]).astype(int); pr = np.concatenate([x["probs"] for x in d]).astype(np.float32)
f2 = ["E:/vesuvius_ladder2_rows_0.npz", "E:/vesuvius_ladder2_rows_1.npz"]; f3 = ["E:/vesuvius_ladder3_rows_0.npz", "E:/vesuvius_ladder3_rows_1.npz"]
lad = C.load_ladder(f2, f3); n = len(lad["truth"]); typ = (lad["truth"] >= 1) & (lad["truth"] <= 4); fold = lad["fold"]; pg = np.zeros(n); cg = np.zeros(n, int)
for f in range(5):
    trm = (fold != f) & typ; sub = {k: (v[trm] if hasattr(v, "__len__") and len(v) == n else v) for k, v in lad.items()}; a1, a2 = C.gm_slopes(sub); X, cand, _ = C.features(lad, a1, a2); y = (cand == lad["truth"]).astype(int); te = fold == f
    pg[te] = C.fit(X[trm], y[trm]).predict_proba(X[te])[:, 1]; cg[te] = cand[te]
idx = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(lad["ri"], lad["seq"]))}; sg = []; sc = []
for j, key in enumerate(zip(ri, sq)):
    i = idx.get((int(key[0]), int(key[1])))
    if i is not None and lad["truth"][i] == kk[j]: sg.append(i); sc.append(j)
sg = np.array(sg); sc = np.array(sc); truth = lad["truth"][sg]; rr = lad["ri"][sg]; fo = fold[sg]; tt = (truth >= 1) & (truth <= 4)
c_n, p_n, P9 = cc[sc], pp[sc], pr[sc]; c_g, p_g = cg[sg], pg[sg]; agree = (c_n == c_g)
ent = -(P9 * np.log(np.clip(P9, 1e-6, 1))).sum(1); pn_of_g = P9[np.arange(len(sc)), np.clip(c_g, 0, 8)]
Xs = np.c_[p_n, p_g, c_n, c_g, agree, pn_of_g, ent, np.log1p(lad["L"][sg]), np.sort(P9, 1)[:, -2]]; y_n = (c_n == truth).astype(int)
ps = np.zeros(len(truth))
for f in range(5):
    trm = (fo != f) & tt; ps[fo == f] = C.fit(Xs[trm], y_n[trm]).predict_proba(Xs[fo == f])[:, 1]


def thr(c, p, mask, target):
    o = np.argsort(-p[mask]); ok = (c[mask][o] == truth[mask][o]).astype(float); cum = np.cumsum(ok) / np.arange(1, len(ok) + 1); ii = np.nonzero((cum >= target) & (np.arange(1, len(ok) + 1) >= 20))[0]
    return float(p[mask][o][ii.max()]) if len(ii) else np.inf


M = {"counter": (c_n, p_n), "GBM": (c_g, p_g), "agree": (c_n, np.where(agree, np.minimum(p_n, p_g), 0.0)), "stack": (c_n, ps)}
zmid = lad["z"][sg]; CERT = {}
for target in (0.98, 0.99, 0.995):
    print(f"\n=== overall precision target {target * 100:.1f}% (pairs {int(tt.sum())} of 1-4 windings; thresholds from the other folds)")
    for name, (c, p) in M.items():
        cert = np.zeros(len(truth), bool)
        for f in range(5): cert[fo == f] = p[fo == f] >= thr(c, p, tt & (fo != f), target)
        cert &= tt; CERT[(name, target)] = cert.copy(); right = (c == truth); cov = cert.sum() / tt.sum(); prec = right[cert].mean() if cert.any() else float("nan")
        rng = np.random.default_rng(0); colls = np.unique(rr); idxc = {u: np.nonzero(rr == u)[0] for u in colls}; bs = []
        for b in range(NB):
            pick = rng.choice(colls, len(colls), replace=True); ii = np.concatenate([idxc[u] for u in pick]); ce = cert[ii]; bs.append((ce.sum() / tt[ii].sum(), right[ii][ce].mean() if ce.any() else np.nan))
        bs = np.array(bs); per = "; ".join(f"{k}-wrap {np.mean(cert[truth == k]) * 100:.1f}%/{right[cert & (truth == k)].mean() * 100 if (cert & (truth == k)).any() else float('nan'):.1f}% (missed {int(((c < truth) & cert & (truth == k)).sum())})" for k in (1, 2, 3, 4))
        print(f"  {name:8s}: coverage {cov * 100:.1f}% ({np.percentile(bs[:, 0], 2.5) * 100:.1f}-{np.percentile(bs[:, 0], 97.5) * 100:.1f}), precision {prec * 100:.2f}% ({np.nanpercentile(bs[:, 1], 2.5) * 100:.1f}-{np.nanpercentile(bs[:, 1], 97.5) * 100:.1f}); {per}")

print("\n=== one-wrap pairs by z band and in total (PCU-style rows), target 99.0% / 99.5% overall (thresholds from the other folds)")
for target in (0.99, 0.995):
    for lo_z, hi_z in ((8400, 9400), (11000, 12000), (0, 99999)):
        m = (truth == 1) & (zmid >= lo_z) & (zmid < hi_z)
        row = []
        for name in ("GBM", "stack"):
            c, p = M[name]; cert = CERT[(name, target)] & m; right = (c == truth)
            row.append(f"{name}: coverage {cert.sum() / m.sum() * 100:.1f}% precision {right[cert].mean() * 100 if cert.any() else float('nan'):.2f}% ({int((~right[cert]).sum())} errors of {int(cert.sum())})")
        print(f"  target {target * 100:.1f}%, z {lo_z}-{hi_z} (n={int(m.sum())}): " + "; ".join(row))

# dump per-pair decisions for the same-pair comparison with PCU (compare_pcu.py)
out = dict(ri=rr, seq=lad["seq"][sg], truth=truth, z=zmid, L=lad["L"][sg], fold=fo, c_n=c_n, p_n=p_n, c_g=c_g, p_g=p_g, p_s=ps)
for target in (0.98, 0.99, 0.995):
    for name in M: out[f"cert_{name}_{int(target * 1000)}"] = CERT[(name, target)]
np.savez(os.environ.get("DUMP_OUT", "E:/vesuvius_counter/stack_pairs_oof.npz"), **out); print("dumped", len(truth), "pairs")
