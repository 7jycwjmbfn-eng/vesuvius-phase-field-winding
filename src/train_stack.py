"""Train the final stacking model of the counter network and the gradient-boosting certifier on the human ladders and fix its thresholds.
Inputs: counter predictions on the ladder pairs (counter_train --ladder output of a model trained on truth pairs only) and the out-of-fold gradient-boosting probabilities (slopes inside the folds).
Output: E:/vesuvius_counter/stack_model.pkl with the stacking model (trained on all ladder pairs of 1-4 windings, exclusions as in certlib.load_ladder) and thresholds for overall precision 98 / 99 / 99.5%
(lowest confidence whose cumulative precision on the out-of-fold stacking predictions stays >= the target, at least 50 pairs).
  python train_stack.py PRED_LAD0.npz PRED_LAD1.npz"""
import sys, pickle
import numpy as np
import certlib as C

d = [np.load(f) for f in sys.argv[1:]]; cc = np.concatenate([x["c"] for x in d]); pp = np.concatenate([x["p"] for x in d]); kk = np.concatenate([x["k"] for x in d]).astype(int); ri = np.concatenate([x["ri"] for x in d]).astype(int); sq = np.concatenate([x["seq"] for x in d]).astype(int); pr = np.concatenate([x["probs"] for x in d]).astype(np.float32)
f2 = ["E:/vesuvius_ladder2_rows_0.npz", "E:/vesuvius_ladder2_rows_1.npz"]; f3 = ["E:/vesuvius_ladder3_rows_0.npz", "E:/vesuvius_ladder3_rows_1.npz"]
lad = C.load_ladder(f2, f3); n = len(lad["truth"]); typ = (lad["truth"] >= 1) & (lad["truth"] <= 4); fold = lad["fold"]; pg = np.zeros(n); cg = np.zeros(n, int)
for f in range(5):
    trm = (fold != f) & typ; sub = {k: (v[trm] if hasattr(v, "__len__") and len(v) == n else v) for k, v in lad.items()}; a1, a2 = C.gm_slopes(sub); X, cand, _ = C.features(lad, a1, a2); y = (cand == lad["truth"]).astype(int); te = fold == f
    pg[te] = C.fit(X[trm], y[trm]).predict_proba(X[te])[:, 1]; cg[te] = cand[te]
idx = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(lad["ri"], lad["seq"]))}; sg = []; sc = []
for j, key in enumerate(zip(ri, sq)):
    i = idx.get((int(key[0]), int(key[1])))
    if i is not None and lad["truth"][i] == kk[j]: sg.append(i); sc.append(j)
sg = np.array(sg); sc = np.array(sc); truth = lad["truth"][sg]; fo = fold[sg]; tt = (truth >= 1) & (truth <= 4)


def stack_features(p_n, c_n, P9, p_g, c_g, L):
    agree = (c_n == c_g); ent = -(P9 * np.log(np.clip(P9, 1e-6, 1))).sum(1); pn_of_g = P9[np.arange(len(c_n)), np.clip(c_g, 0, 8)]
    return np.c_[p_n, p_g, c_n, c_g, agree, pn_of_g, ent, np.log1p(L), np.sort(P9, 1)[:, -2]]


Xs = stack_features(pp[sc], cc[sc], pr[sc], pg[sg], cg[sg], lad["L"][sg]); y_n = (cc[sc] == truth).astype(int); ps = np.zeros(len(truth))
for f in range(5):
    trm = (fo != f) & tt; ps[fo == f] = C.fit(Xs[trm], y_n[trm]).predict_proba(Xs[fo == f])[:, 1]
final = C.fit(Xs[tt], y_n[tt]); taus = {}
o = np.argsort(-ps[tt]); ok = y_n[tt][o]; cum = np.cumsum(ok) / np.arange(1, len(ok) + 1)
for target in (0.98, 0.99, 0.995):
    ii = np.nonzero((cum >= target) & (np.arange(1, len(ok) + 1) >= 50))[0]; taus[target] = float(ps[tt][o][ii.max()]) if len(ii) else float("inf")
    print(f"target {target * 100:.1f}%: tau {taus[target]:.4f}, OOF coverage {np.mean(ps[tt] >= taus[target]) * 100:.1f}%, precision {y_n[tt][ps[tt] >= taus[target]].mean() * 100:.2f}%")
pickle.dump(dict(model=final, taus=taus), open("E:/vesuvius_counter/stack_model.pkl", "wb")); print("saved stack_model.pkl", len(truth), "pairs")
