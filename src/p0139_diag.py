"""Exploratory diagnostics of the PHerc0139 reading (not part of the preregistered test; the certifier and its threshold are not changed).
 1. phase votes alone: share of pairs where >= m of the 6 phase votes equal the mode (mode in 1..4), coverage and precision per dataset and per k
 2. calibration: mean predicted probability against realised accuracy of the no-lasagna certifier on ladders (out-of-fold) and on 0139 pairs
 3. shift of the inputs the certifier uses: pair length per wrap, confidence statistics
  python p0139_diag.py PAIRS_0139.npz PAIRS_W2_A.npz"""
import sys
import numpy as np
sys.path.insert(0, "../wfield")
import certlib as C
import nolas as N

f2 = ["E:/vesuvius_ladder2_rows_0.npz", "E:/vesuvius_ladder2_rows_1.npz"]; f3 = ["E:/vesuvius_ladder3_rows_0.npz", "E:/vesuvius_ladder3_rows_1.npz"]
lad = C.load_ladder(f2, f3); typl = (lad["truth"] >= 1) & (lad["truth"] <= 4)
sets = {"ladders (1-4 wraps)": ({k: v[typl] if hasattr(v, "__len__") and len(v) == len(typl) else v for k, v in lad.items()}), "0139 truth pairs": C.rows_to_dict(np.load(sys.argv[1])["rows"]), "W2 truth pairs": C.rows_to_dict(np.load(sys.argv[2])["rows"])}


def mode_info(D):
    V = np.rint(np.nan_to_num(D["V"], nan=-1)).astype(int); out = []
    for v in V:
        vv = v[v >= 0]
        if len(vv) == 0: out.append((0, 0)); continue
        b = np.bincount(vv, minlength=12); out.append((int(b.argmax()), int(b.max())))
    return np.array(out)


for name, D in sets.items():
    truth = D["truth"]; mi = mode_info(D); ok4 = (mi[:, 0] >= 1) & (mi[:, 0] <= 4)
    print(f"== {name}: pairs {len(truth)}; mean log2 of pair length per wrap: {np.mean(np.log2(D['L'] / np.maximum(truth, 1))):.2f}; median length per wrap {np.median(D['L'] / np.maximum(truth, 1)):.1f}; v6 minconf median {np.median(D['minc']):.2f}, meanconf median {np.median(D['meanc']):.2f}")
    for m in (4, 5, 6):
        c = ok4 & (mi[:, 1] >= m); line = f"   >= {m} of 6 phase votes equal the mode: coverage {c.mean() * 100:.1f}%, precision {np.mean(mi[c, 0] == truth[c]) * 100:.1f}%"
        line += "; by k: " + ", ".join(f"{k}: {np.mean(c[truth == k]) * 100:.0f}% / {np.mean(mi[c & (truth == k), 0] == k) * 100 if (c & (truth == k)).any() else float('nan'):.0f}%" for k in (1, 2, 3, 4)); print(line)
# calibration of the no-lasagna certifier
X, cand = N.features_nolas(sets["ladders (1-4 wraps)"]); y = (cand == sets["ladders (1-4 wraps)"]["truth"]).astype(int); fold = lad["fold"][typl]; p = np.zeros(len(y))
for f in range(5):
    tr = fold != f; p[fold == f] = C.fit(X[tr], y[tr]).predict_proba(X[fold == f])[:, 1]
final = C.fit(X, y)
for name in ("ladders (1-4 wraps)", "0139 truth pairs", "W2 truth pairs"):
    D = sets[name]; Xd, cd = N.features_nolas(D); yd = (cd == D["truth"]).astype(int); pd = p if name.startswith("ladders") else final.predict_proba(Xd)[:, 1]
    print(f"calibration, {name}: bins of predicted probability -> realised accuracy (n): " + "; ".join(f"[{lo:.2f},{hi:.2f}) {yd[(pd >= lo) & (pd < hi)].mean() * 100:.0f}% ({int(((pd >= lo) & (pd < hi)).sum())})" for lo, hi in ((0, 0.5), (0.5, 0.9), (0.9, 0.98), (0.98, 0.995), (0.995, 1.01)) if ((pd >= lo) & (pd < hi)).any()))
