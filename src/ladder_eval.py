"""Winding-count precision of an unwrapped phase field u against the human winding ladders (layerjudge/relative_windings.json, abs_winding.json).
A ladder collection is a line of points on consecutive windings with the integer winding index wind_a (relative: only differences count).  For every pair of points of one collection
that lies inside the volume of u, the label difference round(u_j) - round(u_i) must equal the winding difference wind_j - wind_i up to one global sign per collection (the unwrapped
direction is arbitrary).  Reported: pair precision for neighbouring windings (|dw| = 1), for all pairs, per-collection majority, coverage.
  python ladder_eval.py U.npy OZ OY OX [D=2] [LADDER.json ...]       point order in the json is (x, y, z) in L2 voxels"""
import os, sys, json
import numpy as np

uf = sys.argv[1]; org = np.array([float(v) for v in sys.argv[2:5]]); D = int(sys.argv[5]) if len(sys.argv) > 5 else 2
files = sys.argv[6:] if len(sys.argv) > 6 else [os.environ.get("LADDER_DIR", "layerjudge/") + "relative_windings.json"]
u = np.load(uf, mmap_mode="r"); shp = np.array(u.shape)
cols = []
for f in files:
    for name, c in json.load(open(f))["collections"].items():
        pts = np.array([p["p"] for p in c["points"].values()], float); w = np.array([p["wind_a"] for p in c["points"].values()], float)
        if len(pts) >= 2: cols.append((f"{f.split('/')[-1]}:{name}", pts, w))
tot_pairs = ok_pairs = 0; n1 = ok1 = 0; used = 0; col_ok = []; rows = []
for name, pts, w in cols:
    zyx = pts[:, [2, 1, 0]]; ix = np.rint((zyx - org - 0.5) / D).astype(int); inside = np.all((ix >= 0) & (ix < shp), 1)
    if inside.sum() < 2: continue
    ix = ix[inside]; ww = w[inside]; val = np.array([float(u[tuple(i)]) for i in ix]); lab = np.rint(val)
    order = np.argsort(ww); ww = ww[order]; lab = lab[order]; val = val[order]
    dw = ww[:, None] - ww[None, :]; dl = lab[:, None] - lab[None, :]; iu = np.triu_indices(len(ww), 1); dw = dw[iu]; dl = dl[iu]
    nz = dw != 0
    if nz.sum() == 0: continue
    s = 1 if (np.sum(dl[nz] == dw[nz]) >= np.sum(dl[nz] == -dw[nz])) else -1                  # one sign per collection
    good = (dl == s * dw) & nz; adj = nz & (np.abs(dw) == 1)
    tot_pairs += int(nz.sum()); ok_pairs += int(good.sum()); n1 += int(adj.sum()); ok1 += int((good & adj).sum()); used += 1
    col_ok.append(good[nz].mean()); rows.append((name, len(ww), int(nz.sum()), float(good[nz].mean())))
print(f"{uf.split('/')[-1]}: collections with >= 2 points inside: {used} of {len(cols)}; pairs {tot_pairs} (neighbouring windings {n1})")
if tot_pairs:
    print(f"  pair precision, all pairs {ok_pairs / tot_pairs * 100:.1f}%  |  neighbouring windings {ok1 / max(n1, 1) * 100:.1f}%  |  collections fully right {np.mean(np.array(col_ok) == 1.0) * 100:.0f}%  |  per-collection mean {np.mean(col_ok) * 100:.1f}%")
