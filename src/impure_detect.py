"""Can certified links detect impure level-set pieces?  A link whose two end points (counts c >= 1 sheets apart) both lie on ONE piece says the piece spans two sheets.
Per piece: number of such links; truth purity from the dense truth as in bigeval.py (share of the largest consistent sub-piece, pieces >= 0.05 cm2 with >= 30% of vertices in the truth field).
  python impure_detect.py PIECES.npz LADDERS1.json [...]"""
import sys, json
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import isosurf as iso

d = np.load(sys.argv[1]); V0 = d["V"].astype(np.float64); F0 = d["F"]; voff = d["voff"]; foff = d["foff"]; area = d["area"]; npc = len(area)
piece = np.repeat(np.arange(npc), np.diff(voff)); tree = cKDTree(V0[::2]); pid = piece[::2]
ORG = np.array([10496, 1920, 3328.]); links = []
for f in sys.argv[2:]:
    for c in json.load(open(f))["collections"].values():
        pts = sorted(c["points"].values(), key=lambda q: q["wind_a"])
        for a, b in zip(pts[:-1], pts[1:]): links.append(([a["p"][2], a["p"][1], a["p"][0]], [b["p"][2], b["p"][1], b["p"][0]], b["wind_a"] - a["wind_a"]))
A = np.array([l[0] for l in links]); B = np.array([l[1] for l in links]); cc = np.array([l[2] for l in links])
da, ia = tree.query(A, distance_upper_bound=4.0, workers=-1); db, ib = tree.query(B, distance_upper_bound=4.0, workers=-1); oka = np.isfinite(da); okb = np.isfinite(db)
pa = np.where(oka, pid[np.minimum(ia, len(pid) - 1)], -1); pb = np.where(okb, pid[np.minimum(ib, len(pid) - 1)], -1)
both = oka & okb; same = both & (pa == pb); diff = both & (pa != pb)
print(f"links {len(links)}; both end points on a piece {both.sum()} ({both.mean() * 100:.1f}%): on the same piece {same.sum()}, on different pieces {diff.sum()}")
flag = np.bincount(pa[same], minlength=npc)
rows = []
for i in range(npc):
    if area[i] < 0.05: continue
    V = V0[voff[i]:voff[i + 1]]; F = F0[foff[i]:foff[i + 1]]; n = iso.turn_numbers(V); th = iso.theta_wrapped(V); cov = float(np.isfinite(n).mean())
    E = np.unique(np.sort(np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [0, 2]]], 1), axis=0); a, b = E[:, 0], E[:, 1]
    have = np.isfinite(n[a]) & np.isfinite(n[b]); bridge = have & ((n[b] - n[a]) != -np.rint((th[b] - th[a]) / (2 * np.pi))); keep = ~bridge
    nc, sub = connected_components(coo_matrix((np.ones(keep.sum()), (a[keep], b[keep])), shape=(len(V), len(V))), directed=False)
    ar = iso.tri_area(V, F) * iso.VOX_CM ** 2; sa = np.bincount(sub[F[:, 0]], ar, minlength=nc); rows.append((i, area[i], sa.max() / max(sa.sum(), 1e-12), cov, flag[i]))
R = np.array(rows); big = R[:, 3] >= 0.3; R = R[big]; A_, P_, F_ = R[:, 1], R[:, 2], R[:, 4]
imp = A_ * (1 - P_); print(f"scored pieces {len(R)}, area {A_.sum():.2f} cm2, impure area (area x (1 - purity)) {imp.sum():.2f} cm2 ({imp.sum() / A_.sum() * 100:.1f}%)")
for thr in (1, 2, 3):
    m = F_ >= thr; print(f"flagged by >= {thr} same-piece links: {int(m.sum())} pieces, area {A_[m].sum():.2f} cm2; purity of flagged (area-weighted) {(A_[m] * P_[m]).sum() / max(A_[m].sum(), 1e-9) * 100:.1f}% vs unflagged {(A_[~m] * P_[~m]).sum() / A_[~m].sum() * 100:.1f}%; share of all impure area in flagged pieces {imp[m].sum() / imp.sum() * 100:.1f}%; flagged pieces with purity < 0.9: {int((P_[m] < 0.9).sum())} of {int(m.sum())}")
np.savez("E:/vesuvius_impure_detect.npz", R=R)
