"""Piece-wise evaluation of a bigextract npz: seam-aware purity against the truth turn numbers (77f5 Theta field; valid where official segments exist),
area statistics, and SURF export (piece2surf grids) for surfkit.
  python bigeval.py PIECES.npz [OUT_SURF.npz] [MIN_SURF_AREA]
Purity of a piece = share of its (mesh-connected, consistent-edge) largest sub-piece; pieces with < 30% of their vertices inside the truth field are not scored."""
import sys
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import isosurf as iso
import piece2surf as p2s

d = np.load(sys.argv[1]); V0 = d["V"].astype(np.float64); F0 = d["F"]; voff = d["voff"]; foff = d["foff"]; k = d["k"]; area = d["area"]
out = sys.argv[2] if len(sys.argv) > 2 else None; amin_surf = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
rows = []; surfs = {}; names = []
for i in range(len(area)):
    V = V0[voff[i]:voff[i + 1]]; F = F0[foff[i]:foff[i + 1]]
    n = iso.turn_numbers(V); th = iso.theta_wrapped(V); cov = float(np.isfinite(n).mean())
    E = np.unique(np.sort(np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [0, 2]]], 1), axis=0); a, b = E[:, 0], E[:, 1]
    have = np.isfinite(n[a]) & np.isfinite(n[b]); bridge = have & ((n[b] - n[a]) != -np.rint((th[b] - th[a]) / (2 * np.pi))); keep = ~bridge
    nc, sub = connected_components(coo_matrix((np.ones(keep.sum()), (a[keep], b[keep])), shape=(len(V), len(V))), directed=False)
    ar = iso.tri_area(V, F) * iso.VOX_CM ** 2; sa = np.bincount(sub[F[:, 0]], ar, minlength=nc)
    rows.append((area[i], sa.max() / max(sa.sum(), 1e-12), cov, sa.max(), k[i]))
    if area[i] >= amin_surf and out:
        E_local = np.sort(np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [0, 2]]], 1); g = p2s.piece_grid(V, E_local)
        if g is not None: surfs[f"xyz_{len(names)}"] = g; names.append(f"b_{i}_k{k[i]}")
R = np.array(rows); A = R[:, 0]; P = R[:, 1]; C = R[:, 2]; big = (A >= 0.05) & (C >= 0.3)
print(f"{len(A)} pieces >= {A.min():.2f} cm2, total area {A.sum():.2f} cm2; largest pieces (cm2): {np.sort(A)[::-1][:8].round(2).tolist()}")
print(f"pieces >= 0.05 cm2 scored (>= 30% of vertices in the truth field): {big.sum()} of {(A >= 0.05).sum()}; area-weighted purity {(A[big] * P[big]).sum() / A[big].sum() * 100:.1f}% "
      f"(switch share {(1 - (A[big] * P[big]).sum() / A[big].sum()) * 100:.1f}%); pure area (largest consistent sub-piece) {R[big, 3].sum():.2f} cm2; largest pure sub-piece {R[:, 3].max():.2f} cm2")
for lo, hi in ((0.05, 0.1), (0.1, 0.2), (0.2, 0.5), (0.5, 1.0), (1.0, 100)):
    s = big & (A >= lo) & (A < hi)
    if s.any(): print(f"   area {lo}-{hi} cm2: {s.sum()} pieces, area {A[s].sum():.2f}, purity {(A[s] * P[s]).sum() / A[s].sum() * 100:.1f}%")
if out:
    np.savez_compressed(out, names=np.array(names), **surfs); print("saved SURF", out, len(names), "surfaces")
