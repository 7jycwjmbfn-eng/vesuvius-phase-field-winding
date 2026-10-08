"""Level-set pieces (iso npz: V absolute L2 zyx, F, lab) -> SURF npz for surfkit_8c (xyz_<i> float32 [H, W, 3], L3 zyx, NaN = invalid).
Parameterisation: unwrapped azimuth about the umbilicus (accumulated along the mesh edges) x mean radius, and z; cells of H3 L2 voxels, cell value = mean vertex position.
  python piece2surf.py ISO.npz OUT.npz [MIN_AREA_CM2] [MAX_PIECES]"""
import sys
import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order
import isosurf as iso
import train_wfield_8c as tw

H3 = 3.0
uz, uy, ux = tw.umbilicus()


def piece_grid(V, E):
    """V [n,3] L2 zyx of one piece, E [m,2] local edges -> grid [H, W, 3] (L3 zyx) or None"""
    n = len(V); cy = np.interp(V[:, 0], uz, uy); cx = np.interp(V[:, 0], uz, ux); a = np.arctan2(V[:, 1] - cy, V[:, 2] - cx); r = np.hypot(V[:, 1] - cy, V[:, 2] - cx)
    g = coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(n, n)).tocsr(); order, pred = breadth_first_order(g, 0, directed=False, return_predecessors=True)
    au = np.zeros(n); d = np.angle(np.exp(1j * (a[order[1:]] - a[pred[order[1:]]])))
    for i, node in enumerate(order[1:]): au[node] = au[pred[node]] + d[i]                  # sequential: parent value is known before the child
    seen = np.zeros(n, bool); seen[order] = True
    if seen.sum() < 100: return None
    s = au * np.median(r[seen]); t = V[:, 0]; i0 = np.floor((t[seen] - t[seen].min()) / H3).astype(int); j0 = np.floor((s[seen] - s[seen].min()) / H3).astype(int)
    Hh, Ww = i0.max() + 1, j0.max() + 1
    if Hh * Ww > 6e7: return None
    flat = i0 * Ww + j0; cnt = np.bincount(flat, minlength=Hh * Ww).astype(float); grid = np.full((Hh * Ww, 3), np.nan)
    for k in range(3): grid[:, k] = np.bincount(flat, V[seen][:, k], minlength=Hh * Ww)
    grid = grid.reshape(Hh, Ww, 3); cnt = cnt.reshape(Hh, Ww); grid = np.where(cnt[..., None] > 0, grid / np.maximum(cnt[..., None], 1), np.nan)
    for _ in range(2):                                                                   # fill one-cell holes from valid neighbours
        ok = np.isfinite(grid[..., 0]); nb = ndi.uniform_filter(ok.astype(float), 3) * 9
        for k in range(3):
            tot = ndi.uniform_filter(np.where(ok, grid[..., k], 0.0), 3) * 9; fill = ~ok & (nb >= 4); grid[..., k] = np.where(fill, tot / np.maximum(nb, 1), grid[..., k])
    return (grid / 2.0).astype(np.float32)                                              # L2 -> L3


if __name__ == "__main__":
    d = np.load(sys.argv[1]); V = d["V"].astype(np.float64); F = d["F"]; lab = d["lab"]; out = sys.argv[2]
    amin = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05; maxp = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    ar = iso.tri_area(V, F) * iso.VOX_CM ** 2; pa = np.bincount(lab[F[:, 0]], ar); order = [k for k in np.argsort(-pa) if pa[k] >= amin][:maxp]
    fe = np.sort(np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [0, 2]]], 1); fl = lab[fe[:, 0]]
    surfs = {}; names = []
    for k in order:
        vi = np.nonzero(lab == k)[0]; mp = np.full(len(V), -1, np.int64); mp[vi] = np.arange(len(vi)); e = fe[fl == k]; E = np.c_[mp[e[:, 0]], mp[e[:, 1]]]
        gr = piece_grid(V[vi], E)
        if gr is None: continue
        surfs[f"xyz_{len(names)}"] = gr; names.append(f"lv_{k}"); print(f"piece {k}: area {pa[k]:.3f} cm2 -> grid {gr.shape[:2]}, valid cells {np.isfinite(gr[..., 0]).mean() * 100:.0f}%", flush=True)
    np.savez_compressed(out, names=np.array(names), **surfs); print("saved", out, len(names), "surfaces")
