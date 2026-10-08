"""Fill interior holes of surface grids (SURF npz, xyz_<i> = float32 [H, W, 3], NaN = invalid) by harmonic interpolation of the coordinates from the valid cells around the hole.
A hole is a 4-connected set of invalid cells that does not touch the grid border and has at most MAXCELLS cells.  Larger holes and holes open to the border stay.
The output has the same format; the number of filled cells is printed per run (filled area is reported separately in every evaluation).
  python fillholes.py IN.npz OUT.npz MAXCELLS"""
import sys
import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


def fill_grid(P, maxcells):
    valid = np.isfinite(P).all(-1); H, W = valid.shape
    lab, n = ndi.label(~valid)                                                            # 4-connected invalid components
    if n == 0: return P, 0
    sizes = np.bincount(lab.ravel(), minlength=n + 1); border = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]]).tolist())
    out = P.copy(); filled = 0
    for c in range(1, n + 1):
        if c in border or sizes[c] > maxcells or sizes[c] < 1: continue
        cells = np.argwhere(lab == c); idx = {(int(a), int(b)): i for i, (a, b) in enumerate(cells)}; m = len(cells)
        rows = []; cols = []; vals = []; rhs = np.zeros((m, 3)); ok = True
        for i, (a, b) in enumerate(cells):
            nb = 0
            for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                a2, b2 = a + da, b + db
                if not (0 <= a2 < H and 0 <= b2 < W): continue
                nb += 1
                if (a2, b2) in idx: rows.append(i); cols.append(idx[(a2, b2)]); vals.append(-1.0)
                elif valid[a2, b2]: rhs[i] += P[a2, b2]
                else: ok = False                                                          # a neighbour that is invalid but in another component cannot happen (4-connected labels)
            rows.append(i); cols.append(i); vals.append(float(nb))
        if not ok: continue
        A = coo_matrix((vals, (rows, cols)), shape=(m, m)).tocsc()
        try: sol = np.stack([spsolve(A, rhs[:, k]) for k in range(3)], 1)
        except Exception: continue
        if not np.isfinite(sol).all(): continue
        out[cells[:, 0], cells[:, 1]] = sol; filled += m
    return out, filled


def main(src, dst, maxcells):
    d = np.load(src, allow_pickle=True); names = d["names"] if "names" in d.files else None; out = {}; tot = 0; tot_cells = 0
    for k in d.files:
        if not k.startswith("xyz_"): continue
        P, f = fill_grid(d[k].astype(np.float64), maxcells); out[k] = P.astype(np.float32); tot += f; tot_cells += int(np.isfinite(P).all(-1).sum())
    if names is not None: out["names"] = names
    np.savez_compressed(dst, **out); print(f"filled {tot} cells (valid cells before: {tot_cells}, +{tot / max(tot_cells, 1) * 100:.1f}%), max hole {maxcells} cells -> {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
