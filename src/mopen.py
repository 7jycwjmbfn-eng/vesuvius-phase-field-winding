"""Morphological opening of the level-set pieces on the mesh: erode every piece by R (L2 voxels, geodesic distance to the mesh boundary), split it into the connected
components that remain, then grow each component back by R inside the original mesh (nearest-component assignment, only within R).  Connections narrower than about 2R
(thin bridges between neighbouring sheets, vortex necks) are cut; wide parts are unchanged.  Components smaller than MIN_AREA cm2 are dropped.
  python mopen.py PIECES.npz R OUT.npz [MIN_AREA]"""
import sys, time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

VOX_CM = 9.6e-4


def tri_area(V, F):
    return 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)


def open_piece(V, F, R, amin):
    n = len(V); e = np.sort(np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [0, 2]]], 1); key = e[:, 0].astype(np.int64) * n + e[:, 1]
    uk, cnt = np.unique(key, return_counts=True); a = uk // n; b = uk % n; w = np.linalg.norm(V[a] - V[b], axis=1) + 1e-6
    bnd = np.zeros(n, bool); sel = cnt == 1; bnd[a[sel]] = True; bnd[b[sel]] = True
    G = coo_matrix((np.r_[w, w], (np.r_[a, b], np.r_[b, a])), shape=(n, n)).tocsr()
    if not bnd.any(): return [(V, F)]
    d = dijkstra(G, directed=False, indices=np.nonzero(bnd)[0], min_only=True, limit=R * 1.01)
    keep = d >= R                                                                              # eroded core (inf where farther than the limit)
    if not keep.any(): return []
    fk = keep[F].all(1); Fk = F[fk]
    if len(Fk) == 0: return []
    nc, lab = connected_components(coo_matrix((np.ones(len(Fk) * 2), (np.r_[Fk[:, 0], Fk[:, 1]], np.r_[Fk[:, 1], Fk[:, 2]])), shape=(n, n)), directed=False)
    lab = np.where(keep, lab, -1)
    # grow back: nearest kept vertex within R
    dd, pred, src = dijkstra(G, directed=False, indices=np.nonzero(keep)[0], min_only=True, return_predecessors=True, limit=R * 1.01)
    full = np.where(keep, lab, np.where((dd <= R * 1.01) & (src >= 0), lab[np.clip(src, 0, n - 1)], -1))
    lf = full[F]; ok = (lf[:, 0] >= 0) & (lf[:, 0] == lf[:, 1]) & (lf[:, 1] == lf[:, 2]); out = []
    ar = tri_area(V, F) * VOX_CM ** 2
    for c in np.unique(lf[ok, 0]):
        fi = np.nonzero(ok & (lf[:, 0] == c))[0]
        if ar[fi].sum() < amin: continue
        vi = np.unique(F[fi]); mp = np.full(n, -1, np.int64); mp[vi] = np.arange(len(vi)); out.append((V[vi], mp[F[fi]].astype(np.int32)))
    return out


def main(pf, R, outf, amin=0.02):
    d = np.load(pf); V0 = d["V"]; F0 = d["F"]; voff = d["voff"]; foff = d["foff"]; k = d["k"]; area = d["area"]; t0 = time.time()
    Vs = []; Fs = []; ks = []; ars = []
    for i in range(len(area)):
        V = V0[voff[i]:voff[i + 1]].astype(np.float64); F = F0[foff[i]:foff[i + 1]]
        for v, f in open_piece(V, F, R, amin):
            Vs.append(v.astype(np.float32)); Fs.append(f); ks.append(k[i]); ars.append(tri_area(v, f).sum() * VOX_CM ** 2)
        if i % 100 == 0: print(f"  piece {i}/{len(area)}, {time.time() - t0:.0f} s", flush=True)
    ars = np.array(ars); order = np.argsort(-ars); print(f"{len(area)} pieces -> {len(ars)} after opening R={R}; total area {area.sum():.2f} -> {ars.sum():.2f} cm2; largest {ars[order[:6]].round(2).tolist()}", flush=True)
    voffs = np.r_[0, np.cumsum([len(v) for v in Vs])]; foffs = np.r_[0, np.cumsum([len(f) for f in Fs])]
    np.savez_compressed(outf, V=np.concatenate(Vs), F=np.concatenate(Fs), voff=voffs, foff=foffs, k=np.array(ks), area=ars)


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]), sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else 0.02)
