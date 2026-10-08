"""Cut the bridges of wrapped-phase surface pieces with the tile-unwrapping labels, without losing the area at the sync seams.
The wrapped psi = 0 pieces (bigextract2 with WRAPPED=1) are continuous across the tile seams, but where the sheet phase has a vortex a piece can pass from one sheet to the
next (a bridge).  Every vertex gets the integer label l = round(u) of the synced unwrapped field and the tile whose core contains it.  A mesh edge (a, b) is kept when
  l_b - l_a == 0, or (a and b lie in different tiles that form an overlap pair of the LP, and l_b - l_a == r_AB, the LP residual of that pair, i.e. the relabelling the
  local overlap data demands at that seam).  Every other edge joins two different sheets (or is noise) and is cut.  Faces keep only if all three edges are kept.
  python labelcut.py PIECES.npz U.npy PAIRS.npz OUT.npz OZ OY OX [MIN_AREA]       (D = 2, T = 48, S = 24, as bigsync2)"""
import os, sys, time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import bigsync2 as bs

pf, uf, paf, outf = sys.argv[1:5]; ORG = np.array([float(v) for v in sys.argv[5:8]]); AMIN = float(sys.argv[8]) if len(sys.argv) > 8 else 0.02
D = 2; T = 48; S = 24; VOX_CM = 9.6e-4; STRICT = bool(os.environ.get("STRICT"))
d = np.load(pf); V0 = d["V"]; F0 = d["F"]; voff = d["voff"]; foff = d["foff"]; k0 = d["k"]; area0 = d["area"]
u = np.load(uf, mmap_mode="r"); SHAPE = u.shape
P = np.load(paf); rows, cols, dd, ww, n = P["rows"], P["cols"], P["dd"], P["ww"], int(P["n"])
p = bs.irls_sync(n, rows, cols, dd, ww); res = p[cols] - p[rows] - dd
rmap = {}
for r_, c_, v in zip(rows, cols, res): rmap[(int(r_), int(c_))] = int(v); rmap[(int(c_), int(r_))] = -int(v)
st = [bs.starts(N, T, S) for N in SHAPE]; nt = [len(s) for s in st]; cen = [np.array(s) + T / 2.0 for s in st]; near = [np.abs(np.arange(N)[:, None] - cen[a][None]).argmin(1) for a, N in enumerate(SHAPE)]
print(f"{len(area0)} wrapped pieces; tile pairs {len(rows)}, residual != 0 in {(res != 0).mean() * 100:.1f}%", flush=True)


def tri_area(V, F):
    return 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)


Vs_out = []; Fs_out = []; ks = []; ars = []; t0 = time.time(); kept_e = tot_e = 0
for i in range(len(area0)):
    V = V0[voff[i]:voff[i + 1]].astype(np.float64); F = F0[foff[i]:foff[i + 1]]; nv = len(V)
    ix = np.clip(np.rint((V - ORG - 0.5) / D).astype(int), 0, np.array(SHAPE) - 1)
    lab = np.rint(np.asarray(u[ix[:, 0], ix[:, 1], ix[:, 2]])).astype(np.int64)
    tile = (near[0][ix[:, 0]] * nt[1] + near[1][ix[:, 1]]) * nt[2] + near[2][ix[:, 2]]
    fe = np.sort(np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [0, 2]]], 1); key = fe[:, 0].astype(np.int64) * nv + fe[:, 1]; uk = np.unique(key); a = uk // nv; b = uk % nv; dl = lab[b] - lab[a]
    ok = dl == 0; diff = (~ok) & (tile[a] != tile[b]) & (not STRICT)                                       # STRICT=1: cut every edge with a label change (seams included); kjoin2 re-joins them with its conflict filter
    for j in np.nonzero(diff)[0]:
        r_ = rmap.get((int(tile[a[j]]), int(tile[b[j]])))
        if r_ is not None and r_ != 0 and dl[j] == r_: ok[j] = True
    kept_e += int(ok.sum()); tot_e += len(ok)
    nf = len(F); fok = ok[np.searchsorted(uk, key)]; fk = fok[:nf] & fok[nf:2 * nf] & fok[2 * nf:]                      # a face keeps only if its three edges are kept
    if not fk.any(): continue
    Fk = F[fk]; nc, comp = connected_components(coo_matrix((np.ones(len(Fk) * 2), (np.r_[Fk[:, 0], Fk[:, 1]], np.r_[Fk[:, 1], Fk[:, 2]])), shape=(nv, nv)), directed=False)
    ar = tri_area(V, Fk) * VOX_CM ** 2; pa = np.bincount(comp[Fk[:, 0]], ar, minlength=nc)
    for c in np.nonzero(pa >= AMIN)[0]:
        fi = np.nonzero(comp[Fk[:, 0]] == c)[0]; vi = np.unique(Fk[fi]); mp = np.full(nv, -1, np.int64); mp[vi] = np.arange(len(vi))
        Vs_out.append(V[vi].astype(np.float32)); Fs_out.append(mp[Fk[fi]].astype(np.int32)); ks.append(int(np.bincount(np.clip(lab[vi] - lab[vi].min(), 0, 1000)).argmax() + lab[vi].min())); ars.append(float(pa[c]))
    if i % 20 == 0: print(f"  piece {i}/{len(area0)} {time.time() - t0:.0f} s", flush=True)
ars = np.array(ars); o = np.argsort(-ars)
print(f"edges kept {kept_e / tot_e * 100:.1f}%; {len(area0)} wrapped pieces ({area0.sum():.2f} cm2) -> {len(ars)} consistent pieces ({ars.sum():.2f} cm2); largest {ars[o[:8]].round(2).tolist()}", flush=True)
voffs = np.r_[0, np.cumsum([len(v) for v in Vs_out])]; foffs = np.r_[0, np.cumsum([len(f) for f in Fs_out])]
np.savez_compressed(outf, V=np.concatenate(Vs_out), F=np.concatenate(Fs_out), voff=voffs, foff=foffs, k=np.array(ks), area=ars); print("saved", outf)
