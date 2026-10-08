"""Truth-free join of pieces across tile-offset discontinuities.  A sheet that crosses the boundary between two tile cores A -> B whose LP offsets are inconsistent by
r = p_B - p_A - d_AB is relabelled k -> k + r.  Two pieces are joined when (i) their junction vertices (within R voxels) lie in the cores of two different tiles that
form an overlap pair of the LP, (ii) that pair has residual r != 0 and (iii) k_B - k_A = r.   Evaluated against the truth turn numbers.
  python kjoin2.py PIECES.npz PAIRS.npz R [MIN_PAIRS] [OUT_SURF.npz]      (D = 2, T = 48, S = 24, origin = the big box)"""
import sys, os
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import isosurf as iso
import bigsync2 as bs

d = np.load(sys.argv[1]); P = np.load(sys.argv[2]); R = float(sys.argv[3]); MINP = int(sys.argv[4]) if len(sys.argv) > 4 else 2
ORG = np.array([float(v) for v in os.environ.get("BOX_ORG", "10496,1920,3328").split(",")]); D = 2; T = 48; S = 24; SHAPE = (256, 1536, 1536)
rows, cols, dd, ww, n = P["rows"], P["cols"], P["dd"], P["ww"], int(P["n"])
p = bs.irls_sync(n, rows, cols, dd, ww); res = p[cols] - p[rows] - dd                              # the same solution as the run (IRLS + ICM is deterministic)
print(f"tile pairs {len(rows)}, with residual != 0: {(res != 0).sum()} ({(res != 0).mean() * 100:.1f}%), residual values {dict(zip(*np.unique(res, return_counts=True)))}", flush=True)
st = [bs.starts(N, T, S) for N in SHAPE]; nt = [len(s) for s in st]
cen = [np.array(s) + T / 2.0 for s in st]; near = [np.abs(np.arange(N)[:, None] - cen[a][None]).argmin(1) for a, N in enumerate(SHAPE)]
rmap = {}
for r_, c_, v in zip(rows, cols, res): rmap[(int(r_), int(c_))] = int(v); rmap[(int(c_), int(r_))] = -int(v)        # pair (A -> B) residual r;  (B -> A) = -r
V = d["V"]; voff = d["voff"]; k = d["k"]; area = d["area"]; npieces = len(k); piece = np.repeat(np.arange(npieces), np.diff(voff))
sel = np.arange(0, len(V), 2); Vs = V[sel].astype(np.float64); ps = piece[sel]
tree = cKDTree(Vs); pr = tree.query_pairs(R, output_type="ndarray"); a, b = pr[:, 0], pr[:, 1]; m = ps[a] != ps[b]; a, b = a[m], b[m]
idx = np.rint((Vs[np.unique(np.r_[a, b])] - ORG - 0.5) / D).astype(int); uvs = np.unique(np.r_[a, b])
tile = {}
for v_, ix in zip(uvs, idx):
    ix = np.clip(ix, 0, np.array(SHAPE) - 1); tile[v_] = (int(near[0][ix[0]]) * nt[1] + int(near[1][ix[1]])) * nt[2] + int(near[2][ix[2]])
fa = np.array([tile[x] for x in a]); fb = np.array([tile[x] for x in b])
valid = np.zeros(len(a), bool)
for i in range(len(a)):
    if fa[i] == fb[i]: continue
    r_ = rmap.get((int(fa[i]), int(fb[i])))
    if r_ is None or r_ == 0: continue
    if k[ps[b[i]]] - k[ps[a[i]]] == r_: valid[i] = True
print(f"junction vertex pairs of different pieces within {R} voxels: {len(a)}; in different tiles with a nonzero-residual overlap pair and dk == r: {valid.sum()}", flush=True)
a, b = a[valid], b[valid]
ua = np.unique(np.r_[a, b]); nn = np.full(len(Vs), np.nan); th = np.full(len(Vs), np.nan); nn[ua] = iso.turn_numbers(Vs[ua]); th[ua] = iso.theta_wrapped(Vs[ua])
have = np.isfinite(nn[a]) & np.isfinite(nn[b]); cons = np.zeros(len(a), bool); cons[have] = (nn[b[have]] - nn[a[have]]) == -np.rint((th[b[have]] - th[a[have]]) / (2 * np.pi))
lo = np.minimum(ps[a], ps[b]); hi = np.maximum(ps[a], ps[b]); key = lo.astype(np.int64) * npieces + hi; uk, inv = np.unique(key, return_inverse=True)
cnt = np.bincount(inv); nh = np.bincount(inv, have); nc = np.bincount(inv, cons & have); plo = uk // npieces; phi = uk % npieces; join = cnt >= MINP
CONF = float(os.environ.get("CONFLICT", "-1"))          # stack-conflict threshold: share of overlapping (azimuth, z) cells with |dr| > 4 voxels; -1 = filter off
if CONF >= 0:
    uz, uy, ux = iso.tw.umbilicus(); cellcache = {}
    def cells(q):
        if q in cellcache: return cellcache[q]
        v = V[voff[q]:voff[q + 1]:3].astype(np.float64); cy = np.interp(v[:, 0], uz, uy); cx = np.interp(v[:, 0], uz, ux)
        az = (np.degrees(np.arctan2(v[:, 1] - cy, v[:, 2] - cx)) % 360) / 0.5; r = np.hypot(v[:, 1] - cy, v[:, 2] - cx)
        key = az.astype(np.int64) * 256 + (v[:, 0] // 8).astype(np.int64) % 256; o = np.argsort(key); key = key[o]; r = r[o]
        uq, st_ = np.unique(key, return_index=True); rm = np.add.reduceat(r, st_) / np.diff(np.r_[st_, len(key)]); cellcache[q] = (uq, rm); return cellcache[q]
    rej = 0
    for i in np.nonzero(join)[0]:
        ka_, ra_ = cells(int(plo[i])); kb_, rb_ = cells(int(phi[i])); common, ia, ib = np.intersect1d(ka_, kb_, return_indices=True)
        if len(common) >= 20 and (np.abs(ra_[ia] - rb_[ib]) > 4).mean() > CONF: join[i] = False; rej += 1
    print(f"stack-conflict filter (> {CONF * 100:.0f}% of >= 20 overlapping cells with |dr| > 4 voxels): rejected {rej} of the candidate joins", flush=True)
tr = join & (nh > 0) & (nc >= 0.5 * np.maximum(nh, 1))
print(f"joined piece pairs (>= {MINP} valid junction pairs): {join.sum()}; with truth {int((join & (nh > 0)).sum())}; true continuations {int(tr.sum())}, false {int((join & (nh > 0) & ~tr).sum())}", flush=True)
G = coo_matrix((np.ones(join.sum()), (plo[join], phi[join])), shape=(npieces, npieces)); ncomp, grp = connected_components(G, directed=False); ga = np.bincount(grp, area, minlength=ncomp)
Tg = coo_matrix((np.ones(tr.sum()), (plo[tr], phi[tr])), shape=(npieces, npieces)); nt_, tcomp = connected_components(Tg, directed=False); ta = np.bincount(tcomp, area, minlength=nt_)
scored = np.zeros(ncomp, bool); pur = np.zeros(ncomp); mx = np.zeros(ncomp)
for g in np.nonzero(ga >= 0.05)[0]:
    mem = np.nonzero(grp == g)[0]
    if len(mem) == 1: continue
    tc = np.unique(tcomp[mem]); sa = ta[tc]; pur[g] = sa.max() / sa.sum(); mx[g] = sa.max(); scored[g] = True
big = np.argsort(-ga)[:10]; s = scored
print(f"groups {ncomp} from {npieces} pieces; largest areas {ga[big].round(2).tolist()}")
if s.any(): print(f"multi-piece groups >= 0.05 cm2: {s.sum()}, area {ga[s].sum():.1f} cm2, area-weighted purity {(ga[s] * pur[s]).sum() / ga[s].sum() * 100:.1f}%, largest pure group {mx[s].max():.2f} cm2, largest group {ga.max():.2f} cm2")
if len(sys.argv) > 5:
    import piece2surf as p2s
    F = d["F"]; foff = d["foff"]; surfs = {}; names = []; Ja = sel[a][join[inv]]; Jb = sel[b][join[inv]]
    for g in np.argsort(-ga):
        if ga[g] < 0.05: break
        mem = np.nonzero(grp == g)[0]; goff = {}; vs = []; fs = []; off = 0
        for q in mem:
            goff[q] = off; vs.append(V[voff[q]:voff[q + 1]].astype(np.float64)); fs.append(F[foff[q]:foff[q + 1]] + off); off += voff[q + 1] - voff[q]
        Vg = np.concatenate(vs); Fg = np.concatenate(fs); E = np.sort(np.r_[Fg[:, [0, 1]], Fg[:, [1, 2]], Fg[:, [0, 2]]], 1)
        if len(mem) > 1:
            m_ = np.isin(piece[Ja], mem) & np.isin(piece[Jb], mem)
            if m_.any():
                ea = np.array([goff[piece[x]] + (x - voff[piece[x]]) for x in Ja[m_]]); eb = np.array([goff[piece[x]] + (x - voff[piece[x]]) for x in Jb[m_]]); E = np.r_[E, np.sort(np.c_[ea, eb], 1)]
        gr = p2s.piece_grid(Vg, E)
        if gr is not None: surfs[f"xyz_{len(names)}"] = gr; names.append(f"grp_{g}_{len(mem)}p")
    np.savez_compressed(sys.argv[5], names=np.array(names), **surfs); print("saved SURF", sys.argv[5], len(names), "surfaces", flush=True)

import os
if os.environ.get("DEBUG_GROUPS"):
    for g in (int(x) for x in os.environ["DEBUG_GROUPS"].split(",")):
        mem = np.nonzero(grp == g)[0]; tc = tcomp[mem]
        print(f"group {g}: {len(mem)} pieces, area {ga[g]:.2f} cm2, purity {pur[g] * 100:.0f}% (scored {scored[g]}); true-continuation components {len(np.unique(tc))}")
        for q in mem[np.argsort(-area[mem])]:
            vq = V[voff[q]:voff[q + 1]]; print(f"    piece {q:4d} k {k[q]:4d} area {area[q]:.3f} cm2  true component {tcomp[q]}  z {vq[:, 0].min():.0f}-{vq[:, 0].max():.0f}  y {vq[:, 1].min():.0f}-{vq[:, 1].max():.0f}  x {vq[:, 2].min():.0f}-{vq[:, 2].max():.0f}")
