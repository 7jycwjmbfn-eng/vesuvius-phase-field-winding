"""Where do the level-set pieces end?  For every mesh boundary vertex of the kept pieces of a big box, record
  - the distance to the box edge (L2 voxels),
  - the psi confidence, the CT value and the surface-prediction value at the vertex,
  - |grad u| (per L2 voxel, from the 2x down-sampled unwrapped field),
  - whether another piece with |dk| <= 1 lies within R voxels (a join candidate) and whether any other piece does.
Boundary length is counted in vertices (subsampled by STEP).  Output: area-weighted fractions per class.
  python breakdiag.py PIECES.npz U.npy CONF.npy CT.npy SP.npy OZ OY OX [MINA] [R]"""
import sys
import numpy as np
from scipy.spatial import cKDTree

pf, uf, cf, ctf, spf = sys.argv[1:6]; org = np.array([float(x) for x in sys.argv[6:9]]); MINA = float(sys.argv[9]) if len(sys.argv) > 9 else 0.05; R = float(sys.argv[10]) if len(sys.argv) > 10 else 6.0
STEP = 4
d = np.load(pf); V = d["V"]; F = d["F"]; voff = d["voff"]; foff = d["foff"]; k = d["k"]; area = d["area"]; n = len(k)
u = np.load(uf, mmap_mode="r"); cf_ = np.load(cf, mmap_mode="r"); ct = np.load(ctf, mmap_mode="r"); sp = np.load(spf, mmap_mode="r")
shape2 = np.array(ct.shape); sel = np.nonzero(area >= MINA)[0]
print(f"{n} pieces, {len(sel)} with area >= {MINA} cm2, area share {area[sel].sum() / area.sum() * 100:.1f}%", flush=True)

# boundary vertices of selected pieces
BV = []; BP = []
for p in sel:
    f = F[foff[p]:(foff[p + 1] if p + 1 < n else len(F))]
    nv = (voff[p + 1] if p + 1 < n else len(V)) - voff[p]
    e = np.sort(np.r_[f[:, [0, 1]], f[:, [1, 2]], f[:, [0, 2]]], 1); key = e[:, 0].astype(np.int64) * nv + e[:, 1]
    uk, cnt = np.unique(key, return_counts=True); bk = uk[cnt == 1]; bv = np.unique(np.r_[bk // nv, bk % nv])[::STEP]
    BV.append(bv + voff[p]); BP.append(np.full(len(bv), p))
BV = np.concatenate(BV); BP = np.concatenate(BP); print(f"boundary vertices (every {STEP}th): {len(BV)}", flush=True)

P = V[BV].astype(np.float64) - org[None]                                                    # box-local L2 coords
edge = np.minimum(P, shape2[None] - 1 - P).min(1)
idx2 = np.clip(np.rint(P / 2).astype(int), 0, np.array(u.shape) - 1)
conf = np.asarray(cf_[idx2[:, 0], idx2[:, 1], idx2[:, 2]]).astype(np.float32)
idx1 = np.clip(np.rint(P).astype(int), 0, shape2 - 1)
ctv = np.asarray(ct[idx1[:, 0], idx1[:, 1], idx1[:, 2]]).astype(np.float32); spv = np.asarray(sp[idx1[:, 0], idx1[:, 1], idx1[:, 2]]).astype(np.float32)
hi = np.array(u.shape) - 2
ic = np.clip(idx2, 1, hi); g = np.zeros(len(P))
for ax in range(3):
    ip = ic.copy(); im = ic.copy(); ip[:, ax] += 1; im[:, ax] -= 1
    g += ((np.asarray(u[ip[:, 0], ip[:, 1], ip[:, 2]]) - np.asarray(u[im[:, 0], im[:, 1], im[:, 2]])) / 4.0) ** 2                  # per L2 voxel
g = np.sqrt(g)

# neighbours: other pieces within R of the boundary vertex (all vertices of the selected pieces, every 2nd)
allv = np.concatenate([np.arange(voff[p], voff[p + 1] if p + 1 < n else len(V), 2) for p in sel]); allp = np.repeat(sel, [len(range(voff[p], voff[p + 1] if p + 1 < n else len(V), 2)) for p in sel])
tree = cKDTree(V[allv].astype(np.float64)); near = tree.query_ball_point(V[BV].astype(np.float64), R)
jc = np.zeros(len(BV), bool); oth = np.zeros(len(BV), bool); samek = np.zeros(len(BV), bool)
for i, lst in enumerate(near):
    if not lst: continue
    pp = allp[np.array(lst)]; m = pp != BP[i]
    if not m.any(): continue
    oth[i] = True; dk = np.abs(k[pp[m]] - k[BP[i]]); jc[i] = (dk <= 1).any()
print(f"\nboundary vertices: at box edge (< 24 L2 vox) {np.mean(edge < 24) * 100:.1f}%;  next to another piece with |dk|<=1 within {R:g} vox: {jc.mean() * 100:.1f}%;  next to a piece with |dk|>1 only: {(oth & ~jc).mean() * 100:.1f}%")
inner = (edge >= 24) & ~oth
print(f"free ends (not at the box edge, no piece within {R:g} vox): {inner.mean() * 100:.1f}% of boundary vertices")
q = lambda a: np.percentile(a, [10, 50, 90]).round(3).tolist()
allsel = np.concatenate([np.arange(voff[p], voff[p + 1] if p + 1 < n else len(V), 8) for p in sel]); Pa = V[allsel].astype(np.float64) - org[None]
ia2 = np.clip(np.rint(Pa / 2).astype(int), 0, np.array(u.shape) - 1); ia1 = np.clip(np.rint(Pa).astype(int), 0, shape2 - 1)
confa = np.asarray(cf_[ia2[:, 0], ia2[:, 1], ia2[:, 2]]).astype(np.float32); cta = np.asarray(ct[ia1[:, 0], ia1[:, 1], ia1[:, 2]]).astype(np.float32); spa = np.asarray(sp[ia1[:, 0], ia1[:, 1], ia1[:, 2]]).astype(np.float32)
ica = np.clip(ia2, 1, hi); ga = np.zeros(len(Pa))
for ax in range(3):
    ip = ica.copy(); im = ica.copy(); ip[:, ax] += 1; im[:, ax] -= 1
    ga += ((np.asarray(u[ip[:, 0], ip[:, 1], ip[:, 2]]) - np.asarray(u[im[:, 0], im[:, 1], im[:, 2]])) / 4.0) ** 2
ga = np.sqrt(ga)
print("\nquantities at (p10, p50, p90):             interior vertices            |  free-end vertices           |  join-candidate boundary")
for name, a, b in (("conf", confa, conf), ("CT", cta, ctv), ("surface pred", spa, spv), ("|grad u| per L2 vox", ga, g)):
    print(f"  {name:22s} {q(a)}  |  {q(b[inner]) if inner.any() else '-'}  |  {q(b[jc]) if jc.any() else '-'}")
for lo, hi_ in ((0, 1000), (1000, 1500), (1500, 2000), (2000, 3000)):
    pass
# radius from the umbilicus (y 3457, x 4879 in L2 of the whole scroll; box origin given) to see whether free ends sit far from the core
r = np.hypot(P[:, 1] + org[1] - 3457, P[:, 2] + org[2] - 4879); ra = np.hypot(Pa[:, 1] + org[1] - 3457, Pa[:, 2] + org[2] - 4879)
print("\nfree-end share by radius from the umbilicus (L2 vox):  r range: all-vertex share | free-end share of boundary vertices")
for lo, hi_ in ((0, 500), (500, 1000), (1000, 1500), (1500, 2500)):
    m = (r >= lo) & (r < hi_); ma = (ra >= lo) & (ra < hi_)
    print(f"  {lo:5d}-{hi_:5d}: {ma.mean() * 100:5.1f}% of piece vertices | {m.sum()} boundary verts, free-end {inner[m].mean() * 100 if m.any() else 0:.1f}%, join-candidate {jc[m].mean() * 100 if m.any() else 0:.1f}%")
