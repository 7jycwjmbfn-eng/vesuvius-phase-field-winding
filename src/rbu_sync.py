"""RBU step 2: align the certified radial ladders of neighbouring rays, solve the integer offsets of all ladder segments (L1 sync, bigsync2 irls_sync with integer coordinate descent), and score the result against the truth turn numbers.
Segments = chains of rbu_rays.py.  Two segments of neighbouring rays (same slice, adjacent azimuth; adjacent slices, nearest azimuth; or overlapping windows of one ray) are compared sheet by sheet:
sheet j of A and sheet l of B are the same sheet when their radii differ by at most TOLF * pitch (pitch = median radial distance of consecutive crossings of the two segments, 20 voxels if undefined); every such pair votes for the offset
p_B - p_A = w_A(j) - w_B(l); the mode is the edge (>= MINM matches and >= 60% of the pairs agree).  Truth is used only for scoring.
  python rbu_sync.py RAYS1.pkl [RAYS2.pkl ...]       env TOLF (0.35), MINM (2)"""
import os, sys, pickle, time
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

import isosurf as iso
import bigsync2 as bs

TOLF = float(os.environ.get("TOLF", 0.35)); MINM = int(os.environ.get("MINM", 2)); t0 = time.time()
recs = []; meta = None
for f in sys.argv[1:]:
    d = pickle.load(open(f, "rb")); recs += d["records"]; meta = d
segs = []                                                                                                              # one dict per chain
for ri, r in enumerate(recs):
    for ch in r["chains"]: segs.append(dict(ray=ri, z=r["z"], phi=r["phi"], ip=r["ip"], pts=ch["pts"], w=ch["winds"], r=ch["r"], conf=ch["conf"], prob=ch["probs"]))
n = len(segs); print(f"rays {len(recs)}, segments {n}, crossings {sum(len(s['w']) for s in segs)}", flush=True)
# ray adjacency: by physical position of the ray midpoints (y, x, z); neighbours within DLAT laterally and ZSTEP in z
ray_xy = {}
for ri, r in enumerate(recs):
    if not r["chains"]: continue
    P = np.concatenate([c["pts"] for c in r["chains"]]); ray_xy[ri] = P[:, 1:].mean(0)
by_ray = {}
for si, s in enumerate(segs): by_ray.setdefault(s["ray"], []).append(si)
rid = np.array(sorted(by_ray)); pos3 = np.array([[recs[i]["z"] * 1.0, *ray_xy[i]] for i in rid]); tree = cKDTree(pos3)
zstep = float(meta["zstep"]); pairs_ray = set()
for a, b in tree.query_pairs(r=max(1.8 * meta["arc"], zstep * 1.01) * 1.0, output_type="ndarray"):
    ia, ib = rid[a], rid[b]; dz = abs(recs[ia]["z"] - recs[ib]["z"]); lat = np.linalg.norm(ray_xy[ia] - ray_xy[ib])
    if dz == 0 and lat <= 1.8 * meta["arc"]: pairs_ray.add((min(ia, ib), max(ia, ib)))
    elif dz == zstep and lat <= 1.8 * meta["arc"]: pairs_ray.add((min(ia, ib), max(ia, ib)))
for ia in by_ray: pairs_ray.add((ia, ia))
print(f"ray pairs {len(pairs_ray)}  {time.time() - t0:.0f} s", flush=True)
rows, cols, dd, ww = [], [], [], []; matched = []                                                                          # matched: (seg a, seg b, w_a, w_b, j, l)
for (ia, ib) in pairs_ray:
    for sa in by_ray[ia]:
        for sb in by_ray[ib]:
            if sa == sb or (ia == ib and sa > sb): continue
            A, B = segs[sa], segs[sb]
            if A["r"].max() < B["r"].min() - 30 or B["r"].max() < A["r"].min() - 30: continue
            dr = np.r_[np.diff(A["r"]), np.diff(B["r"])]; dw = np.r_[np.diff(A["w"]), np.diff(B["w"])]; pitch = float(np.median(dr / np.maximum(dw, 1))) if len(dr) else 20.0
            pitch = 20.0 if not np.isfinite(pitch) or pitch < 8 or pitch > 40 else pitch; tol = TOLF * pitch
            D = np.abs(A["r"][:, None] - B["r"][None, :]); j, l = np.nonzero(D <= tol)
            if len(j) < MINM: continue
            # one candidate per sheet: keep the closest l for every j and the closest j for every l
            best = {}
            for jj, ll in zip(j, l):
                if jj not in best or D[jj, ll] < D[jj, best[jj]]: best[jj] = ll
            jj = np.array(list(best.keys())); ll = np.array([best[k] for k in jj]); vote = A["w"][jj] - B["w"][ll]
            vals, cnt = np.unique(vote, return_counts=True); m = int(cnt.max()); mode = int(vals[cnt.argmax()]); frac = m / len(jj)
            if m < MINM or frac < 0.6: continue
            rows.append(sa); cols.append(sb); dd.append(float(mode)); ww.append(float(frac ** 2 * min(1.0, m / 3.0)))
            for a_, b_ in zip(jj[vote == mode], ll[vote == mode]): matched.append((sa, sb, a_, b_))
print(f"edges {len(rows)} (matched sheet pairs {len(matched)})  {time.time() - t0:.0f} s", flush=True)
rows = np.array(rows); cols = np.array(cols); dd = np.array(dd); ww = np.array(ww)
p = bs.irls_sync(n, rows, cols, dd, ww); viol = np.abs(p[cols] - p[rows] - dd) > 0.5
print(f"sync: violated edges {viol.mean() * 100:.2f}% (weight share {ww[viol].sum() / ww.sum() * 100:.2f}%)  {time.time() - t0:.0f} s", flush=True)
nc, comp = connected_components(coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)), directed=False)
# crossings with global index and truth
Gidx = np.concatenate([s["w"] + p[i] for i, s in enumerate(segs)]); Pts = np.concatenate([s["pts"] for s in segs]).astype(np.float64); Sid = np.concatenate([np.full(len(s["w"]), i) for i, s in enumerate(segs)])
T = iso.turn_numbers(Pts); ok = np.isfinite(T); Cmp = comp[Sid]
print(f"truth-valid crossings {int(ok.sum())} of {len(T)}  {time.time() - t0:.0f} s", flush=True)
# orientation of the truth numbers along the radial direction: majority sign of consecutive chain steps
steps = []
for i, s in enumerate(segs):
    t = iso.turn_numbers(s["pts"].astype(np.float64)); v = np.isfinite(t[:-1]) & np.isfinite(t[1:]); steps += list(np.sign(t[1:][v] - t[:-1][v]))
sign_t = 1.0 if np.mean(steps) >= 0 else -1.0; Tn = T * sign_t; print(f"orientation of truth numbers along r: {sign_t:+.0f} (mean step sign {np.mean(steps):.2f})", flush=True)
U = np.load("E:/vesuvius_big_hot/u_v6.npy", mmap_mode="r"); gi = np.clip(np.rint((Pts - np.array([10496, 1920, 3328.]) - 0.5) / 2.0).astype(int), 0, np.array(U.shape) - 1); Lu = np.array([int(np.rint(float(U[tuple(g)]))) for g in gi])
u_sign = 1.0                                                                                                              # u_v6 increases outward in the same convention as truth numbers up to the gauge; sign checked below
d_rbu = Gidx - Tn; d_u = Lu - Tn
if np.mean(np.sign(np.diff(Lu)[Sid[:-1] == Sid[1:]])) < 0: d_u = Lu + Tn; print("u_v6 decreases outward: using Lu + Tn", flush=True)


def mode_share(d, mask):
    if mask.sum() == 0: return 0, 0
    v, c = np.unique(d[mask], return_counts=True); return int(c.max()), int(mask.sum())


good_r = good_u = tot = 0; sizes = []
for c in range(nc):
    m = (Cmp == c) & ok
    if m.sum() == 0: continue
    a, b = mode_share(d_rbu, m); au, _ = mode_share(d_u, m); good_r += a; good_u += au; tot += b; sizes.append((int((Cmp == c).sum()), a / b, au / b, int(b)))
sizes.sort(reverse=True)
print(f"components {nc}; largest components (crossings, rbu acc, u_v6 acc, valid): {[(s[0], round(s[1], 3), round(s[2], 3), s[3]) for s in sizes[:8]]}")
big = [s for s in sizes if s[3] >= 50]
print(f"label accuracy over valid crossings, component-wise gauge: RBU {good_r / tot * 100:.2f}%, u_v6 {good_u / tot * 100:.2f}%  (crossings {tot})")
# global gauge: all components together with their own mode (above) and the largest component alone
if sizes:
    c0 = int(np.argmax(np.bincount(Cmp))); m0 = (Cmp == c0) & ok; a0, b0 = mode_share(d_rbu, m0); au0, _ = mode_share(d_u, m0)
    print(f"largest component: {int((Cmp == c0).sum())} crossings ({(Cmp == c0).mean() * 100:.1f}% of all), valid {b0}: RBU {a0 / b0 * 100:.2f}%, u_v6 {au0 / b0 * 100:.2f}%")
# alignment correctness: matched sheet pairs with equal truth index
eq = []
off = np.r_[0, np.cumsum([len(s["w"]) for s in segs])]
for sa, sb, a_, b_ in matched:
    ta, tb = Tn[off[sa] + a_], Tn[off[sb] + b_]
    if np.isfinite(ta) and np.isfinite(tb): eq.append(ta == tb)
print(f"matched sheet pairs with both truth values: {len(eq)}, same truth sheet: {np.mean(eq) * 100:.2f}%")
pickle.dump(dict(Gidx=Gidx, Pts=Pts, Sid=Sid, T=Tn, Lu=Lu, comp=comp, p=p, segs=[dict(ray=s["ray"], z=s["z"], phi=s["phi"]) for s in segs]), open("E:/vesuvius_rbu_sync_out.pkl", "wb"))
