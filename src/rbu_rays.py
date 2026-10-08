"""RBU step 1: certified ladders on a regular (z, azimuth) grid of radial rays inside the truth rectangle of the W2 slab (box A arrays).
For every slice z (every ZSTEP L2 layers in [ZA, ZB)) and every azimuth (arc spacing ARC voxels) a radial ray from the umbilicus is cut to its longest run inside the truth rectangle (y 2304-3840, x 3584-5120) and into windows of WIN voxels
(overlap OVL).  In each window the generator of autoladder2.py runs unchanged: integer crossings of the unwrapped v6 phase, snapped to the surface-prediction ridge, kept when the v6 confidence is >= PCONF, linked i -> i+1..i+3 when the learned
certifier (human ladders only, regions touching the truth box and the training z range removed, tau at ladder-side overall precision 99%) accepts a count of 1..4; chains of >= 2 points are written.
  python rbu_rays.py OUT.pkl ZA ZB PROC NPROC [ZSTEP]      (PROC / NPROC: every NPROC-th azimuth, so several processes can share the work; env PCONF=0.8)"""
import os, sys, pickle, time
import numpy as np
from scipy import ndimage as ndi
import train_wfield_8c as tw
import certlib as C

out, ZA, ZB, PROC, NPROC = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]); ZSTEP = int(sys.argv[6]) if len(sys.argv) > 6 else 12
PCONF = float(os.environ.get("PCONF", 0.8)); TARGET = 0.99; GAP = 3; ARC = 8.0; WIN = 400.0; OVL = 60.0; t0 = time.time()
f2 = ["E:/vesuvius_ladder2_rows_0.npz", "E:/vesuvius_ladder2_rows_1.npz"]; f3 = ["E:/vesuvius_ladder3_rows_0.npz", "E:/vesuvius_ladder3_rows_1.npz"]
lad = C.load_ladder(f2, f3); s1, s2 = C.gm_slopes(lad); Xl, cl, _ = C.features(lad, s1, s2); yl = (cl == lad["truth"]).astype(int); typl = (lad["truth"] >= 1) & (lad["truth"] <= 4)
pl = np.zeros(len(yl))
for f in range(5):
    tr = (lad["fold"] != f) & typl; te = lad["fold"] == f; pl[te] = C.fit(Xl[tr], yl[tr]).predict_proba(Xl[te])[:, 1]
order = np.argsort(-pl[typl]); cum = np.cumsum(yl[typl][order]) / np.arange(1, typl.sum() + 1); ps = pl[typl][order]; TAU = float(ps[np.nonzero(cum >= TARGET)[0].max()]); model = C.fit(Xl[typl], yl[typl])
print(f"certifier on {int(typl.sum())} ladder pairs, tau {TAU:.4f}; PCONF {PCONF}; {time.time() - t0:.0f} s", flush=True)
box = C.BoxA(); ORG_B = C.ORG_B; uz, uy, ux = tw.umbilicus(); Y0, Y1, X0, X1 = 2304 + 6, 3840 - 6, 3584 + 6, 5120 - 6


def pair(a, b):
    v = box.votes(a, b); T = np.array([[1.0, *v, 0.0]]); X, cand, _ = C.features(C.rows_to_dict(T), s1, s2); return int(cand[0]), float(model.predict_proba(X)[0, 1])


def ladders_on_window(z, pts_yx, r0, cen):
    """pts_yx: (n,2) L2 coordinates of the window samples (step 0.5); returns chains"""
    A = np.c_[np.full(len(pts_yx), float(z)), pts_yx]; G = (A - ORG_B - 0.5) / 2.0; lo = np.floor(G.min(0) - 3).astype(int).clip(0); hi = np.ceil(G.max(0) + 4).astype(int); sl = tuple(slice(l_, h_) for l_, h_ in zip(lo, hi))
    P = np.asarray(box.psi6[sl]).astype(np.float32); Cf = np.asarray(box.conf6[sl]).astype(np.float32); loc = (G - lo).T
    ang = np.unwrap(np.arctan2(ndi.map_coordinates(np.sin(P), loc, order=1, mode="nearest"), ndi.map_coordinates(np.cos(P), loc, order=1, mode="nearest"))); up = ang / (2 * np.pi); sgn = 1.0 if up[-1] >= up[0] else -1.0; up = up * sgn
    cf = ndi.map_coordinates(Cf, loc, order=1, mode="nearest"); upm = np.maximum.accumulate(up); mp = np.arange(np.ceil(up[0]), np.floor(upm[-1]) + 1)
    if len(mp) < 2: return []
    pos = np.array([np.interp(m, upm, np.arange(len(upm))) for m in mp])
    la = np.rint(A - ORG_B).astype(int).clip(0, np.array(box.spv.shape) - 1); spl = ndi.gaussian_filter1d(np.asarray(box.spv[tuple(la.T)]).astype(np.float32), 2.0); newpos = []
    for q in pos:
        qi = int(round(q)); cands = [k for k in range(max(1, qi - 8), min(len(spl) - 1, qi + 9)) if spl[k] >= spl[k - 1] and spl[k] > spl[k + 1] and spl[k] > 60]
        newpos.append(float(min(cands, key=lambda k: abs(k - q))) if cands else float(q))
    pos = np.array(newpos); pos = pos[~np.r_[False, np.diff(pos) < 2.0]]; c_at = np.interp(pos, np.arange(len(cf)), cf); keep = c_at >= PCONF; pos = pos[keep]; c_at = c_at[keep]; nk = len(pos)
    if nk < 2: return []
    pts = np.c_[np.interp(pos, np.arange(len(A)), A[:, 0]), np.interp(pos, np.arange(len(A)), A[:, 1]), np.interp(pos, np.arange(len(A)), A[:, 2])]
    table = {(i, j): pair(pts[i], pts[j]) for i in range(nk) for j in range(i + 1, min(nk, i + 1 + GAP))}
    chains = []; cur = [0]; winds = [0]; probs = [1.0]
    while True:
        i = cur[-1]; nxt = None
        for j in range(i + 1, min(nk, i + 1 + GAP)):
            cand, p = table[(i, j)]
            if 1 <= cand <= 4 and p >= TAU: nxt = (j, cand, p); break
        if nxt is None:
            if len(cur) >= 2: chains.append(dict(idx=list(cur), winds=list(winds), probs=list(probs)))
            j = i + 1
            if j >= nk: break
            cur = [j]; winds = [0]; probs = [1.0]
        else: cur.append(nxt[0]); winds.append(winds[-1] + nxt[1]); probs.append(nxt[2])
    res = []
    for ch in chains:
        ii = np.array(ch["idx"]); P3 = pts[ii]; r = np.hypot(P3[:, 1] - cen[0], P3[:, 2] - cen[1])
        res.append(dict(pts=P3.astype(np.float32), winds=np.array(ch["winds"], int), r=r.astype(np.float32), conf=c_at[ii].astype(np.float32), probs=np.array(ch["probs"], np.float32)))
    return res


records = []; nray = 0
zs = list(range(ZA + ZSTEP // 2, ZB, ZSTEP))
for z in zs:
    cy, cx = np.interp(z, uz, uy), np.interp(z, uz, ux)
    corners = np.array([[Y0, X0], [Y0, X1], [Y1, X0], [Y1, X1]]); ph = np.arctan2(corners[:, 0] - cy, corners[:, 1] - cx); ph0 = np.angle(np.exp(1j * ph).mean()); dph = np.angle(np.exp(1j * (ph - ph0)))
    rr = np.hypot(corners[:, 0] - cy, corners[:, 1] - cx); r_mid = rr.mean(); step_ph = ARC / r_mid; phis = ph0 + np.arange(dph.min() - 0.02, dph.max() + 0.02, step_ph)
    for ip, phi in enumerate(phis):
        if ip % NPROC != PROC: continue
        rg = np.arange(0.0, 2000.0, 0.5); yy = cy + rg * np.sin(phi); xx = cx + rg * np.cos(phi); ins = (yy >= Y0) & (yy <= Y1) & (xx >= X0) & (xx <= X1)
        if not ins.any(): continue
        d = np.diff(np.r_[0, ins.astype(int), 0]); st = np.nonzero(d == 1)[0]; en = np.nonzero(d == -1)[0]; k = int(np.argmax(en - st)); a0, a1 = st[k], en[k]
        if (a1 - a0) < 160: continue
        nray += 1; r_a, r_b = rg[a0], rg[a1 - 1]; wins = []
        s = r_a
        while True:
            e = min(s + WIN, r_b); wins.append((s, e))
            if e >= r_b: break
            s = e - OVL
        rec = dict(z=float(z), phi=float(phi), ip=ip, r_range=(float(r_a), float(r_b)), chains=[])
        for (s, e) in wins:
            m = (rg >= s) & (rg <= e) & ins
            if m.sum() < 160: continue
            rec["chains"] += ladders_on_window(z, np.c_[yy[m], xx[m]], s, (cy, cx))
        records.append(rec)
        if nray % 50 == 0: print(f"  z {z}: {nray} rays, {sum(len(r['chains']) for r in records)} chains, {time.time() - t0:.0f} s", flush=True)
pickle.dump(dict(records=records, tau=TAU, pconf=PCONF, zstep=ZSTEP, arc=ARC, za=ZA, zb=ZB), open(out, "wb")); print(f"saved {out}: {nray} rays, {sum(len(r['chains']) for r in records)} chains, {sum(len(c['winds']) for r in records for c in r['chains'])} points, {time.time() - t0:.0f} s", flush=True)
