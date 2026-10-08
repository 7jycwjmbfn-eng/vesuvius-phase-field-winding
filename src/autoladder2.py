"""Automatic ladder generator, version 2, and its evaluation against the dense truth (W2 held-out slab).
Rules fixed BEFORE the final reading (written in PREREG_bigbox_1006.md):
 - the generator sees only predictions (SNAP=1: each crossing is moved to the nearest local maximum of the surface prediction within 4 voxels).  Along a ray it takes the integer crossings of the unwrapped v6 phase as candidate sheet centres and keeps those whose v6 confidence is >= PCONF.
 - from the current chain end i it tries the next kept points j = i+1, i+2, i+3 (nearest first); a link i->j is accepted when the learned certifier (trained on human ladders only, ladder regions touching the truth box removed)
   gives probability >= tau and a candidate count c in 1..4; wind(j) = wind(i) + c.  If none of the three links is accepted the chain ends and a new chain starts at i+1.
 - tau is read from the ladder out-of-fold predictions at overall precision TARGET (default 0.99).
 - scoring (truth never enters the generator): a link is correct when both end points lie within TOL turns of a truth sheet centre and the truth sheet count between them equals the certified count c.  TOL 0.15 turn (about 3 voxels) is the
   primary criterion; 0.30 is reported as well.
 - rays start in z index range ZRANGE of the slab (development: 0-250, final reading: 250-500); PCONF 0.7 / 0.8 / 0.9 are all reported (0.8 is the default); no value is selected after the final reading.
  python autoladder2.py OUT.pkl N SEED ZLO ZHI        env PCONF (default 0.8), TARGET (0.99), RAYLEN (300)"""
import os, sys, pickle, time
import numpy as np
from scipy import ndimage as ndi
import train_wfield_8c as tw
import certlib as C

out, N, seed, zi0, zi1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
PCONF = float(os.environ.get("PCONF", 0.8)); TARGET = float(os.environ.get("TARGET", 0.99)); RAYLEN = float(os.environ.get("RAYLEN", 300)); GAP = int(os.environ.get("GAP", 3)); SNAP = int(os.environ.get("SNAP", 1)); rng = np.random.default_rng(seed); t0 = time.time()
f2 = ["E:/vesuvius_ladder2_rows_0.npz", "E:/vesuvius_ladder2_rows_1.npz"]; f3 = ["E:/vesuvius_ladder3_rows_0.npz", "E:/vesuvius_ladder3_rows_1.npz"]
lad = C.load_ladder(f2, f3); s1, s2 = C.gm_slopes(lad); Xl, cl, _ = C.features(lad, s1, s2); yl = (cl == lad["truth"]).astype(int); typl = (lad["truth"] >= 1) & (lad["truth"] <= 4)
pl = np.zeros(len(yl))
for f in range(5):
    tr = (lad["fold"] != f) & typl; te = lad["fold"] == f; pl[te] = C.fit(Xl[tr], yl[tr]).predict_proba(Xl[te])[:, 1]
order = np.argsort(-pl[typl]); cum = np.cumsum(yl[typl][order]) / np.arange(1, typl.sum() + 1); ps = pl[typl][order]
taus = {t: float(ps[np.nonzero(cum >= t)[0].max()]) for t in (0.98, 0.99, 0.995)}; model = C.fit(Xl[typl], yl[typl])
print(f"certifier trained on {int(typl.sum())} ladder pairs; thresholds {taus}; PCONF {PCONF}; z index {zi0}-{zi1}; {time.time() - t0:.0f} s", flush=True)
TAU = taus[0.99]
box = C.BoxA(); ORG_T = np.array(tw.ORG, float); lab = np.load(tw.LAB, mmap_mode="r"); zlo = 10500; L = np.asarray(lab[zlo - 9984:11000 - 9984]); uz, uy, ux = tw.umbilicus(); ORG_B = C.ORG_B


def pair(a, b):
    v = box.votes(a, b); T = np.array([[1.0, *v, 0.0]]); X, cand, _ = C.features(C.rows_to_dict(T), s1, s2); return int(cand[0]), float(model.predict_proba(X)[0, 1])


stats = []; nray = 0; tries = 0
while nray < N and tries < 800 * N:
    tries += 1
    z = int(rng.integers(zi0, zi1)); y = int(rng.integers(42, L.shape[1] - 42)); x = int(rng.integers(42, L.shape[2] - 42))
    if L[z, y, x] == 255: continue
    cy, cx = np.interp(zlo + z, uz, uy), np.interp(zlo + z, uz, ux); gy = ORG_T[1] + y - cy; gx = ORG_T[2] + x - cx; r = np.hypot(gy, gx)
    if r < 300: continue
    th = rng.normal(0, np.radians(25)); d = np.array([gy / r * np.cos(th) - gx / r * np.sin(th), gy / r * np.sin(th) + gx / r * np.cos(th)]); steps = np.arange(0, RAYLEN, 0.5); ys = y + d[0] * steps; xs = x + d[1] * steps; zs = z + rng.uniform(-0.35, 0.35) * steps
    inside = (ys >= 0) & (ys <= L.shape[1] - 1) & (xs >= 0) & (xs <= L.shape[2] - 1) & (zs >= 0) & (zs <= L.shape[0] - 1); n_in = int(inside.argmin()) if not inside.all() else len(steps)
    ys = ys[:n_in]; xs = xs[:n_in]; zs = zs[:n_in]; lv = L[np.rint(zs).astype(int), np.rint(ys).astype(int), np.rint(xs).astype(int)]
    bad = np.nonzero(lv == 255)[0]; n_ok = int(bad[0]) if len(bad) else len(lv)
    if n_ok < 160: continue
    lv = lv[:n_ok]; ys = ys[:n_ok]; xs = xs[:n_ok]; zs = zs[:n_ok]
    ut = np.unwrap(lv.astype(np.float64) * 2 * np.pi / 255) / (2 * np.pi); sg = 1.0 if ut[-1] >= ut[0] else -1.0; ut = ut * sg
    if np.any(np.diff(ut) < -0.12): continue
    ms = np.arange(np.ceil(ut[0]), np.floor(ut[-1]) + 1)
    if len(ms) < 3: continue
    A = np.c_[zlo + zs, ys + ORG_T[1], xs + ORG_T[2]]; G = (A - ORG_B - 0.5) / 2.0; lo = np.floor(G.min(0) - 3).astype(int).clip(0); hi = np.ceil(G.max(0) + 4).astype(int)
    sl = tuple(slice(l_, h_) for l_, h_ in zip(lo, hi)); P = np.asarray(box.psi6[sl]).astype(np.float32); Cf = np.asarray(box.conf6[sl]).astype(np.float32); loc = (G - lo).T
    ang = np.unwrap(np.arctan2(ndi.map_coordinates(np.sin(P), loc, order=1, mode="nearest"), ndi.map_coordinates(np.cos(P), loc, order=1, mode="nearest"))); up = ang / (2 * np.pi); up = up * (1.0 if up[-1] >= up[0] else -1.0)
    cf = ndi.map_coordinates(Cf, loc, order=1, mode="nearest"); upm = np.maximum.accumulate(up); mp = np.arange(np.ceil(up[0]), np.floor(upm[-1]) + 1)
    if len(mp) < 2: continue
    nray += 1; pos = np.array([np.interp(m, upm, np.arange(len(upm))) for m in mp])
    if SNAP:                                                                                                                  # move each crossing to the nearest local maximum of the surface prediction within +-4 voxels (8 samples)
        la = np.rint(A - ORG_B).astype(int).clip(0, np.array(box.spv.shape) - 1); spl = ndi.gaussian_filter1d(np.asarray(box.spv[tuple(la.T)]).astype(np.float32), 2.0); newpos = []
        for q in pos:
            qi = int(round(q)); cands = [k for k in range(max(1, qi - 8), min(len(spl) - 1, qi + 9)) if spl[k] >= spl[k - 1] and spl[k] > spl[k + 1] and spl[k] > 60]
            newpos.append(float(min(cands, key=lambda k: abs(k - q))) if cands else float(q))
        pos = np.array(newpos); dup = np.r_[False, np.diff(pos) < 2.0]; pos = pos[~dup]
    c_at = np.interp(pos, np.arange(len(cf)), cf); keep = c_at >= PCONF; pos = pos[keep]; c_at = c_at[keep]
    pz = np.interp(pos, np.arange(len(zs)), zs); py = np.interp(pos, np.arange(len(ys)), ys); px = np.interp(pos, np.arange(len(xs)), xs); pts = np.c_[zlo + pz, py + ORG_T[1], px + ORG_T[2]]
    ut_at = np.interp(pos, np.arange(len(ut)), ut); near = np.rint(ut_at).astype(int); off = np.abs(ut_at - near)
    nk = len(pts); table = {}
    for i in range(nk):
        for j in range(i + 1, min(nk, i + 1 + GAP)): table[(i, j)] = pair(pts[i], pts[j])
    stats.append(dict(ray=nray, zstart=z, n_truth=len(ms), n_all=len(mp), snap=SNAP, n_kept=nk, conf=c_at, off=off, near=near, pts=pts, table=table, truth_first=int(ms[0]), truth_last=int(ms[-1])))
    if nray % 25 == 0: print(f"  {nray} rays ({tries} tries), {time.time() - t0:.0f} s", flush=True)
pickle.dump(dict(stats=stats, taus=taus, pconf=PCONF, gap=GAP, zrange=(zi0, zi1)), open(out, "wb")); print("saved", out, len(stats), "rays", f"{time.time() - t0:.0f} s", flush=True)
