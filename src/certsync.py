"""Certified winding counts as extra constraints of the tile synchronisation (bigsync2.py).
The tile offsets p solve  min sum w |p_b - p_a - dd|  over the overlap pairs (saved in U_PAIRS.npz).  Every certified link (point a -> point b of an automatic ladder, count c, phase sign s along the ray) says that the
unwrapped field must change by s * c between the two points.  With u_old = tile field + p_old this gives the extra edge  p_tb - p_ta = (p_old_tb - p_old_ta) + s*c - (round(u_old(b)) - round(u_old(a)))  with weight W.
The new offsets come from the same IRLS + integer coordinate descent as the original sync; U_new = U_old + (p_new - p_old)[tile of the voxel] (tile cores as in the original pass 2), no unwrapping is repeated.
  python certsync.py U_OLD.npy U_PAIRS.npz U_NEW.npy W LADDERS1.json [LADDERS2.json ...]"""
import sys, json, time
import numpy as np
from bigsync2 import irls_sync, starts

import os
T, S = 48, 24; ORG = np.array([float(v) for v in os.environ.get("BOX_ORG", "10496,1920,3328").split(",")]); SHAPE = (256, 1536, 1536)                # BOX_ORG: origin of the box (L2 z, y, x)
uold_path, pairs_path, unew_path, W = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]); ladders = sys.argv[5:]; t0 = time.time()
st = [starts(n, T, S) for n in SHAPE]; nt = [len(s) for s in st]; cen = [np.array(s) + T / 2.0 for s in st]; near = [np.abs(np.arange(N)[:, None] - cen[a][None]).argmin(1) for a, N in enumerate(SHAPE)]
d = np.load(pairs_path); rows, cols, dd, ww, n = d["rows"], d["cols"], d["dd"], d["ww"], int(d["n"]); assert n == nt[0] * nt[1] * nt[2]
p_old = irls_sync(n, rows, cols, dd, ww); viol = np.abs(p_old[cols] - p_old[rows] - dd) > 0.5; obj = float((ww * np.abs(p_old[cols] - p_old[rows] - dd)).sum())
print(f"recomputed old offsets: violated {viol.mean() * 100:.2f}% (weight share {ww[viol].sum() / ww.sum() * 100:.2f}%), L1 objective {obj:.1f}  (log of the original run: 31.45%, 19.02%, 54056.3 for the v6 box A sync)  {time.time() - t0:.0f} s", flush=True)
U = np.load(uold_path, mmap_mode="r")
lr, lc, ld = [], [], []; n_links = 0; n_same = 0; n_same_bad = 0; n_agree = 0
for f in ladders:
    for cid, c in json.load(open(f))["collections"].items():
        sg = c["metadata"].get("phase_sign_along_ray")
        if sg is None: continue
        pts = sorted(c["points"].values(), key=lambda q: q["wind_a"])
        G = np.array([[q["p"][2], q["p"][1], q["p"][0]] for q in pts]); G = (G - ORG - 0.5) / 2.0; gi = np.rint(G).astype(int); ok = np.all((gi >= 0) & (gi < np.array(SHAPE)), 1)
        w = np.array([q["wind_a"] for q in pts]); uv = np.array([float(U[tuple(g)]) if o else np.nan for g, o in zip(gi, ok)])
        tile = [(near[0][g[0]] * nt[1] + near[1][g[1]]) * nt[2] + near[2][g[2]] if o else -1 for g, o in zip(gi, ok)]
        for q in range(len(pts) - 1):
            if not (ok[q] and ok[q + 1]): continue
            n_links += 1; cnt = w[q + 1] - w[q]; e = int(round(sg * cnt - (round(uv[q + 1]) - round(uv[q]))))
            if e == 0: n_agree += 1
            ta, tb = tile[q], tile[q + 1]
            if ta == tb: n_same += 1; n_same_bad += (e != 0); continue
            lr.append(ta); lc.append(tb); ld.append(float(p_old[tb] - p_old[ta] + e))
print(f"links {n_links}: label already agrees with the certified count for {n_agree} ({n_agree / max(n_links, 1) * 100:.1f}%); links inside one tile {n_same} (of which disagree {n_same_bad}); usable edges between two tiles {len(lr)}; files {len(ladders)}", flush=True)
lr = np.array(lr, int); lc = np.array(lc, int); ld = np.array(ld, float)
rows2 = np.r_[rows, lr]; cols2 = np.r_[cols, lc]; dd2 = np.r_[dd, ld]; ww2 = np.r_[ww, np.full(len(lr), W)]
p_new = irls_sync(n, rows2, cols2, dd2, ww2); delta = p_new - p_old
violn = np.abs(p_new[cols] - p_new[rows] - dd) > 0.5; objn = float((ww * np.abs(p_new[cols] - p_new[rows] - dd)).sum())
vl_old = np.abs(p_old[lc] - p_old[lr] - ld) > 0.5; vl_new = np.abs(p_new[lc] - p_new[lr] - ld) > 0.5
print(f"new offsets: overlap-pair violations {violn.mean() * 100:.2f}% (weight share {ww[violn].sum() / ww.sum() * 100:.2f}%), objective {objn:.1f}; link edges violated old {vl_old.mean() * 100:.2f}% -> new {vl_new.mean() * 100:.2f}%; tiles changed {int((delta != 0).sum())} of {n}  {time.time() - t0:.0f} s", flush=True)
np.savez_compressed(unew_path.replace(".npy", "_pairs.npz"), rows=rows2, cols=cols2, dd=dd2, ww=ww2, n=n)                          # edges of the new sync (the join step recomputes p_new from them)
import os
if not os.environ.get("REUSE_U"):
    D3 = delta.reshape(nt); Un = np.lib.format.open_memmap(unew_path, mode="w+", dtype=np.float32, shape=SHAPE)
    for z in range(SHAPE[0]):
        sl = D3[near[0][z]][near[1]][:, near[2]]; Un[z] = np.asarray(U[z]) + sl.astype(np.float32)
    Un.flush()
np.savez_compressed(unew_path.replace(".npy", "_offsets.npz"), p_old=p_old, p_new=p_new); print(f"saved {unew_path} {time.time() - t0:.0f} s", flush=True)
