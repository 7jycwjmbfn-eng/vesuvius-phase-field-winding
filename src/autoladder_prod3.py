"""Production mode of the automatic ladder generator: no truth, any region of box A (L2 z 10496-11008, y 1920-4992, x 3328-6400).
Writes the ladders in the VC point-collection format of the community winding ladders (vc_pointcollections_json_version 1, metadata.winding_is_absolute = false, points {p: [x, y, z] in L2 voxels, wind_a}).
Extra fields (ignored by readers that do not know them): per point `link_prob` (certifier probability of the link that reaches it, 1.0 for the first point of a ladder), `v6_conf`; per collection `metadata.source`.
Rules are those of autoladder2.py (SNAP on, GAP 3, tau from ladder out-of-fold precision TARGET, kept crossings have v6 confidence >= PCONF).  Chains with fewer than MINPTS points are not written.
  python autoladder_prod.py OUT.json N SEED [MINPTS]       env PCONF (0.8), TARGET (0.99), RAYLEN (300)"""
import os, sys, json, time
import numpy as np
from scipy import ndimage as ndi
import train_wfield_8c as tw
import certlib as C

out, N, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); MINPTS = int(sys.argv[4]) if len(sys.argv) > 4 else 3
PCONF = float(os.environ.get("PCONF", 0.8)); TARGET = float(os.environ.get("TARGET", 0.99)); RAYLEN = float(os.environ.get("RAYLEN", 300)); GAP = 3; rng = np.random.default_rng(seed); t0 = time.time()
import certnet as CN
net = CN.CertNet(stack_path=os.environ.get("STACK_PATH", "E:/vesuvius_counter/stack_model.pkl"), target=TARGET); TAU = net.tau
print(f"CertNet; tau {TAU:.4f} for overall precision {TARGET * 100:.1f}% on the ladders; PCONF {PCONF}", flush=True)
box = net.box; ORG_B = C.ORG_B; shp = np.array(box.psi6.shape); uz, uy, ux = tw.umbilicus()


coll = {}; nray = 0; tries = 0; nlad = 0; npts = 0; cid = 0
while nray < N and tries < 200 * N:
    tries += 1
    a0 = ORG_B + np.array([rng.uniform(4, 508), rng.uniform(100, 2900), rng.uniform(100, 2900)]); cy, cx = np.interp(a0[0], uz, uy), np.interp(a0[0], uz, ux); gy = a0[1] - cy; gx = a0[2] - cx; r = np.hypot(gy, gx)
    if r < 150: continue
    th = rng.normal(0, np.radians(25)); d = np.array([gy / r * np.cos(th) - gx / r * np.sin(th), gy / r * np.sin(th) + gx / r * np.cos(th)]); steps = np.arange(0, RAYLEN, 0.5); A = np.c_[a0[0] + rng.uniform(-0.35, 0.35) * steps, a0[1] + d[0] * steps, a0[2] + d[1] * steps]
    G = (A - ORG_B - 0.5) / 2.0; ins = np.all((G >= 2) & (G <= shp - 3), 1); n_in = int(ins.argmin()) if not ins.all() else len(G)
    if n_in < 160: continue
    A = A[:n_in]; G = G[:n_in]; lo = np.floor(G.min(0) - 3).astype(int).clip(0); hi = np.ceil(G.max(0) + 4).astype(int); sl = tuple(slice(l_, h_) for l_, h_ in zip(lo, hi))
    P = np.asarray(box.psi6[sl]).astype(np.float32); Cf = np.asarray(box.conf6[sl]).astype(np.float32); loc = (G - lo).T
    ang = np.unwrap(np.arctan2(ndi.map_coordinates(np.sin(P), loc, order=1, mode="nearest"), ndi.map_coordinates(np.cos(P), loc, order=1, mode="nearest"))); up = ang / (2 * np.pi); sgn = 1.0 if up[-1] >= up[0] else -1.0; up = up * sgn
    cf = ndi.map_coordinates(Cf, loc, order=1, mode="nearest"); upm = np.maximum.accumulate(up); mp = np.arange(np.ceil(up[0]), np.floor(upm[-1]) + 1)
    if len(mp) < MINPTS: continue
    nray += 1; pos = np.array([np.interp(m, upm, np.arange(len(upm))) for m in mp])
    la = np.rint(A - ORG_B).astype(int).clip(0, np.array(box.spv.shape) - 1); spl = ndi.gaussian_filter1d(np.asarray(box.spv[tuple(la.T)]).astype(np.float32), 2.0); newpos = []
    for q in pos:
        qi = int(round(q)); cands = [k for k in range(max(1, qi - 8), min(len(spl) - 1, qi + 9)) if spl[k] >= spl[k - 1] and spl[k] > spl[k + 1] and spl[k] > 60]
        newpos.append(float(min(cands, key=lambda k: abs(k - q))) if cands else float(q))
    pos = np.array(newpos); pos = pos[~np.r_[False, np.diff(pos) < 2.0]]; c_at = np.interp(pos, np.arange(len(cf)), cf); keep = c_at >= PCONF; pos = pos[keep]; c_at = c_at[keep]
    nk = len(pos)
    if nk < MINPTS: continue
    pts = np.c_[np.interp(pos, np.arange(len(A)), A[:, 0]), np.interp(pos, np.arange(len(A)), A[:, 1]), np.interp(pos, np.arange(len(A)), A[:, 2])]
    keys = [(i, j) for i in range(nk) for j in range(i + 1, min(nk, i + 1 + GAP))]; cnts, prs = net.certify([(pts[i], pts[j]) for i, j in keys]); table = {k_: (int(c_), float(p_)) for k_, c_, p_ in zip(keys, cnts, prs)}
    cur = [0]; winds = [0]; probs = [1.0]; i = 0
    def flush():
        global cid, nlad, npts
        if len(cur) >= MINPTS:
            cid += 1; nlad += 1; npts += len(cur)
            coll[str(cid)] = {"autoFillConstant": 0.0, "autoFillMode": 1, "color": [float(rng.random()), float(rng.random()), float(rng.random())], "metadata": {"winding_is_absolute": False, "source": "autoladder_prod3 (phase-field v6 + counter network + stacked certifier)", "ray": nray, "phase_sign_along_ray": sgn},
                              "name": f"auto{cid}", "points": {str(w_ + 1): {"creation_time": 0, "p": [float(pts[q][2]), float(pts[q][1]), float(pts[q][0])], "wind_a": float(w), "link_prob": float(pr), "v6_conf": float(c_at[q])} for w_, (q, w, pr) in enumerate(zip(cur, winds, probs))}}
    while True:
        i = cur[-1]; nxt = None
        for j in range(i + 1, min(nk, i + 1 + GAP)):
            cand, p = table[(i, j)]
            if 1 <= cand <= 4 and p >= TAU: nxt = (j, cand, p); break
        if nxt is None:
            flush(); j = i + 1
            if j >= nk: break
            cur = [j]; winds = [0]; probs = [1.0]
        else: cur.append(nxt[0]); winds.append(winds[-1] + nxt[1]); probs.append(nxt[2])
    if nray % 100 == 0: print(f"  {nray} rays ({tries} tries), {nlad} ladders, {npts} points, {time.time() - t0:.0f} s", flush=True)
json.dump({"collections": coll, "vc_pointcollections_json_version": "1"}, open(out, "w")); print(f"saved {out}: {nray} rays, {nlad} ladders, {npts} points, {time.time() - t0:.0f} s", flush=True)
