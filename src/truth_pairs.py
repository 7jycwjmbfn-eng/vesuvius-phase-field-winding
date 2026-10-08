"""Large-sample check of the winding-count certificate on pairs built from the dense truth of the official segments (W2, held-out z 10500-11000), independent of the human ladders.
Endpoints: random points on truth sheet centres (truth phase label near 0), the second endpoint is the k-th truth sheet centre along the radial line from the umbilicus (k = 1..4), so the truth count is k.
The same votes as the ladder rows are computed from the box A arrays (D = 2 grid): v6 phase along the straight line and two parallel lines, frozen_8c, two minimum-cost-path counts, official lasagna grad_mag
(straight and path), cos peaks, surface-prediction crossings (straight and path), CT profile peaks, confidences.
  python truth_pairs.py OUT.npz N [SEED]
env BOX=P139: the PHerc0139 test box instead (truth psi_box.npy over the whole box, centre line from centre_w023.json, arrays from E:/vesuvius_p0139_hot, cos and grad_mag all zero);
env P139_STUB=1 with BOX=P139: v6 phase := frozen_8c phase and v6 confidence := 1, path test only.
Columns: k, v6, off1, off2, frozen, path_a, path_b, gm, gm_path, cos_peaks, sp_straight, sp_path, ct_peaks, minconf, meanconf, path_minconf, L, z"""
import os, sys, time, json
import numpy as np
from scipy import ndimage as ndi
from skimage.graph import route_through_array

out = sys.argv[1]; N = int(sys.argv[2]); seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0; rng = np.random.default_rng(seed); t0 = time.time()
if os.environ.get("BOX") == "P139":                                                                                       # PHerc0139 test box: truth grid = box grid (L2), arrays on the D = 2 grid of the same box
    H = "E:/vesuvius_p0139_hot/"; ORG_T = ORG_B = np.array([4352, 1664, 1280.]); zlo, zhi = 4352, 5376; lab = np.load("G:/vesuvius_77f5/p0139/psi_box.npy", mmap_mode="r")
    cj = json.load(open("G:/vesuvius_77f5/p0139/centre_w023.json")); oc = np.argsort(cj["z"]); uz, uy, ux = (np.array(cj[t], float)[oc] for t in ("z", "y", "x"))        # centre line (L2 voxels), outside the box
    psif = np.load(H + "pred/frozen8c_d2.npy", mmap_mode="r"); const = lambda v, shape, dt: np.broadcast_to(np.asarray(v, dt), shape)                  # read-only constant views, no memory
    if os.environ.get("P139_STUB") == "1": psi6 = psif; conf6 = const(1, psif.shape, np.float16)                           # path test only: v6 := frozen_8c phase, confidence 1
    else: psi6 = np.load(H + "pred/v6nolas_psi.npy", mmap_mode="r"); conf6 = np.load(H + "pred/v6nolas_conf.npy", mmap_mode="r")
    cosv = const(0, (512, 768, 768), np.uint8); gmv = const(0, (256, 384, 384), np.float32)                                # no registered lasagna: cos and grad_mag votes are empty
    spv = np.load(H + "sp_big.npy", mmap_mode="r"); ctv = np.load(H + "ct_big.npy", mmap_mode="r")
else:
    import train_wfield_8c as tw                                                                                              # imported only here (pulls in torch)
    H = "E:/vesuvius_big_hot/"; ORG_T = np.array(tw.ORG, float); ORG_B = np.array([10496, 1920, 3328.])                          # truth box and box A origins (L2 z, y, x)
    lab = np.load(tw.LAB, mmap_mode="r"); zlo, zhi = 10500, 11000; uz, uy, ux = tw.umbilicus()
    psi6 = np.load(H + "pred_v6_psi.npy", mmap_mode="r"); conf6 = np.load(H + "pred_v6_conf.npy", mmap_mode="r"); psif = np.load("D:/vesuvius_big_hot/pred/frozen8c_psi.npy", mmap_mode="r")
    cosv = np.load(H + "las/cos.npy", mmap_mode="r"); gmv = np.asarray(np.load(H + "las/gm.npy", mmap_mode="r")).astype(np.float32) / 1000.0
    spv = np.load("D:/vesuvius_big_hot/sp_big.npy", mmap_mode="r"); ctv = np.load("D:/vesuvius_big_hot/ct_big.npy", mmap_mode="r")
L = np.asarray(lab[zlo - int(ORG_T[0]):zhi - int(ORG_T[0])])                                                               # memmap view (no copy): only the voxels along the rays are read
print(f"truth slab {L.shape}, {time.time() - t0:.0f} s", flush=True)


def peaks(prof, win=25):
    s = ndi.gaussian_filter1d(prof.astype(np.float64), 1.5); d = s - ndi.uniform_filter1d(s, win); sd = d.std() + 1e-6
    return len([q for q in range(1, len(d) - 1) if d[q] >= d[q - 1] and d[q] > d[q + 1] and d[q] > 0.5 * sd])


rows = []; tries = 0; RAYLEN = float(os.environ.get("RAYLEN", 140)); KMAX = int(os.environ.get("KMAX", 4))                  # env RAYLEN (voxels) and KMAX: longer rays and more wraps
while len(rows) < N and tries < 40 * N:
    tries += 1
    z = int(rng.integers(0, L.shape[0])); y = int(rng.integers(2 + 40, L.shape[1] - 2 - 40)); x = int(rng.integers(2 + 40, L.shape[2] - 2 - 40))
    if L[z, y, x] == 255: continue
    cy, cx = np.interp(zlo + z, uz, uy), np.interp(zlo + z, uz, ux); gy = ORG_T[1] + y - cy; gx = ORG_T[2] + x - cx; r = np.hypot(gy, gx)
    if r < 300: continue
    th = rng.normal(0, np.radians(25)); d = np.array([gy / r * np.cos(th) - gx / r * np.sin(th), gy / r * np.sin(th) + gx / r * np.cos(th)]); steps = np.arange(0, RAYLEN, 0.5); ys = y + d[0] * steps; xs = x + d[1] * steps   # radial direction, rotated by a random angle
    dz = rng.uniform(-0.35, 0.35); zs = z + dz * steps
    inside = (ys >= 0) & (ys <= L.shape[1] - 1) & (xs >= 0) & (xs <= L.shape[2] - 1) & (zs >= 0) & (zs <= L.shape[0] - 1); n_in = int(inside.argmin()) if not inside.all() else len(steps)
    if n_in < 100: continue
    ys = ys[:n_in]; xs = xs[:n_in]; zs = zs[:n_in]; steps = steps[:n_in]; lv = L[np.rint(zs).astype(int), np.rint(ys).astype(int), np.rint(xs).astype(int)]
    bad = np.nonzero(lv == 255)[0]; n_ok = int(bad[0]) if len(bad) else len(lv)
    if n_ok < 60: continue
    lv = lv[:n_ok]; ys = ys[:n_ok]; xs = xs[:n_ok]; zs = zs[:n_ok]; steps = steps[:n_ok]
    u = np.unwrap(lv.astype(np.float64) * 2 * np.pi / 255) / (2 * np.pi)                                                      # absolute phase in turns, sheet centres at integers
    sgn = 1.0 if u[-1] >= u[0] else -1.0; u = u * sgn
    if np.any(np.diff(u) < -0.12): continue                                                                                   # the truth phase must run monotonically along the ray
    ms = np.arange(np.ceil(u[0]), np.floor(u[-1]) + 1)                                                                        # integer crossings = truth sheet centres
    if len(ms) < 2: continue
    pos = np.array([np.interp(m, np.maximum.accumulate(u), np.arange(len(u))) for m in ms])                                  # fractional index of each crossing
    i0 = int(rng.integers(0, len(ms) - 1)); k = int(rng.integers(1, min(KMAX, len(ms) - 1 - i0) + 1)); i1 = i0 + k
    f = lambda q: (np.interp(q, np.arange(len(ys)), ys), np.interp(q, np.arange(len(xs)), xs))
    ya, xa = f(pos[i0]); yb, xb = f(pos[i1]); za = np.interp(pos[i0], np.arange(len(zs)), zs); zb = np.interp(pos[i1], np.arange(len(zs)), zs)
    a = np.array([zlo + za, ya + ORG_T[1], xa + ORG_T[2]], float); b = np.array([zlo + zb, yb + ORG_T[1], xb + ORG_T[2]], float)
    ga = (a - ORG_B - 0.5) / 2.0; gb = (b - ORG_B - 0.5) / 2.0; Lg = float(np.linalg.norm(gb - ga)); n = int(max(Lg * 2, 8)); line = ga + (gb - ga) * np.linspace(0, 1, n + 1)[:, None]
    lo = np.floor(np.minimum(ga, gb) - 8).astype(int).clip(0); hi = np.ceil(np.maximum(ga, gb) + 9).astype(int)
    sl = tuple(slice(l_, h_) for l_, h_ in zip(lo, hi)); P6 = np.asarray(psi6[sl]).astype(np.float32); C6 = np.asarray(conf6[sl]).astype(np.float32); PF = np.asarray(psif[sl]).astype(np.float32)
    loc = line - lo; samp = lambda vol, c: ndi.map_coordinates(vol, c.T, order=1, mode="nearest")
    def count(P, c):
        a_ = np.unwrap(np.arctan2(samp(np.sin(P), c), samp(np.cos(P), c))); return (a_[-1] - a_[0]) / (2 * np.pi)
    v6 = count(P6, loc); dirv = (gb - ga) / max(Lg, 1e-6); ref = np.array([1.0, 0, 0]) if abs(dirv[0]) < 0.9 else np.array([0, 1.0, 0]); perp = np.cross(dirv, ref); perp /= np.linalg.norm(perp)
    off1 = count(P6, loc + 3 * perp); off2 = count(P6, loc - 3 * perp); fr = count(PF, loc)
    pa = []
    for power in (2, 4):
        cost = 1.0 / (np.clip(C6, 0.02, 1.2) ** power) + 0.05; ia = tuple(np.rint(ga - lo).astype(int).clip(0, np.array(cost.shape) - 1)); ib = tuple(np.rint(gb - lo).astype(int).clip(0, np.array(cost.shape) - 1))
        path, _ = route_through_array(cost, ia, ib, fully_connected=True, geometric=True); path = np.array(path); an = np.unwrap(np.arctan2(np.sin(P6)[tuple(path.T)], np.cos(P6)[tuple(path.T)]))
        pa.append(((an[-1] - an[0]) / (2 * np.pi), float(C6[tuple(path.T)].min()), path + lo))
    cf = samp(C6, loc); kk = ndi.map_coordinates(np.asarray(cosv[sl]).astype(np.float32) / 255.0, loc.T, order=1, mode="nearest"); ks = ndi.uniform_filter1d(kk, 3)
    pk = [q for q in range(1, len(ks) - 1) if ks[q] >= ks[q - 1] and ks[q] > ks[q + 1] and ks[q] > 0.6]
    cnt_cos = len(pk) + (1 if ks[0] > 0.6 and (len(pk) == 0 or pk[0] > 2) else 0) + (1 if ks[-1] > 0.6 and (len(pk) == 0 or pk[-1] < len(ks) - 3) else 0) - 1
    abs_line = line * 2.0 + ORG_B + 0.5 - ORG_B                                                                               # L2 coordinates inside box A
    la = np.rint(abs_line).astype(int).clip(0, np.array(spv.shape) - 1); spl = np.asarray(spv[tuple(la.T)]) > 127; sp_s = int((spl[1:] & ~spl[:-1]).sum())
    pp = np.rint((pa[0][2] * 2.0 + 0.5)).astype(int).clip(0, np.array(spv.shape) - 1); spp = np.asarray(spv[tuple(pp.T)]) > 127; sp_p = int((spp[1:] & ~spp[:-1]).sum())
    gmi = float(ndi.map_coordinates(gmv, (line / 2.0).T, order=1, mode="nearest").sum() * (Lg * 2.0 / n)); gmp = float(gmv[tuple(np.clip(pa[0][2] // 2, 0, np.array(gmv.shape) - 1).T)].sum())
    prof = np.asarray(ctv[tuple(la.T)]).astype(np.float32); ct_p = peaks(prof)
    rows.append((k, v6, off1, off2, fr, pa[0][0], pa[1][0], gmi, gmp, cnt_cos, sp_s, sp_p, ct_p, float(cf.min()), float(cf.mean()), pa[0][1], Lg * 2.0, a[0]))
    if len(rows) % 200 == 0: print(f"  {len(rows)} pairs ({tries} tries), {time.time() - t0:.0f} s", flush=True); np.savez(out, rows=np.array(rows, float))
np.savez(out, rows=np.array(rows, float)); print("saved", out, len(rows), "pairs", flush=True)
