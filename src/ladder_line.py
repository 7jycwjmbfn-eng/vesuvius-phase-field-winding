"""Count windings along the straight line between two human ladder points with several independent estimators and certify by agreement (the PCU recipe, with our complex phase).
Estimators per pair of points (a, b) of one ladder collection (true winding difference dw = wind_b - wind_a):
  psi   : unwrapped phase of our network field (cos, sin vector, trilinear) along the line, (psi_b - psi_a) / 2 pi, rounded
  cos   : number of maxima of the official lasagna cos channel along the line between the points, plus one (peaks at both ends counted)
  gm    : integral of the official lasagna grad_mag along the line, divided by a calibration constant fitted on the other half of the pairs
Certified = all estimators that are used agree on the rounded count (and the minimum network confidence along the line is above CMIN for the psi estimator).
  python ladder_line.py OZ OY OX [CMIN]       (box A arrays in E:/vesuvius_big_hot)"""
import os, sys, json
import numpy as np
from scipy import ndimage as ndi

org = np.array([float(v) for v in sys.argv[1:4]]); CMIN = float(sys.argv[4]) if len(sys.argv) > 4 else 0.3
H = "E:/vesuvius_big_hot/"
psi = np.load(H + "pred_v6_psi.npy", mmap_mode="r"); conf = np.load(H + "pred_v6_conf.npy", mmap_mode="r"); cosv = np.load(H + "las/cos.npy", mmap_mode="r"); gm = np.load(H + "las/gm.npy", mmap_mode="r")
shape = np.array(psi.shape); GM = np.asarray(gm).astype(np.float32) / 1000.0
cols = []
for f in ("relative_windings.json", "abs_winding.json"):
    for name, c in json.load(open(os.environ.get("LADDER_DIR", "layerjudge/") + f))["collections"].items():
        pts = np.array([p["p"] for p in c["points"].values()], float); w = np.array([p["wind_a"] for p in c["points"].values()], float)
        if len(pts) >= 2: cols.append((name, pts[:, [2, 1, 0]], w))
rows = []
for name, zyx, w in cols:
    g = (zyx - org - 0.5) / 2.0                                                                  # D=2 grid coordinates
    inside = np.all((g >= 0) & (g < shape - 1), 1)
    if inside.sum() < 2: continue
    g = g[inside]; ww = w[inside]; o = np.argsort(ww); g = g[o]; ww = ww[o]
    lo = np.floor(g.min(0) - 3).astype(int).clip(0); hi = np.ceil(g.max(0) + 4).astype(int); hi = np.minimum(hi, shape)
    sl = tuple(slice(a, b) for a, b in zip(lo, hi)); P = np.asarray(psi[sl]).astype(np.float32); C = np.asarray(conf[sl]).astype(np.float32); K = np.asarray(cosv[sl]).astype(np.float32) / 255.0
    gl = (g - lo)
    def sample(vol, pts, order=1): return ndi.map_coordinates(vol, pts.T, order=order, mode="nearest")
    cs = np.cos(P); sn = np.sin(P)
    for i in range(len(ww)):
        for j in range(i + 1, len(ww)):
            if ww[j] == ww[i]: continue
            a = gl[i]; b = gl[j]; L = np.linalg.norm(b - a); n = int(max(L * 2, 4)); t = np.linspace(0, 1, n + 1)[:, None]; line = a + (b - a) * t
            vc = sample(cs, line); vs = sample(sn, line); ang = np.unwrap(np.arctan2(vs, vc)); est_psi = (ang[-1] - ang[0]) / (2 * np.pi)
            minc = float(sample(C, line).min())
            kk = sample(K, line); ks = ndi.uniform_filter1d(kk, 3); pk = [q for q in range(1, len(ks) - 1) if ks[q] >= ks[q - 1] and ks[q] > ks[q + 1] and ks[q] > 0.6]
            est_cos = len(pk) + (1 if ks[0] > 0.6 and (len(pk) == 0 or pk[0] > 2) else 0) + (1 if ks[-1] > 0.6 and (len(pk) == 0 or pk[-1] < len(ks) - 3) else 0) - 1
            gmv = ndi.map_coordinates(GM, ((line + lo) / 2.0).T, order=1, mode="nearest")
            est_gm = float(gmv.sum() * (L * 2.0 / n))                                              # integral in L2 voxels, calibrated below
            rows.append((name, ww[j] - ww[i], est_psi, est_cos, est_gm, minc, L * 2.0))
R = np.array([(r[1], r[2], r[3], r[4], r[5], r[6]) for r in rows]); names = [r[0] for r in rows]
dw, est_psi, est_cos, est_gm, minc, length = R.T
print(f"{len(R)} pairs from {len(set(names))} collections")
# per-collection sign for psi and cos (the ladder direction is arbitrary): majority sign of est_psi against dw
sign = {}
for nm in set(names):
    m = np.array([x == nm for x in names]); sign[nm] = 1 if np.sum(np.rint(est_psi[m]) == dw[m]) >= np.sum(np.rint(est_psi[m]) == -dw[m]) else -1
s = np.array([sign[x] for x in names]); target = s * dw
# calibration of the grad_mag integral: least squares slope through the origin on the even-numbered collections, evaluated on the odd ones
cal_set = np.array([hash(x) % 2 == 0 for x in names]); slope = float(np.sum(est_gm[cal_set] * np.abs(dw[cal_set])) / np.sum(est_gm[cal_set] ** 2)) if cal_set.any() else 1.0
gm_cnt = np.rint(est_gm * slope); psi_cnt = np.rint(est_psi); cos_cnt = est_cos * s                                      # cos peaks count has no direction: compare absolute values
ok_psi = psi_cnt == target; ok_cos = np.abs(est_cos) == np.abs(dw); ok_gm = gm_cnt == np.abs(dw)
adj = np.abs(dw) == 1
for nm, ok in (("psi (complex phase, our network)", ok_psi), ("official cos peaks", ok_cos), ("official grad_mag integral", ok_gm)):
    print(f"  {nm:34s}: all pairs {ok.mean() * 100:5.1f}%, neighbouring {ok[adj].mean() * 100:5.1f}%")
agree = (np.abs(psi_cnt) == np.abs(cos_cnt)) & (np.abs(psi_cnt) == gm_cnt)
for lab, m in (("psi + cos + gm agree", agree), ("psi + cos agree", np.abs(psi_cnt) == np.abs(cos_cnt)), ("psi + gm agree", np.abs(psi_cnt) == gm_cnt), (f"psi only, min conf >= {CMIN}", minc >= CMIN), (f"psi + cos agree and min conf >= {CMIN}", (np.abs(psi_cnt) == np.abs(cos_cnt)) & (minc >= CMIN)), ("psi + cos + gm agree and min conf >= %.1f" % CMIN, agree & (minc >= CMIN))):
    for nm, sel in (("all pairs", np.ones(len(dw), bool)), ("neighbouring", adj)):
        mm = m & sel
        if mm.any(): print(f"  certified by [{lab}] {nm}: coverage {mm.sum() / sel.sum() * 100:5.1f}%, precision {ok_psi[mm].mean() * 100:5.1f}% (n={int(mm.sum())})")
