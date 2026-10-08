"""Third pass over the ladder regions (same row order as ladder_run2.py, new columns only): v6 phase counted along minimum-cost paths that avoid low-confidence vortex cores,
plus votes that do not depend on the phase network: surface-prediction crossings (straight line and path) and the number of peaks of the smoothed CT profile.
Columns: ri, dw, path_count_a (cost 1/conf^2), path_count_b (cost 1/conf^4), path_minconf_a, path_len_ratio_a, sp_cross_path, ct_peaks, ct_prominence_sum, lasagna_gm_on_path.
  python ladder_run3.py OUT.npz START END"""
import os, sys, json, time
import numpy as np
import torch
import zarr
from scipy import ndimage as ndi
from skimage.graph import route_through_array
import train_wfield_8c as tw
import archs as A
import predict_v6 as pv

G = "G:/vesuvius_ladder/"; out = sys.argv[1]; regs = json.load(open("E:/vesuvius_ladder_regions.json")); i0 = int(sys.argv[2]); i1 = int(sys.argv[3])
ct_z = zarr.open(G + "ct/2", mode="r"); sp_z = zarr.open(G + "sp/0", mode="r")
las_z = {k: zarr.open(f"{G}lasagna/{n}/{lv}", mode="r") for k, (n, lv) in {"nx": ("nx", 4), "ny": ("ny", 4), "gm": ("grad_mag", 4), "cos": ("cos", 3)}.items()}
m6 = A.Net("resenc").cuda().eval(); m6.load_state_dict(torch.load("G:/vesuvius_wfield_8c/v6_resenc_8kb/ema_final.pt", map_location="cpu")); um = tw.umbilicus()
cols = {}
for f in ("relative_windings.json", "abs_winding.json"):
    for name, c in json.load(open(os.environ.get("LADDER_DIR", "layerjudge/") + f))["collections"].items():
        pts = np.array([p["p"] for p in c["points"].values()], float).reshape(-1, 3)[:, [2, 1, 0]]; w = np.array([p["wind_a"] for p in c["points"].values()], float)
        if len(pts) >= 2: cols[f"{f[:3]}_{name}"] = (pts, w)
rows = []; t0 = time.time(); MARGIN = 8


def peaks(prof, win=25):
    s = ndi.gaussian_filter1d(prof.astype(np.float64), 1.5); base = ndi.uniform_filter1d(s, win); d = s - base; sd = d.std() + 1e-6
    pk = [q for q in range(1, len(d) - 1) if d[q] >= d[q - 1] and d[q] > d[q + 1] and d[q] > 0.5 * sd]
    return len(pk), float(sum(d[q] for q in pk) / sd) if pk else 0.0


for ri in range(i0, i1):
    r = regs[ri]; lo = np.array(r["lo"]); hi = np.array(r["hi"]); pts, w = cols[r["id"]]
    try:
        ct = np.asarray(ct_z[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]); sp = np.asarray(sp_z[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
        sl4 = tuple(slice(l // 4, h // 4) for l, h in zip(lo, hi)); sl2 = tuple(slice(l // 2, h // 2) for l, h in zip(lo, hi))
        las = {"nx": np.asarray(las_z["nx"][sl4]), "ny": np.asarray(las_z["ny"][sl4]), "gm": np.asarray(las_z["gm"][sl4]), "cos": np.asarray(las_z["cos"][sl2])}
    except Exception as e:
        continue
    if (ct > 0).mean() < 0.05: continue
    g = pts - lo; o = np.argsort(w); g = g[o]; ww = w[o]; keep = np.all((g >= 2) & (g < np.array(ct.shape) - 2), 1); g = g[keep]; ww = ww[keep]
    if len(ww) < 2: continue
    pi, pj = np.triu_indices(len(ww), 1); m = ww[pj] != ww[pi]; pi, pj = pi[m], pj[m]
    if len(pi) == 0: continue
    pv.FLIPS = [(0, 0, 0)]
    with torch.no_grad(): psi, conf = pv.predict_chunk(m6, "reg", 0, ct.shape[0], ct, sp, las, um, tuple(int(v) for v in lo))
    psi = psi.astype(np.float32); conf = conf.astype(np.float32); cs = np.cos(psi); sn = np.sin(psi); ang_field = np.arctan2(sn, cs); gm = las["gm"].astype(np.float32) / 1000.0; spb = (sp > 127)
    for k in range(len(pi)):
        a = g[pi[k]]; b = g[pj[k]]; lo_c = np.floor(np.minimum(a, b) - MARGIN).astype(int).clip(0); hi_c = np.ceil(np.maximum(a, b) + MARGIN + 1).astype(int); hi_c = np.minimum(hi_c, ct.shape)
        cc = conf[lo_c[0]:hi_c[0], lo_c[1]:hi_c[1], lo_c[2]:hi_c[2]]; af = ang_field[lo_c[0]:hi_c[0], lo_c[1]:hi_c[1], lo_c[2]:hi_c[2]]
        ia = tuple(np.rint(a - lo_c).astype(int).clip(0, np.array(cc.shape) - 1)); ib = tuple(np.rint(b - lo_c).astype(int).clip(0, np.array(cc.shape) - 1))
        L = float(np.linalg.norm(b - a)); res = []
        if L > 120 or float(np.prod(np.array(cc.shape))) > 4e5:                                                                # long pairs: no path search (memory and time), NaN columns
            res = [(np.nan, np.nan, np.nan, None), (np.nan, np.nan, np.nan, None)]
        for power in (2, 4) if res == [] else ():
            cost = 1.0 / (np.clip(cc, 0.02, 1.2) ** power) + 0.05
            path, _ = route_through_array(cost, ia, ib, fully_connected=True, geometric=True); path = np.array(path)
            an = np.unwrap(af[path[:, 0], path[:, 1], path[:, 2]]); res.append(((an[-1] - an[0]) / (2 * np.pi), float(cc[path[:, 0], path[:, 1], path[:, 2]].min()), len(path) / max(np.linalg.norm(b - a), 1.0), path))
        if res[0][3] is not None:
            path = res[0][3] + lo_c; spp = spb[path[:, 0], path[:, 1], path[:, 2]]; sp_cross = int(((spp[1:]) & (~spp[:-1])).sum()); gmp = float(gm[(path[:, 0] // 4).clip(0, gm.shape[0] - 1), (path[:, 1] // 4).clip(0, gm.shape[1] - 1), (path[:, 2] // 4).clip(0, gm.shape[2] - 1)].sum())
        else: sp_cross = -1; gmp = np.nan
        n = int(max(L, 8)); line = a + (b - a) * np.linspace(0, 1, n + 1)[:, None]; prof = ndi.map_coordinates(ct.astype(np.float32), line.T, order=1, mode="nearest"); npk, prom = peaks(prof)
        rows.append((ri, ww[pj[k]] - ww[pi[k]], res[0][0], res[1][0], res[0][1], res[0][2], sp_cross, npk, prom, gmp))
    print(f"[{ri}] {r['id']}: {len(pi)} pairs, rows {len(rows)}, {time.time() - t0:.0f} s", flush=True)
    if ri % 10 == 0: np.savez(out, rows=np.array(rows, float))
np.savez(out, rows=np.array(rows, float)); print("saved", out, len(rows), "pairs", flush=True)
