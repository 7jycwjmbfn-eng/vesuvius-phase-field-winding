"""Second version of ladder_run.py: more independent votes per pair and vectorised sampling.
Per region (around one ladder collection) the phase is predicted by v6 (plain), v6 with flipped y, v6 with flipped x, frozen_8c (4 channels) and v4 (4 channels); the windings of every pair
of points are counted along the straight line with each phase field, with two parallel lines (+/- OFF voxels, v6) and with the official lasagna cos peaks, grad_mag integral and the surface-prediction crossings.
Columns: ri, dw, c_v6, c_v6fy, c_v6fx, c_frozen, c_v4, c_off1, c_off2 (unwrapped phase / 2 pi), cos_peaks, gm_integral, sp_crossings, minconf, meanconf, L, zmid.
  python ladder_run2.py OUT.npz START END"""
import os, sys, json, time, subprocess
import numpy as np
import torch
import zarr
from scipy import ndimage as ndi
import train_wfield_8c as tw
import train_v3 as t3
import archs as A
import predict_v6 as pv
import predict as p0

G = "G:/vesuvius_ladder/"; out = sys.argv[1]; regs = json.load(open("E:/vesuvius_ladder_regions.json")); i0 = int(sys.argv[2]); i1 = int(sys.argv[3]); OFF = 3.0
ct_z = zarr.open(G + "ct/2", mode="r"); sp_z = zarr.open(G + "sp/0", mode="r")
las_z = {k: zarr.open(f"{G}lasagna/{n}/{lv}", mode="r") for k, (n, lv) in {"nx": ("nx", 4), "ny": ("ny", 4), "gm": ("grad_mag", 4), "cos": ("cos", 3)}.items()}
m6 = A.Net("resenc").cuda().eval(); m6.load_state_dict(torch.load("G:/vesuvius_wfield_8c/v6_resenc_8kb/ema_final.pt", map_location="cpu"))
mf = tw.UNet().cuda().eval(); mf.load_state_dict(torch.load("G:/vesuvius_wfield_8c/run2/frozen_8c.pt", map_location="cpu"))
m4 = t3.V3Net().cuda().eval(); m4.load_state_dict(torch.load("G:/vesuvius_wfield_8c/v4/model_final.pt", map_location="cpu")); um = tw.umbilicus()
cols = {}
for f in ("relative_windings.json", "abs_winding.json"):
    for name, c in json.load(open(os.environ.get("LADDER_DIR", "layerjudge/") + f))["collections"].items():
        pts = np.array([p["p"] for p in c["points"].values()], float).reshape(-1, 3)[:, [2, 1, 0]]; w = np.array([p["wind_a"] for p in c["points"].values()], float)
        if len(pts) >= 2: cols[f"{f[:3]}_{name}"] = (pts, w)


def v6_field(ct, sp, las, lo, flips):
    pv.FLIPS = flips
    with torch.no_grad(): psi, conf = pv.predict_chunk(m6, "reg", 0, ct.shape[0], ct, sp, las, um, tuple(int(v) for v in lo))
    return psi.astype(np.float32), conf.astype(np.float32)


def old_field(model, ct, sp, lo):
    with torch.no_grad(): psi, conf = p0.predict_chunk2(model, 0, ct.shape[0], ct, sp, um, tuple(int(v) for v in lo))
    return psi.astype(np.float32)


rows = []; t0 = time.time()
for ri in range(i0, i1):
    r = regs[ri]; lo = np.array(r["lo"]); hi = np.array(r["hi"]); pts, w = cols[r["id"]]
    try:
        ct = np.asarray(ct_z[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]); sp = np.asarray(sp_z[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
        sl4 = tuple(slice(l // 4, h // 4) for l, h in zip(lo, hi)); sl2 = tuple(slice(l // 2, h // 2) for l, h in zip(lo, hi))
        las = {"nx": np.asarray(las_z["nx"][sl4]), "ny": np.asarray(las_z["ny"][sl4]), "gm": np.asarray(las_z["gm"][sl4]), "cos": np.asarray(las_z["cos"][sl2])}
    except Exception as e:
        print(f"[{ri}] {r['id']}: read failed {type(e).__name__}", flush=True); continue
    if (ct > 0).mean() < 0.05: print(f"[{ri}] {r['id']}: empty CT, skipped", flush=True); continue
    g = pts - lo; o = np.argsort(w); g = g[o]; ww = w[o]; keep = np.all((g >= 2) & (g < np.array(ct.shape) - 2), 1); g = g[keep]; ww = ww[keep]
    if len(ww) < 2: continue
    pi, pj = np.triu_indices(len(ww), 1); m = ww[pj] != ww[pi]; pi, pj = pi[m], pj[m]
    if len(pi) == 0: continue
    # all lines of this collection at once
    A_ = g[pi]; B_ = g[pj]; Ls = np.linalg.norm(B_ - A_, axis=1); ns = np.maximum(Ls, 8).astype(int); offs = np.r_[0, np.cumsum(ns + 1)]
    dirs = (B_ - A_) / Ls[:, None]; ref = np.where(np.abs(dirs[:, [0]]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]])); perp = np.cross(dirs, ref); perp /= np.linalg.norm(perp, axis=1, keepdims=True)
    coords = np.concatenate([A_[k] + (B_[k] - A_[k]) * np.linspace(0, 1, ns[k] + 1)[:, None] for k in range(len(pi))]); rep = np.repeat(np.arange(len(pi)), ns + 1)
    samp = lambda vol, c, s=1.0: ndi.map_coordinates(vol, (c / s).T, order=1, mode="nearest")

    def unwrap_counts(psi, shift=None):
        c = coords if shift is None else coords + shift[rep]; vc = samp(np.cos(psi), c); vs = samp(np.sin(psi), c); ang = np.arctan2(vs, vc); res = np.zeros(len(pi))
        for k in range(len(pi)):
            a = np.unwrap(ang[offs[k]:offs[k + 1]]); res[k] = (a[-1] - a[0]) / (2 * np.pi)
        return res
    psi6, conf6 = v6_field(ct, sp, las, lo, [(0, 0, 0)]); c_v6 = unwrap_counts(psi6); c_off1 = unwrap_counts(psi6, perp * OFF); c_off2 = unwrap_counts(psi6, -perp * OFF)
    psi6y, _ = v6_field(ct, sp, las, lo, [(0, 1, 0)]); c_fy = unwrap_counts(psi6y); psi6x, _ = v6_field(ct, sp, las, lo, [(0, 0, 1)]); c_fx = unwrap_counts(psi6x); del psi6y, psi6x
    c_fr = unwrap_counts(old_field(mf, ct, sp, lo)); c_v4 = unwrap_counts(old_field(m4, ct, sp, lo))
    cf = samp(conf6, coords); kk = samp(las["cos"].astype(np.float32) / 255.0, coords, 2.0); gmv = samp(las["gm"].astype(np.float32) / 1000.0, coords, 4.0); spv = samp((sp > 127).astype(np.float32), coords)
    for k in range(len(pi)):
        sl = slice(offs[k], offs[k + 1]); ks = ndi.uniform_filter1d(kk[sl], 3)
        pk = [q for q in range(1, len(ks) - 1) if ks[q] >= ks[q - 1] and ks[q] > ks[q + 1] and ks[q] > 0.6]
        cnt_cos = len(pk) + (1 if ks[0] > 0.6 and (len(pk) == 0 or pk[0] > 2) else 0) + (1 if ks[-1] > 0.6 and (len(pk) == 0 or pk[-1] < len(ks) - 3) else 0) - 1
        sps = spv[sl]; cnt_sp = int(((sps[1:] > 0.5) & (sps[:-1] <= 0.5)).sum()); st = Ls[k] / ns[k]
        rows.append((ri, ww[pj[k]] - ww[pi[k]], c_v6[k], c_fy[k], c_fx[k], c_fr[k], c_v4[k], c_off1[k], c_off2[k], cnt_cos, float(gmv[sl].sum() * st), cnt_sp, float(cf[sl].min()), float(cf[sl].mean()), Ls[k], float(lo[0] + (A_[k][0] + B_[k][0]) / 2)))
    print(f"[{ri}] {r['id']}: {len(ww)} points, {len(pi)} pairs, rows {len(rows)}, {time.time() - t0:.0f} s", flush=True)
    if ri % 10 == 0: np.savez(out, rows=np.array(rows, float))
np.savez(out, rows=np.array(rows, float)); print("saved", out, len(rows), "pairs", flush=True)
