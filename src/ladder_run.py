"""For every human winding-ladder collection: read the downloaded region (CT, surface prediction, lasagna), run the v6 phase network, and count windings along the straight line
between every pair of ladder points of the collection with several estimators (our complex phase, official lasagna cos peaks, official lasagna grad_mag integral, surface-prediction
peaks).  One row per pair is written to the output npz for the certifier.
  python ladder_run.py OUT.npz [MODEL.pt] [START] [END]"""
import os, sys, json, time, subprocess
import numpy as np
import torch
import zarr
from scipy import ndimage as ndi
import train_wfield_8c as tw
import archs as A
import predict_v6 as pv

G = "G:/vesuvius_ladder/"; out = sys.argv[1]; mp = sys.argv[2] if len(sys.argv) > 2 else "G:/vesuvius_wfield_8c/v6_resenc_8kb/ema_final.pt"
regs = json.load(open("E:/vesuvius_ladder_regions.json")); i0 = int(sys.argv[3]) if len(sys.argv) > 3 else 0; i1 = int(sys.argv[4]) if len(sys.argv) > 4 else len(regs)
LAS = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHercParis4/representations/predictions/lasagna/20260411134726-lasagna-20260419180421-L2/PHercParis4-20260411134726-las-sd2-5b17ff6c_{}.ome.zarr/{}"
for nm, src in (("ct/2", "G:/vesuvius_big/ct/2/.zarray"), ("sp/0", "G:/vesuvius_big/sp/0/.zarray")):
    if not os.path.exists(G + nm + "/.zarray"): os.makedirs(G + nm, exist_ok=True); open(G + nm + "/.zarray", "wb").write(open(src, "rb").read())
for name, lv in (("cos", 3), ("nx", 4), ("ny", 4), ("grad_mag", 4)):
    p = f"{G}lasagna/{name}/{lv}/.zarray"
    if not os.path.exists(p): os.makedirs(os.path.dirname(p), exist_ok=True); subprocess.run(["curl", "-s", "-m", "60", "-o", p, LAS.format(name, f"{lv}/.zarray")])
ct_z = zarr.open(G + "ct/2", mode="r"); sp_z = zarr.open(G + "sp/0", mode="r")
las_z = {k: zarr.open(f"{G}lasagna/{n}/{lv}", mode="r") for k, (n, lv) in {"nx": ("nx", 4), "ny": ("ny", 4), "gm": ("grad_mag", 4), "cos": ("cos", 3)}.items()}
model = A.Net("resenc").cuda().eval(); model.load_state_dict(torch.load(mp, map_location="cpu")); um = tw.umbilicus()
cols = {}
for f in ("relative_windings.json", "abs_winding.json"):
    for name, c in json.load(open(os.environ.get("LADDER_DIR", "layerjudge/") + f))["collections"].items():
        pts = np.array([p["p"] for p in c["points"].values()], float).reshape(-1, 3)[:, [2, 1, 0]]; w = np.array([p["wind_a"] for p in c["points"].values()], float)
        if len(pts) >= 2: cols[f"{f[:3]}_{name}"] = (pts, w)
rows = []; t0 = time.time()
for ri in range(i0, i1):
    r = regs[ri]; lo = np.array(r["lo"]); hi = np.array(r["hi"]); pts, w = cols[r["id"]]
    try:
        ct = np.asarray(ct_z[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]); sp = np.asarray(sp_z[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
        las = {"nx": np.asarray(las_z["nx"][lo[0] // 4:hi[0] // 4, lo[1] // 4:hi[1] // 4, lo[2] // 4:hi[2] // 4]), "ny": np.asarray(las_z["ny"][lo[0] // 4:hi[0] // 4, lo[1] // 4:hi[1] // 4, lo[2] // 4:hi[2] // 4]),
               "gm": np.asarray(las_z["gm"][lo[0] // 4:hi[0] // 4, lo[1] // 4:hi[1] // 4, lo[2] // 4:hi[2] // 4]), "cos": np.asarray(las_z["cos"][lo[0] // 2:hi[0] // 2, lo[1] // 2:hi[1] // 2, lo[2] // 2:hi[2] // 2])}
    except Exception as e:
        print(f"[{ri}] {r['id']}: read failed {type(e).__name__}", flush=True); continue
    if (ct > 0).mean() < 0.05: print(f"[{ri}] {r['id']}: empty CT, skipped", flush=True); continue
    with torch.no_grad(): psi, conf = pv.predict_chunk(model, "reg", 0, ct.shape[0], ct, sp, las, um, tuple(int(v) for v in lo))
    psi = psi.astype(np.float32); conf = conf.astype(np.float32); cs = np.cos(psi); sn = np.sin(psi)
    g = pts - lo                                                                                 # local L2 coordinates
    o = np.argsort(w); g = g[o]; ww = w[o]
    if not np.all((g >= 2) & (g < np.array(ct.shape) - 2)): keep = np.all((g >= 2) & (g < np.array(ct.shape) - 2), 1); g = g[keep]; ww = ww[keep]
    gm = las["gm"].astype(np.float32) / 1000.0; cosv = las["cos"].astype(np.float32) / 255.0; spf = (sp > 127).astype(np.float32)
    for i in range(len(ww)):
        for j in range(i + 1, len(ww)):
            if ww[j] == ww[i]: continue
            a = g[i]; b = g[j]; L = float(np.linalg.norm(b - a)); n = int(max(L * 1.0, 8)); t = np.linspace(0, 1, n + 1)[:, None]; line = a + (b - a) * t; st = L / n
            samp = lambda vol, pt: ndi.map_coordinates(vol, pt.T, order=1, mode="nearest")
            vc = samp(cs, line); vs = samp(sn, line); ang = np.unwrap(np.arctan2(vs, vc)); est_psi = (ang[-1] - ang[0]) / (2 * np.pi)
            cf = samp(conf, line); kk = samp(cosv, line / 2.0); ks = ndi.uniform_filter1d(kk, 3)
            pk = [q for q in range(1, len(ks) - 1) if ks[q] >= ks[q - 1] and ks[q] > ks[q + 1] and ks[q] > 0.6]
            cnt_cos = len(pk) + (1 if ks[0] > 0.6 and (len(pk) == 0 or pk[0] > 2) else 0) + (1 if ks[-1] > 0.6 and (len(pk) == 0 or pk[-1] < len(ks) - 3) else 0) - 1
            est_gm = float(samp(gm, line / 4.0).sum() * st)
            sps = samp(spf, line); cnt_sp = int(((sps[1:] > 0.5) & (sps[:-1] <= 0.5)).sum())
            rows.append((ri, ww[j] - ww[i], est_psi, cnt_cos, est_gm, float(cf.min()), float(cf.mean()), cnt_sp, L, float(lo[0] + (a[0] + b[0]) / 2)))
    print(f"[{ri}] {r['id']}: {len(ww)} points, rows so far {len(rows)}, {time.time() - t0:.0f} s", flush=True)
    if ri % 10 == 0: np.savez(out, rows=np.array(rows, float))
np.savez(out, rows=np.array(rows, float)); print("saved", out, len(rows), "pairs", flush=True)
