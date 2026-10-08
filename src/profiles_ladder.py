"""Winding-counter profiles for the human ladder pairs (test set of the end-to-end counting network).
Same regions, same pair enumeration and same filters as ladder_run3.py (pairs sorted by winding, upper triangle, equal windings removed), so that the pair (ri, seq) lines up with the rows
of E:/vesuvius_ladder2_rows_*.npz and ladder3_rows_*.npz (seq = index inside the region among ALL enumerated pairs, dropped ones included).
Per region: v6 once (FLIPS = [(0, 0, 0)], native L2 resolution) and frozen_8c once, both reduced to the D = 2 grid exactly as the box A arrays (sin and cos averaged over 2x2x2, then arctan2;
confidence = mean of the per-voxel confidence; the psi array is stored as float16 like pred_v6_psi.npy), then one profile per pair along the straight line a -> b.

PROFILE file keys (npz):
  P  float16 [N, 14, 256]  channels: 0 CT/255, 1 surface prediction/255, 2 sin(psi_v6), 3 cos(psi_v6), 4 v6 confidence clipped to [0, 1.2], 5 sin(psi_frozen), 6 cos(psi_frozen),
                           7, 8 sin, cos of psi_v6 on the parallel line shifted by +3 L2 voxels along perp, 9, 10 the same for -3, 11 lasagna cos/255, 12 lasagna grad_mag/255, 13 valid mask
  S  float32 [N, 6]        L/100, r_a/1000, |cos(angle between line and radial direction at a)|, dz/L (signed, b - a), z_mid/10000, jitter (NaN for ladders)
  k  int16 [N]             true winding difference |wind_b - wind_a|
  ri int32, seq int32, dw float32 (|wind_b - wind_a|), zmid float32 (absolute L2 z of the midpoint), n_dropped (int, pairs with L > 255 L2 voxels, counted over all regions of this run)
Sampling: n = min(ceil(L) + 1, 256) points at positions a + (b - a) * linspace(0, 1, n), so the first and last point are exactly the endpoints and the step is L / (n - 1) (0.9 to 1.0 L2 voxel
for L >= 10); the remaining entries are zero.  psi / conf (D = 2 grid): trilinear in sin and cos, coordinate (x - 0.5) / 2 for a region-local L2 coordinate x (same as certlib BoxA.votes);
CT and surface prediction: trilinear on the L2 grid, coordinate x; lasagna cos on the L2/2 grid and grad_mag on the L2/4 grid: coordinate (x - 0.5) / 2 and (x - 0.5) / 4 (as in certlib).
The perpendicular direction is taken as in certlib.votes (ref = [1, 0, 0], or [0, 1, 0] when |dir_z| >= 0.9; perp = normalise(cross(dir, ref))), but the shift is 3 L2 voxels
(certlib shifts by 3 D = 2 voxels = 6 L2 voxels in its coordinates; ladder_run2 shifts by 3 L2 voxels).
Memory: a region whose full-resolution float16 psi + conf exceed 300 MB is predicted in z chunks (multiples of 64 planes) and reduced to D = 2 chunk by chunk; smaller regions use one call, as ladder_run3.py does.
  python profiles_ladder.py OUT.npz START END      (region index range, like ladder_run3.py; saves every region with ri % 10 == 0 and at the end)"""
import os, sys, json, time, gc
import numpy as np
import torch
import zarr
from scipy import ndimage as ndi
import train_wfield_8c as tw
import archs as A
import predict_v6 as pv
import predict as p0

G = "G:/vesuvius_ladder/"; out = sys.argv[1]; regs = json.load(open("E:/vesuvius_ladder_regions.json")); i0 = int(sys.argv[2]); i1 = int(sys.argv[3])
NCH, NMAX, SHIFT, LMAX, CHUNK_BYTES = 14, 256, 3.0, 255, 3e8
ct_z = zarr.open(G + "ct/2", mode="r"); sp_z = zarr.open(G + "sp/0", mode="r")
las_z = {k: zarr.open(f"{G}lasagna/{n}/{lv}", mode="r") for k, (n, lv) in {"nx": ("nx", 4), "ny": ("ny", 4), "gm": ("grad_mag", 4), "cos": ("cos", 3)}.items()}
GPU_CAP_GB = float(os.environ.get("GPU_CAP_GB", "2.4")); SPLIT = int(os.environ.get("SPLIT", "2"))                       # allocator cap (the CUDA context comes on top, about 0.5 GB) and tiles per forward call (the predictors batch 4)
torch.cuda.set_per_process_memory_fraction(GPU_CAP_GB * 2**30 / torch.cuda.get_device_properties(0).total_memory)


class Chunked(torch.nn.Module):
    """forward in pieces of SPLIT tiles to cut the activation memory; keep_ch: keep only the first channels of the output (the v6 regression head; the 24 class channels are not needed for READOUT = reg)"""
    def __init__(self, m, keep_ch=None): super().__init__(); self.m = m; self.keep = keep_ch
    def forward(self, x, **kw): return torch.cat([(self.m(c, **kw)[:, :self.keep] if self.keep else self.m(c, **kw)) for c in x.split(SPLIT)], 0)


_readout = A.Net.readout                                                                                              # mode "reg" only uses the first two output channels (readout: reg = o[:, :2]); the full readout also builds the 24-bin softmax and discards it
A.Net.readout = staticmethod(lambda o, mode=None: o[:, :2].float() if mode == "reg" else _readout(o, mode))
m6 = A.Net("resenc").cuda().eval(); m6.load_state_dict(torch.load("G:/vesuvius_wfield_8c/v6_resenc_8kb/ema_final.pt", map_location="cpu")); m6c = Chunked(m6, 2)
mf = tw.UNet().cuda().eval(); mf.load_state_dict(torch.load("G:/vesuvius_wfield_8c/run2/frozen_8c.pt", map_location="cpu")); mfc = Chunked(mf); um = tw.umbilicus(); uz, uy, ux = um
cols = {}
for f in ("relative_windings.json", "abs_winding.json"):
    for name, c in json.load(open(os.environ.get("LADDER_DIR", "layerjudge/") + f))["collections"].items():
        pts = np.array([p["p"] for p in c["points"].values()], float).reshape(-1, 3)[:, [2, 1, 0]]; w = np.array([p["wind_a"] for p in c["points"].values()], float)
        if len(pts) >= 2: cols[f"{f[:3]}_{name}"] = (pts, w)


def to_d2(psi, conf):
    """psi, conf: float16 [nz, ny, nx] (full L2 resolution, nz even) -> float16 D = 2 arrays (vector average of psi, mean of conf); y and x are cropped to even size"""
    nz, ny, nx = psi.shape; ny2, nx2 = ny // 2 * 2, nx // 2 * 2; po = np.empty((nz // 2, ny2 // 2, nx2 // 2), np.float16); co = np.empty_like(po)
    for k in range(0, nz, 16):
        q = psi[k:k + 16, :ny2, :nx2].astype(np.float32); sh = (q.shape[0] // 2, 2, ny2 // 2, 2, nx2 // 2, 2)
        sn = np.sin(q).reshape(sh).mean((1, 3, 5)); cs = np.cos(q).reshape(sh).mean((1, 3, 5)); o0 = k // 2
        po[o0:o0 + sn.shape[0]] = np.arctan2(sn, cs).astype(np.float16); co[o0:o0 + sn.shape[0]] = conf[k:k + 16, :ny2, :nx2].astype(np.float32).reshape(sh).mean((1, 3, 5)).astype(np.float16)
    return po, co


def field_d2(fn, shape0, ny, nx):
    """run a predictor fn(za0, za1) -> (psi, conf) and return the D = 2 arrays.  A region whose full-resolution float16 psi + conf stay below CHUNK_BYTES is predicted in ONE call (identical to
    ladder_run3.py); a bigger region goes in z chunks of a multiple of 64 planes (the tile stride) to bound the memory, each chunk reduced to D = 2 at once"""
    z2 = shape0 // 2 * 2; zch = 64 * max(1, int(CHUNK_BYTES // (64 * ny * nx * 4))); po = np.empty((z2 // 2, ny // 2, nx // 2), np.float16); co = np.empty_like(po)
    for a in range(0, z2, zch):
        b = min(a + zch, z2); p, r = fn(a, b); pd, cd = to_d2(p, r); po[a // 2:b // 2] = pd; co[a // 2:b // 2] = cd; del p, r, pd, cd
    return po, co


def profiles(ri, lo, ct, sp, las, A_, B_, Ls, psi6d, conf6d, psifd):
    """profiles of the pairs (A_, B_ region-local L2 coordinates, Ls = lengths, all <= LMAX) -> P [n, 14, 256] float16, S [n, 6]"""
    n_pair = len(A_); ns = np.minimum(np.ceil(Ls).astype(int) + 1, NMAX); ns = np.maximum(ns, 2); offs = np.r_[0, np.cumsum(ns)]; rep = np.repeat(np.arange(n_pair), ns)
    pos = np.concatenate([np.arange(n) for n in ns]); t = np.concatenate([np.minimum(np.arange(n), L_) / max(L_, 1e-6) for n, L_ in zip(ns, Ls)])                        # step of 1 L2 voxel from a, last point on b (same convention as profiles_truth)
    coords = A_[rep] + (B_ - A_)[rep] * t[:, None]
    d = (B_ - A_) / np.maximum(Ls, 1e-6)[:, None]; ref = np.where(np.abs(d[:, [0]]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]])); perp = np.cross(d, ref); perp /= np.maximum(np.linalg.norm(perp, axis=1, keepdims=True), 1e-9)
    P = np.zeros((n_pair, NCH, NMAX), np.float16)

    def put(ch, v): P[rep, ch, pos] = v.astype(np.float16)

    def samp(vol, c): return ndi.map_coordinates(vol, c.T, order=1, mode="nearest", output=np.float32)
    put(0, samp(ct, coords) / 255.0); put(1, samp(sp, coords) / 255.0)
    s6 = np.sin(psi6d.astype(np.float32)); c6 = np.cos(psi6d.astype(np.float32)); cd = (coords - 0.5) / 2.0
    put(2, samp(s6, cd)); put(3, samp(c6, cd)); put(4, np.clip(samp(conf6d.astype(np.float32), cd), 0.0, 1.2))
    for ch, sgn in ((7, 1.0), (9, -1.0)):
        cs_ = (coords + sgn * SHIFT * perp[rep] - 0.5) / 2.0; put(ch, samp(s6, cs_)); put(ch + 1, samp(c6, cs_))
    del s6, c6
    sf = np.sin(psifd.astype(np.float32)); put(5, samp(sf, cd)); del sf; cf = np.cos(psifd.astype(np.float32)); put(6, samp(cf, cd)); del cf
    put(11, samp(las["cos"], (coords - 0.5) / 2.0) / 255.0); put(12, samp(las["gm"], (coords - 0.5) / 4.0) / 255.0); put(13, np.ones(len(rep)))
    za = lo[0] + A_[:, 0]; cy = np.interp(za, uz, uy); cx = np.interp(za, uz, ux); gy = lo[1] + A_[:, 1] - cy; gx = lo[2] + A_[:, 2] - cx; r = np.hypot(gy, gx) + 1e-6
    S = np.c_[Ls / 100.0, r / 1000.0, np.abs(d[:, 1] * gy + d[:, 2] * gx) / r, (B_[:, 0] - A_[:, 0]) / np.maximum(Ls, 1e-6), (lo[0] + (A_[:, 0] + B_[:, 0]) / 2) / 10000.0, np.full(n_pair, np.nan)].astype(np.float32)
    return P, S


def save(path, parts, n_dropped):
    if parts: P = np.concatenate([q["P"] for q in parts]); S = np.concatenate([q["S"] for q in parts]); k = np.concatenate([q["k"] for q in parts]); ri_ = np.concatenate([q["ri"] for q in parts]); seq = np.concatenate([q["seq"] for q in parts]); dw = np.concatenate([q["dw"] for q in parts]); zm = np.concatenate([q["zmid"] for q in parts])
    else: P = np.zeros((0, NCH, NMAX), np.float16); S = np.zeros((0, 6), np.float32); k = np.zeros(0, np.int16); ri_ = np.zeros(0, np.int32); seq = np.zeros(0, np.int32); dw = np.zeros(0, np.float32); zm = np.zeros(0, np.float32)
    np.savez(path, P=P, S=S, k=k, ri=ri_, seq=seq, dw=dw, zmid=zm, n_dropped=np.int64(n_dropped))


parts = []; n_dropped = 0; t0 = time.time(); times = []
for ri in range(i0, i1):
    t_reg = time.time(); r = regs[ri]; lo = np.array(r["lo"]); hi = np.array(r["hi"]); pts, w = cols[r["id"]]
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
    A_ = g[pi]; B_ = g[pj]; Ls = np.linalg.norm(B_ - A_, axis=1); sel = np.nonzero(Ls <= LMAX)[0]; n_dropped += len(pi) - len(sel)
    if len(sel) == 0: print(f"[{ri}] {r['id']}: {len(pi)} pairs, all longer than {LMAX} L2 voxels, dropped", flush=True); continue
    t_pred = time.time(); S0, S1, S2 = ct.shape; pv.FLIPS = [(0, 0, 0)]
    psi6d, conf6d = field_d2(lambda a, b: pv.predict_chunk(m6c, "reg", a, b, ct, sp, las, um, tuple(int(v) for v in lo)), S0, S1 // 2 * 2, S2 // 2 * 2)
    psifd, _ = field_d2(lambda a, b: p0.predict_chunk2(mfc, a, b, ct, sp, um, tuple(int(v) for v in lo)), S0, S1 // 2 * 2, S2 // 2 * 2); t_pred = time.time() - t_pred
    t_prof = time.time(); P, S = profiles(ri, lo, ct, sp, las, A_[sel], B_[sel], Ls[sel], psi6d, conf6d, psifd); t_prof = time.time() - t_prof
    dwv = (ww[pj] - ww[pi])[sel]
    parts.append(dict(P=P, S=S, k=np.rint(np.abs(dwv)).astype(np.int16), ri=np.full(len(sel), ri, np.int32), seq=sel.astype(np.int32), dw=np.abs(dwv).astype(np.float32), zmid=(lo[0] + (A_[sel, 0] + B_[sel, 0]) / 2).astype(np.float32)))
    del ct, sp, las, psi6d, conf6d, psifd; gc.collect(); times.append((ri, S0 * S1 * S2 / 1e6, len(sel), t_pred, t_prof, time.time() - t_reg))
    print(f"[{ri}] {r['id']}: {len(pi)} pairs, kept {len(sel)}, dropped {len(pi) - len(sel)}; predict {t_pred:.1f} s, profiles {t_prof:.1f} s, region {time.time() - t_reg:.1f} s, total {time.time() - t0:.0f} s, GPU peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB allocated, {torch.cuda.max_memory_reserved() / 2**30:.2f} GB reserved", flush=True)
    if ri % 10 == 0: save(out, parts, n_dropped)
save(out, parts, n_dropped); print("saved", out, sum(len(q["k"]) for q in parts), "pairs, dropped (L >", LMAX, ")", n_dropped, flush=True)
