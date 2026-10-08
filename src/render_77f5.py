"""Flattened CT rendering of surfaces (SURF npz or tifxyz dirs, anything surfkit_8c.load reads).
The grid's own (row, col) is the 2D parameter; coarse grids are upsampled (bilinear on xyz) to ~1 L3 voxel per pixel.
Per pixel the CT (ctW2_L3, level 3) is sampled along the grid normal at offsets -3..+3 voxels:
7 layer images, a mean and a max image (PNG at native resolution). Optional red overlay of a per-cell int8 mark grid.
  python render_77f5.py SRC NAME[,NAME...] OUTDIR [--switch DUMP.npy]     (NAME = surface name in SRC, or "all")
  As a module: render(P, marks=None) -> dict of images; save_pngs(...)."""
import argparse, os
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
import surfkit_8c as sk

CT_PATH = os.environ.get("CT_PATH", "E:/vesuvius_downstream_tmp/fiberA/ctW2_L3.npy"); CT_ORG = np.array([float(v) for v in os.environ.get("CT_ORG", "4992,640,896").split(",")]).astype(int)   # env overrides: L3 CT file and its L3 origin
OFFS = np.arange(-3, 4)


def grid_step(P):
    v = np.isfinite(P).all(-1)
    dr = np.linalg.norm(P[1:] - P[:-1], axis=-1)[v[1:] & v[:-1]]
    dc = np.linalg.norm(P[:, 1:] - P[:, :-1], axis=-1)[v[:, 1:] & v[:, :-1]]
    return float(np.median(np.r_[dr, dc])) if len(dr) + len(dc) else 1.0


def upsample(P, f):
    """bilinear upsampling of the xyz grid by integer factor f; a pixel is valid only if its 4 corner cells are valid"""
    if f <= 1: return P, np.arange(P.shape[0]), np.arange(P.shape[1])
    H, W, _ = P.shape
    r = np.arange(0, H - 1 + 1e-9, 1.0 / f); c = np.arange(0, W - 1 + 1e-9, 1.0 / f)
    r0 = np.minimum(np.floor(r).astype(int), H - 2); c0 = np.minimum(np.floor(c).astype(int), W - 2)
    fr = (r - r0)[:, None, None]; fc = (c - c0)[None, :, None]
    A = P[r0][:, c0]; B = P[r0][:, c0 + 1]; C = P[r0 + 1][:, c0]; D = P[r0 + 1][:, c0 + 1]
    Q = (1 - fr) * ((1 - fc) * A + fc * B) + fr * ((1 - fc) * C + fc * D)    # NaN propagates from any invalid corner
    return Q, r, c


def normals(Q):
    """unit normals from grid derivatives (central differences, one-sided at edges / next to NaN)"""
    gr = np.gradient(Q, axis=0); gc = np.gradient(Q, axis=1)
    n = np.cross(gr, gc); n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    return n


def sample_ct(Q, n):
    """[7, h, w] CT values at Q + d*n, d = -3..3; NaN where Q invalid. Reads only the bounding box of the surface."""
    ct = np.load(CT_PATH, mmap_mode="r")
    v = np.isfinite(Q).all(-1) & np.isfinite(n).all(-1)
    out = np.full((len(OFFS),) + Q.shape[:2], np.nan, np.float32)
    if not v.any(): return out
    lo = np.maximum(np.floor(Q[v].min(0) - 5).astype(int) - CT_ORG, 0)
    hi = np.minimum(np.ceil(Q[v].max(0) + 6).astype(int) - CT_ORG, np.array(ct.shape))
    box = np.ascontiguousarray(ct[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
    for k, d in enumerate(OFFS):
        c = (Q[v] + d * n[v] - CT_ORG - lo).T
        inb = np.all((c >= 0) & (c <= (np.array(box.shape) - 1)[:, None]), 0)
        val = np.full(c.shape[1], np.nan, np.float32)
        val[inb] = ndi.map_coordinates(box, c[:, inb], order=1)
        o = out[k]; o[v] = val
    return out


def to_u8(img, lo=None, hi=None):
    v = np.isfinite(img)
    if lo is None: lo, hi = (np.percentile(img[v], [1, 99]) if v.any() else (0, 1))
    u = np.clip((np.nan_to_num(img, nan=lo) - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    u[~v] = 0
    return u


def render(P, marks=None):
    """P [H, W, 3] L3 zyx (NaN invalid); marks [H, W] int8 or None. Returns dict of uint8 images (+ RGB overlay)."""
    f = max(1, int(round(grid_step(P))))
    Q, r, c = upsample(P, f); n = normals(Q)
    L = sample_ct(Q, n)
    with np.errstate(all="ignore"):
        mean = np.nanmean(L, 0); mx = np.nanmax(L, 0)
    v = np.isfinite(mean); lo, hi = (np.percentile(mean[v], [1, 99]) if v.any() else (0, 1))
    out = {f"layer{d:+d}": to_u8(L[k], lo, hi) for k, d in enumerate(OFFS)}
    out["mean"] = to_u8(mean, lo, hi); out["max"] = to_u8(mx)
    if marks is not None:
        mk = marks[np.clip(np.rint(r).astype(int), 0, P.shape[0] - 1)][:, np.clip(np.rint(c).astype(int), 0, P.shape[1] - 1)] > 0
        rgb = np.repeat(out["mean"][..., None], 3, -1)
        rgb[mk] = (0.4 * rgb[mk] + 0.6 * np.array([255, 0, 0])).astype(np.uint8)
        out["overlay"] = rgb
    out["_factor"] = f
    return out


def save_pngs(imgs, prefix):
    paths = []
    for k, im in imgs.items():
        if k.startswith("_"): continue
        p = f"{prefix}_{k}.png"; Image.fromarray(im).save(p); paths.append(p)
    return paths


def switch_marks(P, name, names, dump, dilate=2):
    """mark grid cells nearest to the 'beside' samples of surfkit's switch dump for this array"""
    S = np.load(sk.T + "patch_samples_ua.npz")["pts"].astype(np.float64)
    k = list(names).index(name); sel = (dump[:, 1] // 100000).astype(int) == k
    mk = np.zeros(P.shape[:2], np.int8)
    if not sel.any(): return mk, 0
    v = np.isfinite(P).all(-1); idx = np.argwhere(v)
    d, j = cKDTree(P[v]).query(S[dump[sel, 2].astype(int)], distance_upper_bound=25)
    j = j[np.isfinite(d)]
    for rr, cc in idx[j]: mk[rr, cc] = 1
    if dilate: mk = ndi.binary_dilation(mk, iterations=dilate).astype(np.int8) & v.astype(np.int8)
    return mk, int(sel.sum())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("names"); ap.add_argument("outdir")
    ap.add_argument("--switch", default=""); ap.add_argument("--tag", default="_77f5")
    a = ap.parse_args(); os.makedirs(a.outdir, exist_ok=True)
    surfs = sk.load(a.src); want = None if a.names == "all" else a.names.split(",")
    dump = np.load(a.switch) if a.switch else None
    dnames = np.load(a.switch.replace(".npy", "_names.npy")) if a.switch else None
    for name, P in surfs:
        if want is not None and name not in want: continue
        marks, nsw = (switch_marks(P, name, dnames, dump) if dump is not None else (None, 0))
        imgs = render(P, marks)
        paths = save_pngs(imgs, os.path.join(a.outdir, f"flat_{name}{a.tag}"))
        print(f"{name}: grid {P.shape[:2]}, upsample x{imgs['_factor']}, image {imgs['mean'].shape}, switch samples {nsw}, "
              f"marked cells {int(marks.sum()) if marks is not None else 0} -> {paths[0]} ... ({len(paths)} files)", flush=True)


if __name__ == "__main__":
    main()
