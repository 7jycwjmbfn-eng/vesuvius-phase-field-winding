"""psi (sheet phase) -> psi = 0 level set -> connected surface pieces -> area and sheet purity against the truth winding field.
Surface = zero set of sin(psi) restricted to cos(psi) > 0 (marching cubes with a mask), so it cannot jump between sheets by construction;
a piece contains several sheets only where the phase field itself is wrong.
Purity per piece: every vertex with a truth value gets n = round((Theta_vox - ((az - ALPHA) mod 2pi)) / 2pi) (the truth turn number);
purity = share of the modal n; a switch = a vertex not on the modal turn of its piece.
  python isosurf.py MODEL.pt|TRUTH Z0 Z1 Y0 Y1 X0 X1 [cpu|cuda]      (absolute L2 coordinates inside the W2 box, MODEL as train_wfield_8c / train_v3)
The model path containing 'v3' selects train_v3.V3Net."""
import os, sys
import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from skimage import measure
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import train_wfield_8c as tw
import surfkit_8c as sk

BOX = os.environ.get("BOX", "w2")
ORG = np.array(tw.ORG, float); VOX_CM = 9.6e-4; ALPHA = np.radians(10.7); TH_ORG = np.array([4992, 640, 896])
if BOX == "p0139":                                                                    # 77f5's PHerc0139 box: theta_box at the box voxel grid, seam A0
    import json
    ORG = np.array([4352, 1664, 1280], float); VOX_CM = 9.362e-4; A0 = np.radians(130.06405309494068)
    TH = np.load("G:/vesuvius_77f5/p0139/theta_box.npy", mmap_mode="r"); _c = json.load(open("G:/vesuvius_77f5/p0139/centre_w023.json"))
    _cz, _cy, _cx = (np.array(_c[k], float) for k in ("z", "y", "x"))
else:
    TH = np.load("G:/vesuvius_77f5/theta_w2_L3_raw_77f5.npy", mmap_mode="r")


def theta_wrapped(V):
    """wrapped azimuth relative to the seam, per vertex (V absolute voxel zyx of the active box)"""
    if BOX == "p0139":
        a = np.arctan2(V[:, 1] - np.interp(V[:, 0], _cz, _cy), V[:, 2] - np.interp(V[:, 0], _cz, _cx)) - A0
        return np.angle(np.exp(1j * a))                                              # (-pi, pi]
    return np.mod(sk.theta(V / 2.0) - ALPHA, 2 * np.pi)


def predict_cpu(model, z0, z1, y0, y1, x0, x1, dev="cpu", tile=96, ov=16):
    """psi (radians, float32) on the absolute-L2 block; tiles overlap by ov and only their centres are kept (same scheme as eval_wfield_8c)"""
    ct = np.load(tw.D + "ct_w2.npy", mmap_mode="r"); sp = np.load(tw.D + "sp_w2.npy", mmap_mode="r"); uz, uy, ux = tw.umbilicus()
    S = np.array(ct.shape); lo = np.array([z0, y0, x0]) - ORG.astype(int); hi = np.array([z1, y1, x1]) - ORG.astype(int)
    out = np.zeros(tuple(hi - lo), np.float32); st = tile - 2 * ov
    def starts(a, b, n): return sorted({int(np.clip(v, 0, n - tile)) for v in range(a - ov, b - ov, st)})
    for z in starts(lo[0], hi[0], S[0]):
        for y in starts(lo[1], hi[1], S[1]):
            for x in starts(lo[2], hi[2], S[2]):
                c_ = np.asarray(ct[z:z + tile, y:y + tile, x:x + tile]) / 255.0; s_ = np.asarray(sp[z:z + tile, y:y + tile, x:x + tile]) / 255.0
                zz = ORG[0] + z + tile / 2; cy, cx = np.interp(zz, uz, uy), np.interp(zz, uz, ux)
                gy = ORG[1] + y + np.arange(tile)[:, None] - cy; gx = ORG[2] + x + np.arange(tile)[None, :] - cx; r = np.hypot(gy, gx) + 1e-6
                X = np.stack([c_, s_, np.broadcast_to((gy / r)[None], c_.shape), np.broadcast_to((gx / r)[None], c_.shape)])[None].astype(np.float32)
                with torch.no_grad():
                    p = model(torch.from_numpy(X).to(dev)).float()[0].cpu().numpy()
                psi = np.arctan2(p[0], p[1])
                a = np.maximum([z, y, x], lo) ; b = np.minimum([z + tile, y + tile, x + tile], hi)
                # keep the tile centre unless the tile touches the volume border
                for k, (t0, n) in enumerate(zip((z, y, x), S)):
                    if t0 > 0: a[k] = max(a[k], t0 + ov)
                    if t0 + tile < n: b[k] = min(b[k], t0 + tile - ov)
                if np.any(b <= a): continue
                zs_, ys_, xs_ = (slice(a[k] - (z, y, x)[k], b[k] - (z, y, x)[k]) for k in range(3))
                out[a[0] - lo[0]:b[0] - lo[0], a[1] - lo[1]:b[1] - lo[1], a[2] - lo[2]:b[2] - lo[2]] = psi[zs_, ys_, xs_]
    return out


def truth_psi(z0, z1, y0, y1, x0, x1):
    L = np.load(tw.LAB, mmap_mode="r"); lo = np.array([z0, y0, x0]) - ORG.astype(int); hi = np.array([z1, y1, x1]) - ORG.astype(int)
    a = np.asarray(L[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]).astype(np.float32); valid = a < 255
    return np.where(valid, a * (2 * np.pi / 255), 0.0).astype(np.float32), valid


def pieces(psi, origin, valid=None):
    """marching cubes on sin(psi) = 0 with cos(psi) > 0; returns vertices (absolute L2 zyx), faces, component label per vertex"""
    s = np.sin(psi); c = np.cos(psi); mask = c > 0.0
    if valid is not None: mask &= valid
    mask = ndi.binary_erosion(mask, iterations=1) | False
    V, F, _, _ = measure.marching_cubes(s, 0.0, mask=mask, allow_degenerate=False)
    V = V + np.asarray(origin, float)[None]
    n = len(V); e = np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [0, 2]]]
    g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)); nc, lab = connected_components(g, directed=False)
    return V, F, lab


def tri_area(V, F):
    return 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)


def turn_numbers(V):
    """truth turn n per vertex (NaN where the truth field has no value); V absolute L2 zyx"""
    ix = (np.rint(V).astype(np.int64) - ORG.astype(np.int64)) if BOX == "p0139" else (np.rint(V / 2.0).astype(np.int64) - TH_ORG)
    ok = np.all((ix >= 0) & (ix < np.array(TH.shape)), 1)
    n = np.full(len(V), np.nan); th = np.full(len(V), np.nan)
    o = np.nonzero(ok)[0]; srt = o[np.lexsort((ix[o, 2], ix[o, 1], ix[o, 0]))]; th[srt] = TH[ix[srt, 0], ix[srt, 1], ix[srt, 2]]
    a = theta_wrapped(V); f = np.isfinite(th)
    n[f] = np.rint((th[f] - a[f]) / (2 * np.pi)); return n


def report(V, F, lab, tag):
    ar = tri_area(V, F) * VOX_CM ** 2; fa = np.bincount(lab[F[:, 0]], ar, minlength=lab.max() + 1)
    n = turn_numbers(V); order = np.argsort(-fa); tot_ok = tot_sw = 0.0; rows = []
    for k in order[:200]:
        v = lab == k; nn = n[v]; nn = nn[np.isfinite(nn)]
        if len(nn) < 20: rows.append((k, fa[k], np.nan, len(nn))); continue
        u, c = np.unique(nn, return_counts=True); pur = c.max() / c.sum(); rows.append((k, fa[k], pur, len(nn)))
        tot_ok += fa[k] * pur; tot_sw += fa[k] * (1 - pur)
    print(f"[{tag}] {len(fa)} pieces, total area {fa.sum():.3f} cm2; area-weighted purity of the 200 largest: {tot_ok / max(tot_ok + tot_sw, 1e-9) * 100:.1f}% (switch share {tot_sw / max(tot_ok + tot_sw, 1e-9) * 100:.1f}%)")
    for k, a, p, m_ in rows[:8]: print(f"    piece {k}: area {a:.3f} cm2, truth-valid vertices {m_}, purity {p * 100 if np.isfinite(p) else float('nan'):.1f}%")
    return fa, rows


if __name__ == "__main__":
    src = sys.argv[1]; z0, z1, y0, y1, x0, x1 = (int(v) for v in sys.argv[2:8]); dev = sys.argv[8] if len(sys.argv) > 8 else "cpu"
    origin = (z0, y0, x0)
    tpsi, valid = truth_psi(z0, z1, y0, y1, x0, x1)
    V, F, lab = pieces(tpsi, origin, valid); report(V, F, lab, "truth psi (sanity: purity should be ~100%)")
    if src != "TRUTH":
        if "v3" in os.path.basename(os.path.dirname(src)) or "v3" in src:
            import train_v3 as t3; model = t3.V3Net()
        else:
            model = tw.UNet()
        model.load_state_dict(torch.load(src, map_location="cpu")); model = model.to(dev).eval()
        pp = predict_cpu(model, z0, z1, y0, y1, x0, x1, dev); np.save(f"G:/vesuvius_wfield_8c/iso_pred_{os.path.basename(src)}_{z0}_{y0}_{x0}.npy", pp.astype(np.float16))
        V, F, lab = pieces(pp, origin); report(V, F, lab, f"predicted psi, {os.path.basename(src)}")
        V, F, lab = pieces(pp, origin, valid); report(V, F, lab, f"predicted psi restricted to truth-valid voxels, {os.path.basename(src)}")
