"""bigextract.py in z slabs: the same per-layer level sets (band mask |u - k| < BAND, EROSION erosions, marching cubes on u - k), but every layer k is processed
in slabs of SLAB grid layers (plus the one shared grid layer between neighbouring slabs and an erosion margin), and the slab meshes are welded on identical vertex
coordinates.  Peak memory per worker is about 4 GB however tall the volume is; the result equals bigextract.py (checked on a sub-volume).
  [EROSION=1] python bigextract2.py U.npy DOWN ORIGIN_Z ORIGIN_Y ORIGIN_X OUT.npz [MIN_AREA] [BAND] [WORKERS] [SLAB]"""
import os, sys, time
import numpy as np
from multiprocessing import Pool
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from skimage import measure

VOX_CM = 9.6e-4


def tri_area(V, F):
    return 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)


_G = {}


def _init(uf, D, org, amin, band, slab):
    _G.update(uf=uf, D=D, org=org, amin=amin, band=band, slab=slab); _G["u"] = np.load(uf, mmap_mode="r")
    cf = os.environ.get("CONF_FILE"); _G["conf"] = np.load(cf, mmap_mode="r") if cf else None; _G["cmin"] = float(os.environ.get("CONF_MIN", "0"))


def _one_k(k):
    uu = _G["u"]; D = _G["D"]; org = _G["org"]; amin = _G["amin"]; band = _G["band"]; slab = _G["slab"]; Z = uu.shape[0]; ero = int(os.environ.get("EROSION", "1"))
    Vs = []; Fs = []; nv = 0
    for a in range(0, Z - 1, slab):
        b = min(a + slab, Z - 1)                                                              # cells between grid layers a .. b (b is shared with the next slab)
        lo = max(a - ero, 0); hi = min(b + 1 + ero, Z)
        sub = np.asarray(uu[lo:hi])
        m = np.abs(sub - k) < band
        if _G.get("conf") is not None: m &= (np.asarray(_G["conf"][lo:hi]) >= _G["cmin"])                       # confidence gate: CONF_FILE (u grid, float16), CONF_MIN
        if m.sum() < 10: continue
        if ero > 0: m = ndi.binary_erosion(m, iterations=ero)
        m = m[a - lo:b + 1 - lo]
        if m.sum() < 10: continue
        try: V, F, _, _ = measure.marching_cubes(sub[a - lo:b + 1 - lo] - k, 0.0, mask=m, allow_degenerate=False)
        except Exception: continue
        V[:, 0] += a; Vs.append(V.astype(np.float64)); Fs.append(F + nv); nv += len(V)
    if not Vs: return k, [], 0.0
    V = np.concatenate(Vs); F = np.concatenate(Fs); del Vs, Fs
    key = np.round(V * 1000).astype(np.int64); key = (key[:, 0] << 42) ^ (key[:, 1] << 21) ^ key[:, 2]
    _, first, inv = np.unique(key, return_index=True, return_inverse=True); V = V[first]; F = inv[F]
    F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])]
    n = len(V); e = np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [0, 2]]]
    nc, lab = connected_components(coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)), directed=False)
    V = V * D + D / 2.0 - 0.5 + np.asarray(org, float)[None]
    ar = tri_area(V, F) * VOX_CM ** 2; pa = np.bincount(lab[F[:, 0]], ar, minlength=nc); out = []
    for c in np.nonzero(pa >= amin)[0]:
        vi = np.nonzero(lab == c)[0]; mp = np.full(n, -1, np.int64); mp[vi] = np.arange(len(vi)); fi = np.nonzero(lab[F[:, 0]] == c)[0]
        out.append((V[vi].astype(np.float32), mp[F[fi]].astype(np.int32), pa[c]))
    return k, out, float(pa.sum())


def main(uf, D, org, out, amin=0.02, band=0.45, nw=2, slab=96):
    u = np.load(uf, mmap_mode="r"); t0 = time.time()
    samp = np.asarray(u[::8, ::8, ::8]); kmin, kmax = int(np.floor(samp.min())) - 1, int(np.ceil(samp.max())) + 1
    print(f"u {u.shape}, layers {kmin} .. {kmax}, {nw} workers, slab {slab}", flush=True); del u
    store = {"V": [], "F": [], "k": [], "area": []}; total_area = 0.0
    with Pool(nw, initializer=_init, initargs=(uf, D, org, amin, band, slab)) as pool:
        for k, pieces, ta in pool.imap_unordered(_one_k, range(kmin, kmax + 1)):
            total_area += ta
            for V, F, a in pieces: store["V"].append(V); store["F"].append(F); store["k"].append(k); store["area"].append(a)
            print(f"  k {k}: kept {len(pieces)} pieces (area {sum(p[2] for p in pieces):.2f} cm2), {time.time() - t0:.0f} s", flush=True)
    area = np.array(store["area"]); order = np.argsort(-area)
    print(f"kept {len(area)} pieces >= {amin} cm2, total kept area {area.sum():.2f} cm2 of {total_area:.2f} cm2; largest {area[order[:8]].round(2).tolist()}", flush=True)
    sizes = [len(v) for v in store["V"]]; offs = np.r_[0, np.cumsum(sizes)]; foffs = np.r_[0, np.cumsum([len(f) for f in store["F"]])]
    np.savez_compressed(out, V=np.concatenate(store["V"]), F=np.concatenate(store["F"]), voff=offs, foff=foffs, k=np.array(store["k"]), area=area)
    print("saved", out, f"{time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    a = sys.argv; main(a[1], int(a[2]), (float(a[3]), float(a[4]), float(a[5])), a[6], float(a[7]) if len(a) > 7 else 0.02, float(a[8]) if len(a) > 8 else 0.45, int(a[9]) if len(a) > 9 else 2, int(a[10]) if len(a) > 10 else 96)
