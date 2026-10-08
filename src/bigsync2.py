"""Memory-light tile-sync unwrapping for volumes that do not fit in RAM many times over (the big box: 3072 x 3072 x 512 voxels, 1536 x 1536 x 256 after 2x down-sampling).
Pass 1 walks the tile grid slab by slab (z), keeps only the previous slab of unwrapped tiles in memory and records the pairwise integer offsets;
the L1 network LP gives the tile offsets; pass 2 recomputes every tile, adds its offset and writes the tile's core region into a float32 memmap u.
  python bigsync.py PSI.npy DOWN OUT_U.npy [T S WORKERS]
PSI.npy float16 radians [Z, Y, X]; the vector-average down-sampling is done slab by slab."""
import sys, time
import numpy as np
from multiprocessing import Pool
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from skimage.restoration import unwrap_phase

OFFS = [(0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 1, 1), (0, 1, -1), (1, 0, 1), (1, 0, -1), (1, 1, 0), (1, -1, 0)]       # 9 neighbours (half of the 18 face / edge neighbours)
G = {}
LPT = 600                                                                              # LP time limit in seconds (then IRLS)


def irls_sync(n, rows, cols, dd, ww, iters=12):
    """approximate L1 solution of  min sum w |p_cols - p_rows - dd|  by iteratively reweighted least squares, then rounded to integers (tile 0 pinned at 0)"""
    from scipy.sparse.linalg import spsolve
    e = len(rows); A = coo_matrix((np.r_[-np.ones(e), np.ones(e)], (np.r_[np.arange(e), np.arange(e)], np.r_[rows, cols])), shape=(e, n)).tocsr()
    wt = ww.copy(); p = np.zeros(n)
    for it in range(iters):
        W = coo_matrix((wt, (np.arange(e), np.arange(e))), shape=(e, e)).tocsr(); M = (A.T @ W @ A).tocsr() + 1e-6 * coo_matrix((np.ones(n), (np.arange(n), np.arange(n))), shape=(n, n)).tocsr()
        rhs = A.T @ (wt * dd); p = spsolve(M, rhs); p = p - p[0]
        r = np.abs(A @ p - dd); wt = ww / np.maximum(r, 0.05)
    p = np.rint(p).astype(np.int64)
    return icm_refine(n, rows, cols, dd, ww, p)


def icm_refine(n, rows, cols, dd, ww, p, sweeps=8):
    """integer coordinate descent: each tile takes the weighted median of the offsets its overlap pairs ask for (never increases the L1 objective)"""
    e = len(rows); idx = np.r_[rows, cols]; other = np.r_[cols, rows]; sign = np.r_[np.ones(e), -np.ones(e)]; dd2 = np.r_[dd, dd]; w2 = np.r_[ww, ww]
    order = np.argsort(idx, kind="stable"); idx, other, sign, dd2, w2 = idx[order], other[order], sign[order], dd2[order], w2[order]
    starts_ = np.searchsorted(idx, np.arange(n + 1)); dd2 = dd2.astype(np.int64)
    for sw in range(sweeps):
        changed = 0
        for i in range(1, n):                                                             # tile 0 stays pinned
            a, b = starts_[i], starts_[i + 1]
            if a == b: continue
            cand = p[other[a:b]] - (sign[a:b] * dd2[a:b]).astype(np.int64)                # p_i = p_j - d (pair i->j) or p_j + d (pair j->i)
            wt = w2[a:b]; o = np.argsort(cand); cs = np.cumsum(wt[o]); m = cand[o][np.searchsorted(cs, 0.5 * cs[-1])]
            if m != p[i]: p[i] = m; changed += 1
        if changed == 0: break
    return p


def starts(N, T, S):
    s = list(range(0, max(N - T, 0) + 1, S))
    if s[-1] != N - T: s.append(N - T)
    return s


def init(psi_path, D, T):
    P = np.load(psi_path, mmap_mode="r"); G["P"] = P; G["D"] = D; G["T"] = T


def down_tile(z, y, x):
    """vector-averaged D x D x D down-sampled psi tile of T^3 down-sampled voxels at down-sampled origin (z, y, x)"""
    P, D, T = G["P"], G["D"], G["T"]
    blk = np.asarray(P[z * D:(z + T) * D, y * D:(y + T) * D, x * D:(x + T) * D]).astype(np.float32); sh = (T, D, T, D, T, D)
    return np.arctan2(np.sin(blk).reshape(sh).mean((1, 3, 5)), np.cos(blk).reshape(sh).mean((1, 3, 5))).astype(np.float32)


def unwrap_tile(args):
    z, y, x = args
    return (unwrap_phase(down_tile(z, y, x)) / (2 * np.pi)).astype(np.float16)


def slab_tiles(pool, st, i):
    jobs = [(st[0][i], y, x) for y in st[1] for x in st[2]]
    return pool.map(unwrap_tile, jobs, chunksize=16)


def overlap(a0, b0, T):
    """slices of the overlap of two tiles with origins a0, b0 (per axis) inside tile a and tile b, or None"""
    lo = [max(p, q) for p, q in zip(a0, b0)]; hi = [min(p, q) + T for p, q in zip(a0, b0)]
    if any(h - l < 8 for l, h in zip(lo, hi)): return None
    return (tuple(slice(l - p, h - p) for l, h, p in zip(lo, hi, a0)), tuple(slice(l - q, h - q) for l, h, q in zip(lo, hi, b0)))


def main(psi_path, D, out, T=48, S=24, nw=4):
    P = np.load(psi_path, mmap_mode="r"); Zd, Yd, Xd = P.shape[0] // D, P.shape[1] // D, P.shape[2] // D
    st = [starts(n, T, S) for n in (Zd, Yd, Xd)]; nt = [len(s) for s in st]; ntiles = nt[0] * nt[1] * nt[2]; t0 = time.time()
    print(f"volume {P.shape} -> {(Zd, Yd, Xd)}, tiles {nt} = {ntiles}", flush=True)
    idx = lambda i, j, k: (i * nt[1] + j) * nt[2] + k
    rows, cols, dd, ww = [], [], [], []
    with Pool(nw, initializer=init, initargs=(psi_path, D, T)) as pool:
        prev = None
        for i in range(nt[0]):
            cur = slab_tiles(pool, st, i); curd = {(i, j, k): cur[j * nt[2] + k] for j in range(nt[1]) for k in range(nt[2])}
            allt = dict(curd)
            if prev is not None: allt.update(prev)
            for j in range(nt[1]):
                for k in range(nt[2]):
                    key = (i, j, k)
                    for o in OFFS:                                                      # pair (key, key + o): handled when the later tile is the current slab or the earlier is in it
                        k2 = (i + o[0], j + o[1], k + o[2])
                        if o[0] == 1: continue                                           # pairs into the next slab are handled in the next iteration
                        if not (0 <= k2[1] < nt[1] and 0 <= k2[2] < nt[2]): continue
                        ov = overlap((st[0][i], st[1][j], st[2][k]), (st[0][k2[0]], st[1][k2[1]], st[2][k2[2]]), T)
                        if ov is None: continue
                        a_ = curd[key][ov[0]].astype(np.float32); b_ = curd[k2][ov[1]].astype(np.float32)
                        diff = np.rint(a_ - b_).astype(np.int64).ravel(); m0 = diff.min(); cnt = np.bincount(diff - m0)
                        rows.append(idx(*key)); cols.append(idx(*k2)); dd.append(int(cnt.argmax()) + m0); ww.append((cnt.max() / diff.size) ** 4)
            if prev is not None:                                                         # pairs between the previous slab (i-1) and this one
                for j in range(nt[1]):
                    for k in range(nt[2]):
                        for o in OFFS:
                            if o[0] != 1: continue
                            k1 = (i - 1, j, k); k2 = (i, j + o[1], k + o[2])
                            if not (0 <= k2[1] < nt[1] and 0 <= k2[2] < nt[2]): continue
                            ov = overlap((st[0][i - 1], st[1][j], st[2][k]), (st[0][i], st[1][k2[1]], st[2][k2[2]]), T)
                            if ov is None: continue
                            a_ = prev[k1][ov[0]].astype(np.float32); b_ = curd[k2][ov[1]].astype(np.float32)
                            diff = np.rint(a_ - b_).astype(np.int64).ravel(); m0 = diff.min(); cnt = np.bincount(diff - m0)
                            rows.append(idx(*k1)); cols.append(idx(*k2)); dd.append(int(cnt.argmax()) + m0); ww.append((cnt.max() / diff.size) ** 4)
            prev = curd
            print(f"  slab {i + 1}/{nt[0]} done, {len(rows)} pairs, {time.time() - t0:.0f} s", flush=True)
        n, e = ntiles, len(rows); rows = np.array(rows); cols = np.array(cols); dd = np.array(dd, float); ww = np.array(ww)
        A = coo_matrix((np.r_[np.ones(e), -np.ones(e), -np.ones(e), np.ones(e)], (np.r_[np.arange(e), np.arange(e), np.arange(e), np.arange(e)], np.r_[cols, rows, n + np.arange(e), n + e + np.arange(e)])), shape=(e, n + 2 * e)).tocsr()
        np.savez_compressed(out.replace(".npy", "_pairs.npz"), rows=rows, cols=cols, dd=dd, ww=ww, n=n)
        bounds = [(None, None)] * n + [(0, None)] * (2 * e); bounds[0] = (0, 0); t1 = time.time()
        res = linprog(np.r_[np.zeros(n), ww, ww], A_eq=A, b_eq=dd, bounds=bounds, method="highs", options={"time_limit": float(LPT)})
        if res.status == 0: p = np.rint(res.x[:n]).astype(np.int64); how = "LP"
        else: p = irls_sync(n, rows, cols, dd, ww); how = f"IRLS (LP status {res.status}: {res.message[:50]})"
        viol = np.abs(p[cols] - p[rows] - dd) > 0.5; obj = float((ww * np.abs(p[cols] - p[rows] - dd)).sum())
        print(f"sync by {how}: {n} tiles, {e} pairs, {time.time() - t1:.0f} s; violated {viol.mean() * 100:.2f}% (weight share {ww[viol].sum() / ww.sum() * 100:.2f}%), L1 objective {obj:.1f}", flush=True)
        print(f"LP: {n} tiles, {e} pairs (see line above)", flush=True)
        cen = [np.array(s) + T / 2.0 for s in st]; near = [np.abs(np.arange(N)[:, None] - cen[a][None]).argmin(1) for a, N in enumerate((Zd, Yd, Xd))]
        U = np.lib.format.open_memmap(out, mode="w+", dtype=np.float32, shape=(Zd, Yd, Xd))
        for i in range(nt[0]):
            cur = slab_tiles(pool, st, i)
            for j in range(nt[1]):
                for k in range(nt[2]):
                    mz, my, mx = (np.nonzero(near[a] == key)[0] for a, key in enumerate((i, j, k)))
                    if min(len(mz), len(my), len(mx)) == 0: continue
                    ut = cur[j * nt[2] + k]; U[mz[0]:mz[-1] + 1, my[0]:my[-1] + 1, mx[0]:mx[-1] + 1] = ut[np.ix_(mz - st[0][i], my - st[1][j], mx - st[2][k])].astype(np.float32) + p[idx(i, j, k)]
            print(f"  pass 2 slab {i + 1}/{nt[0]} written, {time.time() - t0:.0f} s", flush=True)
        U.flush()
    print(f"saved {out}, total {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    a = sys.argv; LPT = float(a[7]) if len(a) > 7 else 600; main(a[1], int(a[2]), a[3], int(a[4]) if len(a) > 4 else 48, int(a[5]) if len(a) > 5 else 24, int(a[6]) if len(a) > 6 else 4)
