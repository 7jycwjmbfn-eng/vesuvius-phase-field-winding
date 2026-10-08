"""Block-parallel PUMA (graph-cut integer labelling of the wrapped phase), see puma_sync.py for the cost.
The volume is cut into blocks (checkerboard colouring of the block grid).  For one move s = +1 / -1 and one colour, all blocks of that colour are solved in parallel processes: the nodes of
the block core may move, the one-voxel halo of face neighbours is frozen (so every edge between a core and a neighbouring block is accounted exactly) and blocks of one colour share no edge,
so the total cost decreases monotonically.  The block grid is shifted by half a block in every second iteration.  k is kept in a memmap (int16), psi and conf are memmaps (float16).
  python puma_blocks.py PSI.npy CONF.npy K0.npy OUT_PREFIX BZ BY BX [WORKERS] [MAXIT] [MINGAIN]
  K0.npy: int16/int32 initial labels (shape of PSI); writes OUT_PREFIX_k.npy and OUT_PREFIX_u.npy (float32).  Block sizes BZ BY BX are core sizes in voxels."""
import sys, time, os
import numpy as np
import maxflow
from multiprocessing import Pool

Q = 32; BIG = 10 ** 9
_G = {}


def sl(axis, lo, hi):
    t = [slice(None)] * 3; t[axis] = slice(lo, hi); return tuple(t)


def edges(psi, conf, N):
    Ws = []; ws = []
    for a in range(3):
        dp = (psi[sl(a, 1, N[a])] - psi[sl(a, 0, N[a] - 1)]) / (2 * np.pi)
        Ws.append(np.rint(dp - np.angle(np.exp(2j * np.pi * dp)) / (2 * np.pi)).astype(np.int32))
        ws.append((0.05 + np.minimum(conf[sl(a, 1, N[a])], conf[sl(a, 0, N[a] - 1)]) ** 2).astype(np.float32))
    return Ws, ws


def cost(k, Ws, ws, N):
    return sum(float((ws[a] * np.abs(Ws[a] + k[sl(a, 1, N[a])] - k[sl(a, 0, N[a] - 1)])).sum()) for a in range(3))


def init(psi_f, conf_f, k_f, shape):
    _G["psi"] = np.load(psi_f, mmap_mode="r"); _G["conf"] = np.load(conf_f, mmap_mode="r"); _G["k"] = np.load(k_f, mmap_mode="r+")


def solve_block(args):
    lo, hi, s = args                                                         # core bounds [lo, hi) per axis, move s
    P = _G["psi"]; C = _G["conf"]; K = _G["k"]; shp = P.shape
    elo = [max(l - 1, 0) for l in lo]; ehi = [min(h + 1, n) for h, n in zip(hi, shp)]                 # extended block with the one-voxel halo
    ex = tuple(slice(a, b) for a, b in zip(elo, ehi))
    psi = np.asarray(P[ex]).astype(np.float32); conf = np.asarray(C[ex]).astype(np.float32); k = np.asarray(K[ex]).astype(np.int32); N = psi.shape
    Ws, ws = edges(psi, conf, N)
    frozen = np.ones(N, bool); frozen[tuple(slice(l - e, h - e) for l, h, e in zip(lo, hi, elo))] = False       # only the core may move
    c0 = cost(k, Ws, ws, N)
    lam = np.zeros(N, np.float32); caps = []
    for a in range(3):
        m = Ws[a] + k[sl(a, 1, N[a])] - k[sl(a, 0, N[a] - 1)]; w = ws[a]
        pair = w * (np.abs(m + s) + np.abs(m - s) - 2 * np.abs(m)); lin = w * (np.abs(m - s) - np.abs(m))
        lam[sl(a, 0, N[a] - 1)] += lin; lam[sl(a, 1, N[a])] -= lin
        cap = np.zeros(N, np.float32); cap[sl(a, 0, N[a] - 1)] = pair; caps.append(np.rint(cap * Q).astype(np.int64))
    lamq = np.rint(lam * Q).astype(np.int64); del lam
    g = maxflow.GraphInt(); ids = g.add_grid_nodes(N)
    for a in range(3):
        st = np.zeros((3, 3, 3)); idx = [1, 1, 1]; idx[a] = 2; st[tuple(idx)] = 1
        g.add_grid_edges(ids, weights=caps[a], structure=st, symmetric=False)
    g.add_grid_tedges(ids, np.maximum(lamq, 0) + BIG * frozen, np.maximum(-lamq, 0))
    g.maxflow(); x = g.get_grid_segments(ids) & ~frozen
    k2 = k + s * x.astype(np.int32); c1 = cost(k2, Ws, ws, N)
    if c1 < c0 - 1e-6:
        core = tuple(slice(l, h) for l, h in zip(lo, hi)); loc = tuple(slice(l - e, h - e) for l, h, e in zip(lo, hi, elo))
        K[core] = k2[loc].astype(np.int16); return c1 - c0
    return 0.0


def main():
    psi_f, conf_f, k0_f, out = sys.argv[1:5]; B = [int(v) for v in sys.argv[5:8]]; workers = int(sys.argv[8]) if len(sys.argv) > 8 else 8
    maxit = int(sys.argv[9]) if len(sys.argv) > 9 else 12; mingain = float(sys.argv[10]) if len(sys.argv) > 10 else 0.003
    P = np.load(psi_f, mmap_mode="r"); shp = P.shape; t0 = time.time()
    k_f = out + "_k.npy"; K = np.lib.format.open_memmap(k_f, mode="w+", dtype=np.int16, shape=shp); K0 = np.load(k0_f, mmap_mode="r")
    for a in range(0, shp[0], 32): K[a:a + 32] = np.asarray(K0[a:a + 32]).astype(np.int16)
    K.flush(); del K
    with Pool(workers, initializer=init, initargs=(psi_f, conf_f, k_f, shp)) as pool:
        total0 = None
        for it in range(maxit):
            off = [(b // 2) * (it % 2) for b in B]                               # half-block shift in every second iteration
            starts = [list(range(-o if o else 0, n, b)) for o, n, b in zip(off, shp, B)]
            blocks = []
            for iz, z in enumerate(starts[0]):
                for iy, y in enumerate(starts[1]):
                    for ix, x in enumerate(starts[2]):
                        lo = [max(z, 0), max(y, 0), max(x, 0)]; hi = [min(z + B[0], shp[0]), min(y + B[1], shp[1]), min(x + B[2], shp[2])]
                        if all(h > l for l, h in zip(lo, hi)): blocks.append(((iz + iy + ix) % 2, lo, hi))
            gain = 0.0
            for s in (1, -1):
                for colour in (0, 1):
                    jobs = [(lo, hi, s) for c, lo, hi in blocks if c == colour]
                    d = sum(pool.imap_unordered(solve_block, jobs, chunksize=1)); gain += d
                    print(f"  iter {it} move {s:+d} colour {colour}: {len(jobs)} blocks, dE {d:.1f}, {time.time() - t0:.0f} s", flush=True)
            if total0 is None: total0 = abs(gain)
            print(f"iter {it}: total dE {gain:.1f}", flush=True)
            if abs(gain) < mingain * max(total0, 1.0) * 0 + 1e-9 or (it > 1 and abs(gain) < mingain * total0): break
    K = np.load(k_f, mmap_mode="r"); psi = np.load(psi_f, mmap_mode="r"); U = np.lib.format.open_memmap(out + "_u.npy", mode="w+", dtype=np.float32, shape=shp)
    for a in range(0, shp[0], 16): U[a:a + 16] = (np.asarray(psi[a:a + 16]).astype(np.float64) / (2 * np.pi) + np.asarray(K[a:a + 16])).astype(np.float32)
    U.flush(); print(f"saved {out}_k.npy, {out}_u.npy, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
