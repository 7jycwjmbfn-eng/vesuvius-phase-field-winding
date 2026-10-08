"""Remove the neighbourhood of phase vortex lines from the confidence: conf2 = conf where the distance to the nearest vortex plaquette is >= R grid cells, else 0.
A vortex is a unit square of the (down-sampled) phase whose wrapped differences around the square do not sum to zero.  Slab-wise with a margin, so memory stays small.
  python carve.py PSI.npy CONF.npy R OUT.npy"""
import sys
import numpy as np
from scipy import ndimage as ndi
psi_f, conf_f, R, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
P = np.load(psi_f, mmap_mode="r"); C = np.load(conf_f, mmap_mode="r"); Z = P.shape[0]; M = int(np.ceil(R)) + 2; SL = 48
O = np.lib.format.open_memmap(out, mode="w+", dtype=np.float16, shape=P.shape); w = lambda a: np.angle(np.exp(1j * a)); tot = 0; cut = 0
for a in range(0, Z, SL):
    b = min(a + SL, Z); lo = max(a - M, 0); hi = min(b + M, Z); psi = np.asarray(P[lo:hi]).astype(np.float32); vor = np.zeros(psi.shape, bool)
    for (a0, a1) in ((0, 1), (0, 2), (1, 2)):
        sl = lambda i, j: tuple(slice(i if ax == a0 else (j if ax == a1 else 0), (psi.shape[ax] - 1 + i) if ax == a0 else ((psi.shape[ax] - 1 + j) if ax == a1 else psi.shape[ax])) for ax in range(3))
        bad = np.rint((w(psi[sl(1, 0)] - psi[sl(0, 0)]) + w(psi[sl(1, 1)] - psi[sl(1, 0)]) + w(psi[sl(0, 1)] - psi[sl(1, 1)]) + w(psi[sl(0, 0)] - psi[sl(0, 1)])) / (2 * np.pi)) != 0
        for da in (0, 1):
            for db in (0, 1):
                tgt = tuple(slice(da if ax == a0 else (db if ax == a1 else 0), (psi.shape[ax] - 1 + da) if ax == a0 else ((psi.shape[ax] - 1 + db) if ax == a1 else psi.shape[ax])) for ax in range(3))
                vor[tgt] |= bad
    d = ndi.distance_transform_edt(~vor) if vor.any() else np.full(psi.shape, 1e9)
    keep = (d >= R)[a - lo:b - lo]; c = np.asarray(C[a:b]).astype(np.float32); O[a:b] = np.where(keep, c, 0).astype(np.float16); tot += keep.size; cut += int((~keep).sum())
    print(f"  slab {a}-{b}: carved {(~keep).mean() * 100:.1f}%", flush=True)
O.flush(); print(f"carved {cut / tot * 100:.1f}% of the cells (R = {R}) -> {out}")
