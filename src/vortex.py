"""Defect (vortex) density of a predicted phase volume: the number of unit squares (in the three coordinate planes) whose wrapped phase differences around the
square do not sum to zero, per million squares.  psi is vector-averaged by DOWN first.
  python vortex.py PSI.npy ZLO ZHI [DOWN] [CONF.npy MINCONF]"""
import sys
import numpy as np
f = sys.argv[1]; zlo, zhi = int(sys.argv[2]), int(sys.argv[3]); D = int(sys.argv[4]) if len(sys.argv) > 4 else 2
P = np.load(f, mmap_mode="r")[zlo:zhi]; Z, Y, X = P.shape[0] // D, P.shape[1] // D, P.shape[2] // D
s = np.empty((Z, Y, X), np.float32); c = np.empty((Z, Y, X), np.float32)
for k in range(0, Z, 16):
    k1 = min(k + 16, Z); blk = np.asarray(P[k * D:k1 * D, :Y * D, :X * D]).astype(np.float32); sh = (k1 - k, D, Y, D, X, D)
    s[k:k1] = np.sin(blk).reshape(sh).mean((1, 3, 5)); c[k:k1] = np.cos(blk).reshape(sh).mean((1, 3, 5))
psi = np.arctan2(s, c); del s, c
def w(a): return np.angle(np.exp(1j * a))
tot = 0; bad = 0
for (a0, a1) in ((0, 1), (0, 2), (1, 2)):                                          # the three coordinate planes
    sl = lambda i, j: tuple(slice(i if ax == a0 else (j if ax == a1 else 0), (psi.shape[ax] - 1 + i) if ax == a0 else ((psi.shape[ax] - 1 + j) if ax == a1 else psi.shape[ax])) for ax in range(3))
    p00 = psi[sl(0, 0)]; p10 = psi[sl(1, 0)]; p11 = psi[sl(1, 1)]; p01 = psi[sl(0, 1)]
    circ = w(p10 - p00) + w(p11 - p10) + w(p01 - p11) + w(p00 - p01); k_ = np.rint(circ / (2 * np.pi)); tot += k_.size; bad += int((k_ != 0).sum())
print(f"{f.split('/')[-1]} rows {zlo}-{zhi} (down {D}): vortex squares {bad} of {tot}: {bad / tot * 1e6:.0f} per million")
