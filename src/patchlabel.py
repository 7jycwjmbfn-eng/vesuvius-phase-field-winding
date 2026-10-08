"""Label consistency of a global unwrapped field u along the human-verified patches (the ruler's patches).  A patch lies on one sheet, so round(u) at its samples must be constant.
For every patch with >= MINS samples inside the box: share of samples whose round(u) equals the patch's modal label; reported sample-weighted and as the number of patches with every sample in the mode,
and the number of label changes between spatially neighbouring samples (sorted along the patch by nearest-neighbour chains is not needed: we count samples outside the mode).
  [BOX_ORG=10496,1920,3328] python patchlabel.py U1.npy [U2.npy ...]"""
import os, sys
import numpy as np
ORG = np.array([float(v) for v in os.environ.get("BOX_ORG", "10496,1920,3328").split(",")]); MINS = 10
S = np.load("E:/vesuvius_downstream_tmp/fiberA/patch_samples_ua.npz"); p = S["pts"].astype(float); cid = S["cid"]
for up in sys.argv[1:]:
    U = np.load(up, mmap_mode="r"); shp = np.array(U.shape); g = (p * 2.0 - ORG - 0.5) / 2.0; ins = np.all((g >= 1) & (g <= shp - 2), 1) & (p[:, 0] >= 5248) & (p[:, 0] < 5504)
    idx = np.nonzero(ins)[0]; gi = np.rint(g[idx]).astype(int)
    lab = np.array([int(np.rint(np.median([U[tuple(np.clip(gi[k] + d, 0, shp - 1))] for d in ((0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))]))) for k in range(len(idx))])
    tot = 0; inmode = 0; full = 0; npatch = 0; bad = []
    for c in np.unique(cid[idx]):
        m = cid[idx] == c
        if m.sum() < MINS: continue
        v, cn = np.unique(lab[m], return_counts=True); npatch += 1; tot += m.sum(); inmode += cn.max(); full += (cn.max() == m.sum())
        if cn.max() < m.sum(): bad.append((int(c), int(m.sum()), int(cn.max())))
    print(f"{up}: patches {npatch}, samples {tot}; share of samples in the patch's modal label {inmode / tot * 100:.2f}%; patches with a single label {full}/{npatch}; samples outside the mode {tot - inmode}")
