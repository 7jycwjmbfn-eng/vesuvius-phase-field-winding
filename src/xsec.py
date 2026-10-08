"""Paired cross-section on an L3 CT slice with the surfaces of two SURF files (A left, B right); like surfkit_8c.py xsec but the CT file and its origin are arguments.
  python xsec.py SRC_A SRC_B Z Y0 Y1 X0 X1 OUT.png CT_L3.npy ORGZ,ORGY,ORGX"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import surfkit_8c as sk

a, b = sys.argv[1], sys.argv[2]; Z, y0, y1, x0, x1 = (float(v) for v in sys.argv[3:8]); out = sys.argv[8]; ct = np.load(sys.argv[9], mmap_mode="r"); ORG = [int(v) for v in sys.argv[10].split(",")]
img = np.asarray(ct[int(Z) - ORG[0], int(y0) - ORG[1]:int(y1) - ORG[1], int(x0) - ORG[2]:int(x1) - ORG[2]])
fig, ax = plt.subplots(1, 2, figsize=(22, 11 * (y1 - y0) / max(x1 - x0, 1) + 1))
for src, a_, label in ((a, ax[0], "ours"), (b, ax[1], "Will")):
    a_.imshow(img, cmap="gray", extent=[x0, x1, y1, y0]); n = 0
    for i, (name, P) in enumerate(sk.load(src)):
        m = np.isfinite(P).all(-1) & (np.abs(np.nan_to_num(P[..., 0], nan=-1e9) - Z) < 0.6); q = P[m]; q = q[(q[:, 1] >= y0) & (q[:, 1] < y1) & (q[:, 2] >= x0) & (q[:, 2] < x1)]
        if len(q): a_.scatter(q[:, 2], q[:, 1], s=1.5, color=plt.cm.tab20(i % 20)); n += 1
    a_.set_title(f"{label} ({os.path.basename(src)}): {n} surfaces cross z={Z:.0f}")
plt.tight_layout(); plt.savefig(out, dpi=60); print("fig", out)
