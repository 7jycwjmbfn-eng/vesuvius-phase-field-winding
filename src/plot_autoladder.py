"""Figure: automatic ladders over a CT slice of box A.  Points of ladders that lie within +-DZ layers of the slice are drawn on the CT; the label is the wind number (relative within the ladder).
  python plot_autoladder.py LADDERS.json OUT.png [Z0] [Y0] [X0] [SIZE]       (L2 coordinates, defaults: densest slice)"""
import sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open(sys.argv[1]))["collections"]; out = sys.argv[2]; ORG = np.array([10496, 1920, 3328]); DZ = 12
P = [];
for cid, c in d.items():
    for pid, p in c["points"].items(): P.append((int(cid), p["wind_a"], p["p"][2], p["p"][1], p["p"][0]))
P = np.array(P); print(len(d), "ladders", len(P), "points")
if len(sys.argv) > 3:
    z0 = int(sys.argv[3]); y0 = int(sys.argv[4]); x0 = int(sys.argv[5]); size = int(sys.argv[6])
else:
    size = 900; best = (0, None)
    for z0_ in range(10510, 11000, 10):
        s_ = P[np.abs(P[:, 2] - z0_) <= DZ]
        if len(s_) < 3: continue
        for y_ in range(1920, 4992 - size + 1, 150):
            for x_ in range(3328, 6400 - size + 1, 150):
                w = s_[(s_[:, 3] >= y_) & (s_[:, 3] < y_ + size) & (s_[:, 4] >= x_) & (s_[:, 4] < x_ + size)]
                if len(w) < 3: continue
                ids, cnt = np.unique(w[:, 0], return_counts=True); k = int((cnt >= 3).sum())
                if k > best[0]: best = (k, (z0_, y_, x_))
    z0, y0, x0 = best[1]; print("slice", z0, y0, x0, "ladders with >= 3 points in the band", best[0])
ct = np.load("D:/vesuvius_big_hot/ct_big.npy", mmap_mode="r"); img = np.asarray(ct[z0 - 10496, y0 - 1920:y0 - 1920 + size, x0 - 3328:x0 - 3328 + size]).astype(np.float32)
fig, ax = plt.subplots(figsize=(9, 9), dpi=130); ax.imshow(img, cmap="gray", extent=(x0, x0 + size, y0 + size, y0), vmin=np.percentile(img, 1), vmax=np.percentile(img, 99.5), interpolation="nearest")
s = P[(np.abs(P[:, 2] - z0) <= DZ) & (P[:, 3] >= y0) & (P[:, 3] < y0 + size) & (P[:, 4] >= x0) & (P[:, 4] < x0 + size)]
cm = plt.get_cmap("tab20"); n = 0
for cid in np.unique(s[:, 0]):
    q = s[s[:, 0] == cid]; q = q[np.argsort(q[:, 1])]
    if len(q) < 2: continue
    col = cm(int(cid) % 20); ax.plot(q[:, 4], q[:, 3], "-", color=col, lw=0.9, alpha=0.9); ax.scatter(q[:, 4], q[:, 3], s=14, color=col, edgecolor="white", linewidth=0.3, zorder=3); n += 1
    for r in q: ax.text(r[4] + 3, r[3] - 3, f"{int(r[1])}", color="yellow", fontsize=5.5, zorder=4)
ax.set_title(f"automatic ladders on CT, L2 z {z0} (+-{DZ} layers), {n} ladders; labels = wind number within the ladder", fontsize=8); ax.set_xlabel("x (L2 voxels)"); ax.set_ylabel("y (L2 voxels)")
fig.tight_layout(); fig.savefig(out); print("saved", out, n, "ladders drawn")
