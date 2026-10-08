import json, numpy as np, sys
sys.path.insert(0, "."); import train_wfield_8c as tw
lab = np.load(tw.LAB, mmap_mode="r"); ORG = np.array(tw.ORG, float); print("label", lab.shape, "ORG", ORG)
d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "E:/vesuvius_ap3_boxA_merged.json"))["collections"]
sh = np.array(lab.shape) - 1; rows = []
for cid, c in d.items():
    ids = sorted(c["points"], key=int); P = np.array([c["points"][i]["p"] for i in ids])[:, [2, 1, 0]]; W = np.array([c["points"][i]["wind_a"] for i in ids])
    for q in range(len(ids) - 1):
        a, b = P[q], P[q + 1]; n = int(np.ceil(np.linalg.norm(b - a))) + 1; line = a + (b - a) * np.linspace(0, 1, n)[:, None]
        idx = np.rint(line - np.array([9984.0, ORG[1], ORG[2]])).astype(int)
        if np.any(idx < 0) or np.any(idx > sh): continue
        v = np.asarray(lab[tuple(idx.T)]).astype(np.float64)
        if np.any(v == 255): continue
        u = np.unwrap(v * 2 * np.pi / 255) / (2 * np.pi); dl = u[-1] - u[0]; fa = abs(((v[0] / 255) + 0.5) % 1 - 0.5); fb = abs(((v[-1] / 255) + 0.5) % 1 - 0.5)
        rows.append((abs(dl), W[q + 1] - W[q], fa, fb))
R = np.array(rows); print("scored links", len(R))
cnt = np.rint(R[:, 0]) == R[:, 1]
print(f"count equals rounded label difference (no end restriction): {cnt.mean() * 100:.2f}%")
for tol in (0.15, 0.30):
    ends = (R[:, 2] <= tol) & (R[:, 3] <= tol); print(f"TOL {tol}: both ends near centre {ends.mean() * 100:.1f}%, correct among those (count right) {np.mean(cnt[ends]) * 100:.2f}%, overall count right and ends near {np.mean(cnt & ends) * 100:.1f}%")
for k in (1, 2, 3, 4): m = R[:, 1] == k; print(" step", k, "n", int(m.sum()), "count right %.2f%%" % (cnt[m].mean() * 100 if m.any() else float("nan")))
