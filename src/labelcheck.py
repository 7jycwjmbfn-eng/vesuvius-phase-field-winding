"""Label check of a global unwrapped field u against the truth on the held-out automatic ladders (autoladder2.py output).
For every ladder of >= 3 points (certified at tau of the pickle, truth end points within 0.3 turn) and for every single link: truth sheet count, certified count, |round(u(b)) - round(u(a))| of the field.
  python labelcheck.py STATS.pkl U1.npy [U2.npy ...]"""
import sys, pickle
import numpy as np
ORG = np.array([10496, 1920, 3328.]); d = pickle.load(open(sys.argv[1], "rb")); S = d["stats"]; tau = d["taus"][0.99]; GAP = d["gap"]


def build(s, tau):
    chains = []; nk = s["n_kept"]
    if nk == 0: return chains
    cur = [0]; cs = []
    while True:
        i = cur[-1]; nxt = None
        for j in range(i + 1, min(nk, i + 1 + GAP)):
            cand, p = s["table"][(i, j)]
            if 1 <= cand <= 4 and p >= tau: nxt = (j, cand); break
        if nxt is None:
            chains.append((cur, cs)); j = i + 1
            if j >= nk: break
            cur = [j]; cs = []
        else: cur.append(nxt[0]); cs.append(nxt[1])
    return chains


def wilson(k, n, zz=1.96):
    p = k / n; dn = 1 + zz * zz / n; c = p + zz * zz / (2 * n); h = zz * np.sqrt(p * (1 - p) / n + zz * zz / (4 * n * n)); return ((c - h) / dn * 100, (c + h) / dn * 100)


for up in sys.argv[2:]:
    U = np.load(up, mmap_mode="r")
    def ulab(pt):
        g = np.clip(np.rint((pt - ORG - 0.5) / 2.0).astype(int), 0, np.array(U.shape) - 1); return round(float(U[tuple(g)]))
    rows = []
    for s in S:
        for cur, cs in build(s, tau):
            if len(cur) < 3: continue
            a, b = cur[0], cur[-1]
            if s["off"][a] <= 0.3 and s["off"][b] <= 0.3: rows.append((abs(int(s["near"][b] - s["near"][a])), sum(cs), abs(ulab(s["pts"][b]) - ulab(s["pts"][a])), len(cur)))
            for q in range(len(cur) - 1):
                i, j = cur[q], cur[q + 1]
                if s["off"][i] <= 0.3 and s["off"][j] <= 0.3: rows.append((abs(int(s["near"][j] - s["near"][i])), cs[q], abs(ulab(s["pts"][j]) - ulab(s["pts"][i])), -1))
    R = np.array(rows); print(f"== {up}")
    for name, X in (("whole ladders", R[R[:, 3] > 0]), ("single links", R[R[:, 3] < 0])):
        tr, ce, lb = X[:, 0], X[:, 1], X[:, 2]; lo, hi = wilson(int((lb == tr).sum()), len(X))
        print(f"  {name}: n={len(X)}  certified count right {np.mean(ce == tr) * 100:.2f}%   u-label difference right {np.mean(lb == tr) * 100:.2f}% ({lo:.1f}-{hi:.1f})   cert right & label wrong {np.mean((ce == tr) & (lb != tr)) * 100:.2f}%   label right & cert wrong {np.mean((ce != tr) & (lb == tr)) * 100:.2f}%")
