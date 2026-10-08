"""Disputed links of the RBU grid: gap >= G L2 voxels, turn-number step 2, certified count 1.  Count rising edges of the surface prediction (sp > 127) along the straight segment between the two crossings (as certlib BoxA.votes sp_s) and compare with agreeing links."""
import pickle, numpy as np, sys
sys.path.insert(0, "."); import certlib as C
d = pickle.load(open("E:/vesuvius_rbu_sync_out.pkl", "rb")); G = d["Gidx"]; S = d["Sid"]; T = d["T"]; Pts = d["Pts"]; p = d["p"]
W = G - p[S]; same = S[1:] == S[:-1]; dw = (W[1:] - W[:-1])[same]; dT = (T[1:] - T[:-1])[same]; fin = np.isfinite(dT)
A = Pts[:-1][same]; B = Pts[1:][same]; r = np.linalg.norm(B - A, axis=1)
spv = np.load(C.SPCT + "sp_big.npy", mmap_mode="r"); sh = np.array(spv.shape) - 1; org = np.asarray(C.ORG_B, float); print("sp shape", spv.shape, "ORG_B", org)
def sp_s(a, b):
    L = np.linalg.norm(b - a); n = int(np.ceil(L * 2)) + 1; line = a[None] + (b - a)[None] * np.linspace(0, 1, n)[:, None]
    idx = np.rint((line - org) * 1.0).astype(int)
    ok = np.all((idx >= 0) & (idx <= sh), 1)
    if not ok.all(): return -1
    spl = np.asarray(spv[tuple(idx.T)]) > 127; return int((spl[1:] & ~spl[:-1]).sum())
def run(m, name):
    ii = np.nonzero(m)[0]; vals = np.array([sp_s(A[i], B[i]) for i in ii]); okk = vals >= 0
    print(f"{name}: links {len(ii)}, inside sp box {int(okk.sum())}; sp_s histogram {dict(zip(*np.unique(vals[okk], return_counts=True)))}")
    return ii[okk], vals[okk]
for g in (28, 32):
    base = fin & (dT == 2) & (dw == 1) & (r >= g)
    ii, v = run(base, f"disputed gap>={g} (T step 2, cert 1)")
    n1 = int((v == 1).sum()); n2 = int((v == 2).sum()); print(f"  sp_s==1 (cert) {n1}, sp_s==2 (turn numbers) {n2}, other {len(v) - n1 - n2}; cert share of 1-or-2: {n1 / max(n1 + n2, 1) * 100:.1f}%")
for g in (28, 32):
    for k in (1, 2):
        m = fin & (np.abs(dT) == k) & (dw == k) & (r >= g); rng = np.random.default_rng(0); idx = np.nonzero(m)[0]
        if len(idx) > 3000: idx = rng.choice(idx, 3000, replace=False); mm = np.zeros(len(m), bool); mm[idx] = True; m = mm
        ii, v = run(m, f"control agree k={k} gap>={g}")
        print(f"  sp_s==k share {np.mean(v == k) * 100:.1f}%")
