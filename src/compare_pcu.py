"""Matched-pair comparison with the community winding certificate (PCU, Jashann/vesuvius-scrolling, results/e9b_rows.json).
PCU rows: z3 (L3 slice), step (true winding difference 1-3), loc/den/glob measurements, L (pair length in L3 pixels).  Rule (exp/e9_report.py): round(loc) == round(den) == glob, |loc - round| < 0.25, |den - round| < 0.30; the certified count is round(loc).
Our rows: results/pairs_oof.npz (out-of-fold probabilities of the winding-count network, the gradient-boosting certifier and the combined certifier, thresholds from the other four folds; written by stack_ladder_dump.py); pairs are matched by (round(z/2) == z3, truth == step, |L/2 - L_pcu| < 1e-3).

  python src/compare_pcu.py [PCU_ROWS.json] [OURS.npz]

With no arguments the PCU rows are downloaded from the PCU repository (the file is not redistributed here) into ./pcu_e9b_rows.json and our rows are read from results/pairs_oof.npz. Needs only numpy."""
import sys, os, json, math, urllib.request
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PCU_URL = "https://raw.githubusercontent.com/Jashann/vesuvius-scrolling/main/results/e9b_rows.json"
pcu_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.getcwd(), "pcu_e9b_rows.json")
ours_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "results", "pairs_oof.npz")
if not os.path.exists(pcu_path):
    print("downloading the PCU per-pair rows from", PCU_URL, flush=True); urllib.request.urlretrieve(PCU_URL, pcu_path)
P = json.load(open(pcu_path)); O = np.load(ours_path)
z3 = np.array([r["z3"] for r in P]); step = np.array([r["step"] for r in P]); Lp = np.array([r["L"] for r in P]); loc = np.array([r["loc"] for r in P]); den = np.array([r["den"] for r in P]); glob = np.array([r["glob"] for r in P])
rl, rd = np.round(loc), np.round(den); pcu_cert = (rl == rd) & (rl == glob) & (np.abs(loc - rl) < 0.25) & (np.abs(den - rd) < 0.30); pcu_cnt = rl.astype(int)
z = O["z"]; L = O["L"]; k = O["truth"].astype(int)
from collections import defaultdict
idx = defaultdict(list)
for i in range(len(z)): idx[(int(round(z[i] / 2)), int(k[i]))].append(i)
used = set(); mp = np.full(len(P), -1)
for j in range(len(P)):
    c = [i for i in idx.get((int(z3[j]), int(step[j])), []) if i not in used]
    if not c: continue
    b = min(c, key=lambda i: abs(L[i] / 2 - Lp[j]))
    if abs(L[b] / 2 - Lp[j]) < 1e-3: mp[j] = b; used.add(b)
m = mp >= 0; print(f"PCU pairs {len(P)}; matched to our pairs {int(m.sum())} ({m.mean() * 100:.1f}%); by step: " + ", ".join(f"{s}: {int(m[step == s].sum())}/{int((step == s).sum())}" for s in (1, 2, 3)))
print("unmatched PCU pairs per step include ladders excluded from our set (training z 9984-10500, regions touching the truth box) and pairs longer than 255 voxels; PCU certified share of matched vs unmatched:", f"{pcu_cert[m].mean() * 100:.1f}% / {pcu_cert[~m].mean() * 100:.1f}%")
j = np.nonzero(m)[0]; oi = mp[j]; s = step[j]; pc = pcu_cert[j]; pn = pcu_cnt[j]; pr = pn == s
def wl(kk, n, zz=1.96):
    if n == 0: return float("nan")
    p = kk / n; c = p + zz * zz / (2 * n); r = zz * math.sqrt(p * (1 - p) / n + zz * zz / (4 * n * n)); return (c - r) / (1 + zz * zz / n)
def row(name, cert, right, mask):
    n = int(mask.sum()); c = int((cert & mask).sum()); e = int(((~right) & cert & mask).sum())
    return f"{name:34s} n {n:5d} certified {c:5d} ({c / max(n, 1) * 100:5.1f}%) precision {(1 - e / c) * 100 if c else float('nan'):6.2f}% ({e} errors; Wilson low {wl(c - e, c) * 100:.1f}%)"
print("\n=== PCU rule re-derived on the matched pairs (full-set values: 70.0% / 99.62%, 48% / 98.58%, 36% / 94.89%)")
for sv in (1, 2, 3): print(" ", row(f"PCU, step {sv}", pc, pr, s == sv))
rules = {"combined certifier (99% target)": ("cert_stack_990", O["c_n"]), "combined certifier (98% target)": ("cert_stack_980", O["c_n"]), "combined certifier (99.5% target)": ("cert_stack_995", O["c_n"]), "GBM certifier (99% target)": ("cert_GBM_990", O["c_g"]), "counter network alone (98%)": ("cert_counter_980", O["c_n"])}
print("\n=== our rules on the same pairs (thresholds from the other folds, common to steps 1-4)")
for nm, (key, cc) in rules.items():
    cert = O[key][oi]; right = (cc[oi] == s)
    for sv in (1, 2, 3): print(" ", row(f"{nm}, step {sv}", cert, right, s == sv))
cert = O["cert_stack_990"][oi]; c_o = O["c_n"][oi]; ro = c_o == s
print("\n=== overlap of the PCU certificate and the combined certifier (99% target)")
for sv in (1, 2, 3):
    mk = s == sv; a = pc & mk; b = cert & mk
    both = a & b; onlyp = a & ~b; onlyo = b & ~a; neither = mk & ~a & ~b
    print(f"  step {sv}: both {int(both.sum())} (both right {int((both & pr & ro).sum())}, PCU wrong {int((both & ~pr).sum())}, combined certifier wrong {int((both & ~ro).sum())}, counts differ {int((both & (pn != c_o)).sum())});"
          f" only PCU {int(onlyp.sum())} (right {int((onlyp & pr).sum())}); only combined certifier {int(onlyo.sum())} (right {int((onlyo & ro).sum())}); neither {int(neither.sum())}")
print("\n=== combined rules (step-wise)")
for sv in (1, 2, 3):
    mk = s == sv
    union = (pc | cert) & mk; ucnt = np.where(pc, pn, c_o); ur = ucnt == s
    inter = pc & cert & (pn == c_o) & mk
    print("  ", row(f"union (PCU count first), step {sv}", pc | cert, ur, mk)); print("  ", row(f"agreement (both certify, same count), step {sv}", pc & cert & (pn == c_o), pr, mk))
print("\n=== precision of our rules at PCU's coverage per step (ranking by out-of-fold probability within the step, in-sample cut)")
for nm, key in (("combined certifier", "p_s"), ("GBM", "p_g"), ("counter", "p_n")):
    for sv in (1, 2, 3):
        mk = s == sv; n_pcu = int((pc & mk).sum()); idxs = np.nonzero(mk)[0]; pv = O[key][oi][idxs]; o = np.argsort(-pv)[:n_pcu]; cc = (O["c_g"] if nm == "GBM" else O["c_n"])[oi][idxs][o]
        e = int((cc != s[idxs][o]).sum()); print(f"  {nm:8s} step {sv}: top {n_pcu} of {int(mk.sum())} ({n_pcu / mk.sum() * 100:.1f}%): precision {(1 - e / n_pcu) * 100:.2f}% ({e} errors)  [PCU: {(1 - int(((~pr) & pc & mk).sum()) / n_pcu) * 100:.2f}%]")

print("\n=== bootstrap over collections (ladders), 2000 draws: combined certifier (99% target) minus PCU")
rr = O["ri"][oi]; colls = np.unique(rr); groups = {u: np.nonzero(rr == u)[0] for u in colls}; rng = np.random.default_rng(0); res = {sv: [] for sv in (1, 2, 3)}
for b in range(2000):
    pick = rng.choice(colls, len(colls), replace=True); ii = np.concatenate([groups[u] for u in pick])
    for sv in (1, 2, 3):
        mk = ii[s[ii] == sv]; ca, cb = pc[mk], cert[mk]; ea = (~pr[mk]) & ca; eb = (~ro[mk]) & cb
        res[sv].append((cb.mean() - ca.mean(), (1 - eb.sum() / max(cb.sum(), 1)) - (1 - ea.sum() / max(ca.sum(), 1))))
for sv in (1, 2, 3):
    a = np.array(res[sv]); print(f"  step {sv}: coverage difference {np.mean(a[:, 0]) * 100:+.1f} points (95% {np.percentile(a[:, 0], 2.5) * 100:+.1f} to {np.percentile(a[:, 0], 97.5) * 100:+.1f}); precision difference {np.mean(a[:, 1]) * 100:+.2f} points (95% {np.percentile(a[:, 1], 2.5) * 100:+.2f} to {np.percentile(a[:, 1], 97.5) * 100:+.2f})")
