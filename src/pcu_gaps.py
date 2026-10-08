"""Where the PCU report says it cannot do things, and what our certifier does on the same pairs (matched as in compare_pcu.py).
  python pcu_gaps.py PCU_ROWS.json OURS.npz PRED_LAD0.npz PRED_LAD1.npz"""
import os, sys, json, math
import numpy as np
from collections import defaultdict
P = json.load(open(sys.argv[1])); O = np.load(sys.argv[2])
z3 = np.array([r["z3"] for r in P]); step = np.array([r["step"] for r in P]); Lp = np.array([r["L"] for r in P]); loc = np.array([r["loc"] for r in P]); den = np.array([r["den"] for r in P]); glob = np.array([r["glob"] for r in P])
rl, rd = np.round(loc), np.round(den); pcu_cert = (rl == rd) & (rl == glob) & (np.abs(loc - rl) < 0.25) & (np.abs(den - rd) < 0.30); pcu_cnt = rl.astype(int)
z = O["z"]; L = O["L"]; k = O["truth"].astype(int); idx = defaultdict(list)
for i in range(len(z)): idx[(int(round(z[i] / 2)), int(k[i]))].append(i)
used = set(); mp = np.full(len(P), -1)
for j in range(len(P)):
    c = [i for i in idx.get((int(z3[j]), int(step[j])), []) if i not in used]
    if not c: continue
    b = min(c, key=lambda i: abs(L[i] / 2 - Lp[j]))
    if abs(L[b] / 2 - Lp[j]) < 1e-3: mp[j] = b; used.add(b)
j = np.nonzero(mp >= 0)[0]; oi = mp[j]; s = step[j]; pc = pcu_cert[j]; pn = pcu_cnt[j]; pr = pn == s; Lj = Lp[j]; zj = z3[j]
cs = O["cert_stack_990"][oi]; cn_ = O["c_n"][oi]; rn = cn_ == s; cg = O["c_g"][oi]; rg = cg == s
def wl(kk, n, zz=1.96):
    if n == 0: return float("nan")
    p = kk / n; c = p + zz * zz / (2 * n); r = zz * math.sqrt(p * (1 - p) / n + zz * zz / (4 * n * n)); return (c - r) / (1 + zz * zz / n)
def line(name, cert, right, mask):
    n = int(mask.sum()); c = int((cert & mask).sum()); e = int(((~right) & cert & mask).sum())
    return f"{name:22s} n {n:4d} certified {c:4d} ({c / max(n, 1) * 100:5.1f}%) precision {(1 - e / c) * 100 if c else float('nan'):6.2f}% ({e} err)"
print("=== 1. harder band (PCU: fit gain absent; local count 92.0% vs 98.2%, certified 51% vs 78%): z2 11000-12000 = z3 5500-6000; band 1 z3 4200-4700")
for lo, hi, nm in ((4200, 4700, "band 1 (z2 8400-9400)"), (5500, 6000, "band 2 (z2 11000-12000)")):
    mk0 = (zj >= lo) & (zj < hi)
    for sv in (1, 2, 3):
        mk = mk0 & (s == sv); print(f"{nm} step {sv}:"); print("   ", line("PCU", pc, pr, mk)); print("   ", line("stack 99% target", cs, rn, mk)); print("   ", line("GBM 99% target", O['cert_GBM_990'][oi], rg, mk))
print("\n=== 2. dense regions (short pairs = small wrap spacing), step 1; PCU table: <96 um 33% certified, 96-134 70%")
for lo, hi, nm in ((0, 5, "<96 um"), (5, 7, "96-134 um"), (7, 9, "134-173 um"), (9, 12, "173-230 um"), (12, 1e9, ">230 um")):
    mk = (s == 1) & (Lj >= lo) & (Lj < hi); print(f"{nm:11s}", line("PCU", pc, pr, mk)[:75], "|", line("stack", cs, rn, mk)[23:])
print("\n=== 3. pairs where PCU certifies a wrong count (all three Lasagna measurements agree on a wrong integer): what do we say?")
bad = pc & ~pr
for sv in (1, 2, 3):
    mk = bad & (s == sv); nb = int(mk.sum())
    if nb: print(f"step {sv}: PCU wrong-certified {nb}; counter right {int((rn & mk).sum())}, GBM right {int((rg & mk).sum())}; stack certifies {int((cs & mk).sum())} of them (right {int((cs & mk & rn).sum())}); PCU said {sorted(set(pn[mk].tolist()))}")
print("\n=== 4. the error that hurts a fit most: a multi-wrap pair certified with fewer wraps (PCU: 1 of 636 two-wrap pairs certified as one on Paris 4; 5-8% on PHerc0139)")
for sv in (2, 3):
    mk = s == sv
    print(f"step {sv}: PCU certified {int((pc & mk).sum())}, with count < true {int((pc & mk & (pn < s)).sum())}; stack certified {int((cs & mk).sum())}, count < true {int((cs & mk & (cn_ < s)).sum())}; counter count==1 among stack-certified: {int((cs & mk & (cn_ == 1)).sum())}")
print("\n=== 5. long spans (PCU: certifies one wrap at a time; precision falls with distance; crest graph 5-9 apart 88.8%, 20+ apart 54%): counter network on pairs of 4-8 wraps")
d = [np.load(f) for f in sys.argv[3:5]]; cc = np.concatenate([x["c"] for x in d]).astype(int); pp = np.concatenate([x["p"] for x in d]); kk = np.concatenate([x["k"] for x in d]).astype(int); ri = np.concatenate([x["ri"] for x in d]).astype(int); sq = np.concatenate([x["seq"] for x in d]).astype(int)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import certlib as C
f2 = ["E:/vesuvius_ladder2_rows_0.npz", "E:/vesuvius_ladder2_rows_1.npz"]; f3 = ["E:/vesuvius_ladder3_rows_0.npz", "E:/vesuvius_ladder3_rows_1.npz"]
lad = C.load_ladder(f2, f3); key = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(lad["ri"], lad["seq"]))}; fo = np.zeros(len(cc), int); ok = np.zeros(len(cc), bool)
for q, (a, b) in enumerate(zip(ri, sq)):
    i = key.get((int(a), int(b)))
    if i is not None and lad["truth"][i] == kk[q]: fo[q] = lad["fold"][i]; ok[q] = True
def tau_for(c, p, t, mask, target):
    sel = np.nonzero(mask)[0]; o = sel[np.argsort(-p[sel])]; right = (c[o] == t[o]); cum = np.cumsum(right) / np.arange(1, len(o) + 1); g = np.nonzero((cum >= target) & (np.arange(1, len(o) + 1) >= 20))[0]
    return float(p[o][g.max()]) if len(g) else np.inf
for lo, hi in ((4, 4), (5, 6), (7, 8), (5, 8)):
    for target in (0.99, 0.98):
        m = ok & (kk >= lo) & (kk <= hi); cert = np.zeros(len(cc), bool)
        for f in range(5):
            tf = tau_for(cc, pp, kk, m & (fo != f), target); cert |= m & (fo == f) & (pp >= tf)
        n = int(m.sum()); c = int(cert.sum()); e = int((cert & (cc != kk)).sum()); lowc = int((cert & (cc < kk)).sum())
        print(f"true wraps {lo}-{hi}, target {target * 100:.0f}% (threshold from the other folds, same wrap range): pairs {n}, certified {c} ({c / max(n, 1) * 100:.1f}%), precision {(1 - e / max(c, 1)) * 100:.2f}% ({e} wrong, {lowc} with fewer wraps than true)")
