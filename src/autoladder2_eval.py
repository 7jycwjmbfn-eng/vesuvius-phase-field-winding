"""Score autoladder2.py output.  Chains are built offline from the stored pair table:
from chain end i try j = i+1 .. i+GAP in order; accept the first link with certifier probability >= tau and candidate count 1..4, wind(j) = wind(i) + c; otherwise the chain ends.
Link correct: both end points within TOL turns of a truth sheet centre and the truth sheet count equals c.
Reports for tau at ladder-side overall precision 98 / 99 / 99.5%, TOL 0.15 (primary) and 0.30:
  accepted links, link precision (by count c), chains of >= 3 points, share of chains with every link right, sheet recall
Baseline without certification: chain through every kept point (c = phase count of the neighbouring crossings = 1 per step when no crossing was dropped).
  python autoladder2_eval.py STATS.pkl [STATS2.pkl ...]"""
import sys, pickle
import numpy as np

S = []; meta = None
for f in sys.argv[1:]:
    d = pickle.load(open(f, "rb")); S += d["stats"]; meta = d
GAP = meta["gap"]; print(f"{len(S)} rays, PCONF {meta['pconf']}, GAP {GAP}, z index range {meta['zrange']}; kept points {sum(s['n_kept'] for s in S)} of {sum(s['n_all'] for s in S)} crossings; truth sheets {sum(s['n_truth'] for s in S)}")


def wilson(k, n, zz=1.96):
    if n == 0: return (float("nan"), float("nan"))
    p = k / n; d = 1 + zz * zz / n; c = p + zz * zz / (2 * n); h = zz * np.sqrt(p * (1 - p) / n + zz * zz / (4 * n * n)); return ((c - h) / d * 100, (c + h) / d * 100)


def build(s, tau):
    chains = []; i = 0; nk = s["n_kept"]; cur = [0]; links = []
    if nk == 0: return chains
    while True:
        i = cur[-1]; nxt = None
        for j in range(i + 1, min(nk, i + 1 + GAP)):
            cand, p = s["table"][(i, j)]
            if 1 <= cand <= 4 and p >= tau: nxt = (j, cand, p); break
        if nxt is None:
            chains.append((cur, links)); j = i + 1
            if j >= nk: break
            cur = [j]; links = []
        else: cur.append(nxt[0]); links.append((i, nxt[0], nxt[1], nxt[2]))
    return chains


for tgt, tau in sorted(S and meta["taus"].items()):
    print(f"\n=== certifier threshold tau {tau:.4f} (ladder-side overall precision {tgt * 100:.1f}%) ===")
    for tol in (0.15, 0.30):
        nl = 0; ok = 0; byc = {c: [0, 0] for c in (1, 2, 3, 4)}; wrong_kinds = {"count": 0, "off-sheet": 0}; chain_len = []; chain_ok = []; covered = 0
        for s in S:
            inl = set()
            for cur, links in build(s, tau):
                good = True
                for (i, j, c, p) in links:
                    onboth = s["off"][i] <= tol and s["off"][j] <= tol; cnt_ok = (s["near"][j] - s["near"][i]) == c; right = onboth and cnt_ok
                    nl += 1; ok += right; byc[c][0] += 1; byc[c][1] += right; good &= right
                    if not right: wrong_kinds["off-sheet" if not onboth else "count"] += 1
                if len(cur) >= 3:
                    chain_len.append(len(cur)); chain_ok.append(good)
                    for q in cur:
                        if s["off"][q] <= tol: inl.add(int(s["near"][q]))
            covered += len(inl)
        lo, hi = wilson(ok, nl); tot_sheets = sum(s["n_truth"] for s in S)
        print(f"  TOL {tol}: accepted links {nl}, correct {ok / max(nl, 1) * 100:.2f}% ({lo:.1f}-{hi:.1f}); wrong: count {wrong_kinds['count']}, endpoint off the sheet {wrong_kinds['off-sheet']}; by count c: " + ", ".join(f"c={c}: {byc[c][1]}/{byc[c][0]}" for c in byc if byc[c][0]))
        cl = np.array(chain_len); co = np.array(chain_ok)
        if len(cl): print(f"           chains >= 3 points: {len(cl)}, mean {cl.mean():.1f} points, every link right {co.mean() * 100:.1f}%; truth sheets in certified chains {covered}/{tot_sheets} = {covered / tot_sheets * 100:.1f}%")
# baseline: every kept point chained, count from the phase of the neighbours (here the certified count is not used; we ask whether consecutive kept points are on sheets and one apart)
for tol in (0.15, 0.30):
    n = 0; ok = 0
    for s in S:
        for i in range(s["n_kept"] - 1):
            n += 1; ok += (s["off"][i] <= tol and s["off"][i + 1] <= tol and s["near"][i + 1] - s["near"][i] == 1)
    print(f"\nbaseline (no certification, consecutive kept crossings, count taken as 1), TOL {tol}: links {n}, correct {ok / max(n, 1) * 100:.1f}%")
