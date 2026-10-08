"""Second reference for the automatic ladders: certified links scored against the turn-number field (isosurf turn_numbers) instead of the psi label field.
Chains are built exactly as in autoladder2_eval.py (first accepted link among i+1..i+GAP with 1 <= count <= 4 and probability >= tau).  A link is right when the truth turn difference equals the certified count (both end points with a finite turn number).
Also reports chains of >= 3 points with every link right, and the share of truth sheets covered under this reference is not computed (the turn numbers do not give end point offsets).
  python autoladder3_eval_turn.py STATS.pkl [TARGET]"""
import sys, pickle
import numpy as np
d = pickle.load(open(sys.argv[1], "rb")); S = d["stats"]; GAP = d["gap"]; tgt = float(sys.argv[2]) if len(sys.argv) > 2 else 0.99; tau = d["taus"][tgt]


def wilson(k, n, zz=1.96):
    p = k / n; dn = 1 + zz * zz / n; c = p + zz * zz / (2 * n); h = zz * np.sqrt(p * (1 - p) / n + zz * zz / (4 * n * n)); return ((c - h) / dn * 100, (c + h) / dn * 100)


nl = ok = 0; byc = {c: [0, 0] for c in (1, 2, 3, 4)}; nch = 0; nchok = 0; wrong_dir = {"over": 0, "under": 0}
for s in S:
    nk = s["n_kept"]
    if nk < 2: continue
    cur = [0]; links = []
    while True:
        i = cur[-1]; nxt = None
        for j in range(i + 1, min(nk, i + 1 + GAP)):
            c, p = s["table"][(i, j)]
            if 1 <= c <= 4 and p >= tau: nxt = (j, c); break
        if nxt is None:
            if len(cur) >= 3 and all(np.isfinite(s["tn"][q]) for q in cur):
                nch += 1; good = True
                for (a, b, c) in links: good &= (abs(s["tn"][b] - s["tn"][a]) == c)
                nchok += good
            j = i + 1
            if j >= nk: break
            cur = [j]; links = []
        else:
            j, c = nxt; cur.append(j); links.append((i, j, c))
            if np.isfinite(s["tn"][i]) and np.isfinite(s["tn"][j]):
                r = abs(s["tn"][j] - s["tn"][i]) == c; nl += 1; ok += r; byc[c][0] += 1; byc[c][1] += r
                if not r: wrong_dir["over" if c > abs(s["tn"][j] - s["tn"][i]) else "under"] += 1
lo, hi = wilson(ok, nl)
print(f"{sys.argv[1]} (tau {tau:.4f}): links with finite turn numbers {nl}, count equals the turn-number difference {ok / nl * 100:.2f}% ({lo:.1f}-{hi:.1f}); by count: " + ", ".join(f"c={c}: {byc[c][1]}/{byc[c][0]}" for c in byc if byc[c][0]) + f"; wrong: overcount {wrong_dir['over']}, undercount {wrong_dir['under']}; chains >= 3 points {nch}, every link right {nchok / max(nch, 1) * 100:.1f}%")
