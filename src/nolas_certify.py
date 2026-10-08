"""Winding-count certifier WITHOUT lasagna votes (no grad_mag, no cos), for PHerc0139.  Features: nolas.features_nolas.
Trained on the human ladders (certlib.load_ladder: training z range and the 19 regions touching the truth box dropped), 1 to 4 wrap pairs only, one threshold.
Threshold = the point where the ladder leave-fold-out (fold = region % 5) overall precision falls to 98 / 99 / 99.5 %.  The final model (all ladder folds) is then applied to the truth pairs
(truth_pairs.py rows) and reported per k: coverage, precision (Wilson 95%), undercounts (certified cand < truth), multi-wrap pairs certified as one wrap.
  python nolas_certify.py PAIRS1.npz [PAIRS2.npz ...]       (run in a2; env REF=0 skips the with-lasagna reference run on the ladders)"""
import os, sys
import numpy as np
sys.path.insert(0, "../wfield")
import certlib as C
from nolas import features_nolas

F2 = ["E:/vesuvius_ladder2_rows_0.npz", "E:/vesuvius_ladder2_rows_1.npz"]; F3 = ["E:/vesuvius_ladder3_rows_0.npz", "E:/vesuvius_ladder3_rows_1.npz"]
TARGETS = (0.98, 0.99, 0.995)
VOTE_NAMES = ["v6", "frozen", "+3", "-3", "path2", "path4"]


def wilson(k, n, zz=1.96):
    if n == 0: return (float("nan"), float("nan"))
    p = k / n; d = 1 + zz * zz / n; c = p + zz * zz / (2 * n); h = zz * np.sqrt(p * (1 - p) / n + zz * zz / (4 * n * n)); return ((c - h) / d * 100, (c + h) / d * 100)


def pct(a, b): return a / b * 100 if b else float("nan")


def table(name, p, tau, cand, truth, indent="  "):
    typ = (truth >= 1) & (truth <= 4); y = (cand == truth).astype(int); cert = typ & (p >= tau); lo, hi = wilson(int(y[cert].sum()), int(cert.sum()))
    print(f"{indent}[{name}] threshold {tau:.4f}: certified {cert.sum()} of {typ.sum()} = {pct(cert.sum(), typ.sum()):.1f}%, precision {y[cert].mean() * 100 if cert.any() else float('nan'):.2f}% ({lo:.1f}-{hi:.1f}), "
          f"undercounts {int((cand[cert] < truth[cert]).sum())}, multi-wrap certified as one wrap {int(((cand[cert] == 1) & (truth[cert] > 1)).sum())}")
    for t in (1, 2, 3, 4):
        mm = truth == t; c = mm & (p >= tau)
        if mm.sum():
            lo, hi = wilson(int(y[c].sum()), int(c.sum()))
            print(f"{indent}    k={t}: n={int(mm.sum())}, coverage {pct(c.sum(), mm.sum()):5.1f}% ({int(c.sum())}), precision {y[c].mean() * 100 if c.any() else float('nan'):5.1f}% ({lo:.1f}-{hi:.1f}), "
                  f"undercounts {int((cand[c] < t).sum())}, as one wrap {int(((cand[c] == 1) & (t > 1)).sum())}")


def threshold(p, y, typ, target):
    order = np.argsort(-p[typ]); cum = np.cumsum(y[typ][order]) / np.arange(1, typ.sum() + 1); ps = p[typ][order]; ii = np.nonzero(cum >= target)[0]
    return float(ps[ii.max()]) if len(ii) else None


def oof(X, y, typ, fold):
    p = np.zeros(len(y))
    for f in range(5):
        tr = (fold != f) & typ; te = fold == f; p[te] = C.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p


def mode_line(name, cand, truth):
    s = f"  [{name}] phase-mode vote alone: "
    for t in (1, 2, 3, 4):
        mm = truth == t
        if mm.sum(): lo, hi = wilson(int((cand[mm] == t).sum()), int(mm.sum())); s += f"k={t} {np.mean(cand[mm] == t) * 100:.1f}% ({lo:.1f}-{hi:.1f}); "
    typ = (truth >= 1) & (truth <= 4); lo, hi = wilson(int((cand[typ] == truth[typ]).sum()), int(typ.sum())); print(s + f"all 1-4 {np.mean(cand[typ] == truth[typ]) * 100:.1f}% ({lo:.1f}-{hi:.1f})")


if len(sys.argv) < 2: sys.exit("usage: python nolas_certify.py PAIRS1.npz [PAIRS2.npz ...]")
files = sys.argv[1:]
lad = C.load_ladder(F2, F3); Xl, cl = features_nolas(lad); yl = (cl == lad["truth"]).astype(int); typl = (lad["truth"] >= 1) & (lad["truth"] <= 4)
print(f"ladder pairs {len(yl)}, 1-4 wraps {int(typl.sum())} (k=1..4: {[int((lad['truth'] == k).sum()) for k in (1, 2, 3, 4)]}); features {Xl.shape[1]}")
rows = [np.load(f)["rows"] for f in files]; T = np.concatenate([r for r in rows if r.size]); tru = C.rows_to_dict(T); Xt, ct_ = features_nolas(tru); yt = (ct_ == tru["truth"]).astype(int); typt = (tru["truth"] >= 1) & (tru["truth"] <= 4)
src = np.concatenate([np.full(len(r), i) for i, r in enumerate(rows) if r.size])
print(f"truth pairs {len(T)} from {len(files)} files, 1-4 wraps {int(typt.sum())} (k=1..4: {[int((tru['truth'] == k).sum()) for k in (1, 2, 3, 4)]}); other k: {int((~typt).sum())}")
for k in (1, 2, 3, 4):
    mm = tru["truth"] == k
    if mm.sum(): print(f"truth pairs k={k}: single votes " + ", ".join(f"{n} {np.mean(np.rint(tru['V'][mm, i]) == k) * 100:.1f}%" for i, n in enumerate(VOTE_NAMES)) + f", sp1 {np.mean(tru['sp1'][mm] == k) * 100:.1f}%, sp2 {np.mean(tru['sp2'][mm] == k) * 100:.1f}%, ct {np.mean(tru['ct'][mm] == k) * 100:.1f}%")
print("\nphase-mode vote (the candidate count) alone")
mode_line("ladders", cl, lad["truth"]); mode_line("truth pairs", ct_, tru["truth"])

pl = oof(Xl, yl, typl, lad["fold"]); final = C.fit(Xl[typl], yl[typl]); pt = final.predict_proba(Xt)[:, 1]
print("\nladder leave-fold-out (no lasagna votes), threshold at overall precision target; then the same threshold on the truth pairs (final model trained on all ladder folds)")
taus = {}
for target in TARGETS:
    tau = threshold(pl, yl, typl, target)
    if tau is None: print(f"target {target * 100:.1f}%: not reached on the ladders"); continue
    taus[target] = tau; print(f"\n== ladder overall precision >= {target * 100:.1f}%")
    table("ladder OOF", pl, tau, cl, lad["truth"]); table("TRUTH pairs, all files", pt, tau, ct_, tru["truth"])
    if len(files) > 1:
        for i, f in enumerate(files):
            m = src == i
            if m.any(): table(f"TRUTH pairs, {os.path.basename(f)}", pt[m], tau, ct_[m], tru["truth"][m], indent="    ")

if os.environ.get("REF", "1") != "0":
    try:
        s1, s2 = C.gm_slopes(lad); Xr, cr, _ = C.features(lad, s1, s2); yr = (cr == lad["truth"]).astype(int); pr = oof(Xr, yr, typl, lad["fold"])
        print("\nreference: the with-lasagna certifier (certlib features) on the same ladder rows and folds")
        for target in TARGETS:
            tau = threshold(pr, yr, typl, target)
            if tau is not None: table("ladder OOF, with lasagna", pr, tau, cr, lad["truth"])
    except Exception as e: print(f"\nreference run skipped: {type(e).__name__}: {e}")
