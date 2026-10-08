"""Ladder-side tables of the ONE learned certifier (certlib.py features, trained on the human ladders with the training z range and the regions touching the truth box removed; 5 folds by collection).
Out-of-fold predictions, threshold set on them for the stated overall precision over pairs of 1 to 4 windings (precision is the target, coverage is the measurement), per pair type, one-wrap pairs by z band,
95% intervals by a bootstrap over collections (the pairs of one collection share points), and the same table with the grad_mag slopes fitted inside the folds (slopes of the training folds only).
  python ladder_tables.py [NBOOT]"""
import sys
import numpy as np
import certlib as C

NB = int(sys.argv[1]) if len(sys.argv) > 1 else 500
f2 = ["E:/vesuvius_ladder2_rows_0.npz", "E:/vesuvius_ladder2_rows_1.npz"]; f3 = ["E:/vesuvius_ladder3_rows_0.npz", "E:/vesuvius_ladder3_rows_1.npz"]
lad = C.load_ladder(f2, f3); truth = lad["truth"]; fold = lad["fold"]; ri = lad["ri"]; z = lad["z"]; typ = (truth >= 1) & (truth <= 4)
print(f"ladder pairs {len(truth)}; pairs of 1-4 windings {int(typ.sum())} (types {[int((truth == k).sum()) for k in (1, 2, 3, 4)]}); collections {len(np.unique(ri))}")


def oof(slope_in_fold):
    p = np.zeros(len(truth)); cand_all = np.zeros(len(truth), int)
    if not slope_in_fold:
        s1, s2 = C.gm_slopes(lad); X, cand, _ = C.features(lad, s1, s2); y = (cand == truth).astype(int)
        for f in range(5):
            tr = (fold != f) & typ; te = fold == f; p[te] = C.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        return p, cand
    for f in range(5):
        trm = (fold != f) & typ; sub = {k: (v[trm] if hasattr(v, "__len__") and len(v) == len(truth) else v) for k, v in lad.items()}
        s1, s2 = C.gm_slopes(sub); X, cand, _ = C.features(lad, s1, s2); y = (cand == truth).astype(int); te = fold == f
        p[te] = C.fit(X[trm], y[trm]).predict_proba(X[te])[:, 1]; cand_all[te] = cand[te]
    return p, cand_all


def tau_for(p, cand, target):
    y = (cand == truth).astype(int); order = np.argsort(-p[typ]); cum = np.cumsum(y[typ][order]) / np.arange(1, typ.sum() + 1); ps = p[typ][order]; ii = np.nonzero(cum >= target)[0]
    return float(ps[ii.max()]) if len(ii) else None


def stats(p, cand, tau, mask_coll=None):
    y = (cand == truth).astype(int); cert = typ & (p >= tau)
    if mask_coll is not None: pass
    out = {"cov": cert.sum() / typ.sum(), "prec": y[cert].mean()}
    for t in (1, 2, 3, 4):
        m = truth == t; c = m & (p >= tau); out[f"cov{t}"] = c.sum() / m.sum(); out[f"prec{t}"] = y[c].mean() if c.any() else np.nan; out[f"under{t}"] = int(((cand[c] < t)).sum())
    return out


for name, sif in (("slopes fitted on all ladder pairs (as used by the generator and the truth-pair check)", False), ("slopes fitted inside the folds", True)):
    p, cand = oof(sif); y = (cand == truth).astype(int); print(f"\n=== {name}")
    for target in (0.98, 0.99, 0.995):
        tau = tau_for(p, cand, target); s = stats(p, cand, tau)
        print(f"overall precision target {target * 100:.1f}% (tau {tau:.4f}): coverage {s['cov'] * 100:.1f}%, precision {s['prec'] * 100:.2f}%; " + "; ".join(f"{t}-wrap cov {s[f'cov{t}'] * 100:.1f}% right {s[f'prec{t}'] * 100:.1f}% (certified with a missed sheet {s[f'under{t}']})" for t in (1, 2, 3, 4)))
        if sif and target == 0.99:
            rng = np.random.default_rng(0); colls = np.unique(ri); idx_by = {c: np.nonzero(ri == c)[0] for c in colls}; boots = []
            for b in range(NB):
                pick = rng.choice(colls, len(colls), replace=True); ii = np.concatenate([idx_by[c] for c in pick]); tt = typ[ii]; cert = tt & (p[ii] >= tau)
                row = [cert.sum() / tt.sum(), y[ii][cert].mean()]
                for t in (1, 2, 3, 4):
                    m = truth[ii] == t; c = m & (p[ii] >= tau); row += [c.sum() / max(m.sum(), 1), y[ii][c].mean() if c.any() else np.nan]
                boots.append(row)
            B = np.array(boots); lo = np.nanpercentile(B, 2.5, 0) * 100; hi = np.nanpercentile(B, 97.5, 0) * 100
            print(f"   bootstrap over collections ({NB} resamples), 95% interval: overall coverage {lo[0]:.1f}-{hi[0]:.1f}, precision {lo[1]:.1f}-{hi[1]:.1f}; " + "; ".join(f"{t}-wrap cov {lo[2 * t]:.1f}-{hi[2 * t]:.1f} right {lo[2 * t + 1]:.1f}-{hi[2 * t + 1]:.1f}" for t in (1, 2, 3, 4)))
            print("   one-wrap pairs by L2 z band at this threshold (coverage, precision of the certified):")
            for lo_z, hi_z in ((0, 8000), (8000, 10000), (8400, 9400), (10000, 12000), (11000, 12000), (12000, 14000), (14000, 16000), (16000, 99999)):
                m = (truth == 1) & (z >= lo_z) & (z < hi_z); c = m & (p >= tau)
                if m.sum() >= 20:
                    cl = np.unique(ri[m]); bb = []
                    for b in range(NB):
                        pick = rng.choice(cl, len(cl), replace=True); ii = np.concatenate([np.nonzero(m & (ri == c_))[0] for c_ in pick]); cc = c[ii]; bb.append((cc.mean(), y[ii][cc].mean() if cc.any() else np.nan))
                    bb = np.array(bb); print(f"     z {lo_z}-{hi_z}: n={int(m.sum())} ({len(cl)} collections), coverage {c.sum() / m.sum() * 100:.0f}%, precision {y[c].mean() * 100:.1f}%; bootstrap precision {np.nanpercentile(bb[:, 1], 2.5) * 100:.1f}-{np.nanpercentile(bb[:, 1], 97.5) * 100:.1f}")
            print(f"   pairs of 5 or more windings certified with a count of 1 to 4 (same threshold): {int(((truth >= 5) & (p >= tau) & (cand >= 1) & (cand <= 4)).sum())} of {int((truth >= 5).sum())}; certified pairs of 1-4 windings {int((typ & (p >= tau)).sum())}")
