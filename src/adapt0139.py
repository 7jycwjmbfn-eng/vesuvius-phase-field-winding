"""Adaptation experiments of the no-lasagna winding-count certifier on PHerc0139 (exploration; the released certifier and its thresholds are not changed).
Question: can the 0139 failure (99% tier: precision 92.27% instead of 99%, all errors are overcounts) be repaired with a small amount of 0139 data, or do the votes carry a systematic bias?
Data (existing files only): E:/vesuvius_p0139_truthpairs_A.npz (6000 truth pairs, seed 300 of wfield/truth_pairs.py), the human ladders (certlib.load_ladder), the W2 truth pairs of Paris 4.
Protocol (see nolas_certify.py for the unchanged reference): label y = (cand == truth), cand = phase-mode count (nolas.features_nolas), certified = score >= tau, 1-4 wrap pairs.
  Blocks: the 1024 z layers (column z, 4352-5376) are cut into 4 blocks of 256 layers.  Test block j; training pairs = 0139 pairs at least GAP = 64 layers away from block j.
  Threshold tau: chosen from training data only.  Inner leave-one-block-out over the remaining 3 blocks (training pairs at least 64 layers from both the inner and the outer test block) gives out-of-block scores;
  tau = largest score at which the cumulative precision of these scores is >= target (same rule as nolas_certify.threshold).  Test blocks are then pooled (every pair is tested once, with the tau of its own block).
  'oracle' = same rule with tau taken from the pooled test scores themselves (upper bound of what any threshold choice can give for these scores; it is not a result that could be obtained without the test labels).
Sections (python adapt0139.py [base] [a] [b] [c] [d] [e] [f] [g] [all]):
  base  reproduce the unchanged certifier on the 6000 pairs (ladder-side threshold) and per z block
  a     calibration only (ladder model fixed; Platt / isotonic / Platt with cand and length), threshold-only recalibration
  b     ladders + 0139 training blocks, 0139 weight 1 / 5 / 20
  c     0139 training blocks only (gradient boosting; logistic regression as a second model)
  d     feature ablation (no log1p(L); no confidences minc, meanc, pmin) for (b) weight 5, also the ladder-only model and (c)
  e     structure of the overcounts: length per wrap, single votes, spatial concentration (z, y-x, radius), confidence, independent image votes
  f     learning curve over the number of 0139 training pairs (ranking metrics, oracle threshold)
  g     robustness: the same suite (without d) with 4 blocks by azimuth and 4 blocks by radius around the centre line (pairs within 64 voxels in the y-x plane of the test block are not used for training); add the argument full to include d
Writes E:/vesuvius_inv_tmp/adapt0139/results.json (results_partial.json when not all sections are run).  Needs the endpoint cache (made once from G:/vesuvius_77f5/p0139/psi_box.npy, ~1 min) for e and g."""
import os, sys, json, time, itertools
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "wfield"))
import certlib as C
from nolas import features_nolas, NAMES as FNAMES
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from scipy import stats

F2 = ["E:/vesuvius_ladder2_rows_0.npz", "E:/vesuvius_ladder2_rows_1.npz"]; F3 = ["E:/vesuvius_ladder3_rows_0.npz", "E:/vesuvius_ladder3_rows_1.npz"]
P139 = "E:/vesuvius_p0139_truthpairs_A.npz"; W2 = ["E:/vesuvius_truthpairs_A.npz", "E:/vesuvius_truthpairs_B.npz"]
OUT = "E:/vesuvius_inv_tmp/adapt0139/"; CACHE = OUT + "p0139_endpoints_seed300.npz"; SEED_PAIRS = 300
TARGETS = (0.99, 0.98); KS = (1, 2, 3, 4); GAP = 64; ZLO = 4352; ZW = 256; NB = 4
ALL = list(range(18)); NO_L = [i for i in ALL if i != 16]; NO_CONF = [i for i in ALL if i not in (13, 14, 15)]; NO_BOTH = [i for i in ALL if i not in (13, 14, 15, 16)]
RES = {}


# ---------------------------------------------------------------- basic helpers
def wilson(k, n, zz=1.96):
    if n == 0: return (float("nan"), float("nan"))
    p = k / n; d = 1 + zz * zz / n; c = p + zz * zz / (2 * n); h = zz * np.sqrt(p * (1 - p) / n + zz * zz / (4 * n * n)); return (max((c - h) / d * 100, 0.0), min((c + h) / d * 100, 100.0))


def threshold(p, y, target):
    """Largest score at which the cumulative precision (scores sorted downward) is >= target; None if never (same rule as nolas_certify)."""
    order = np.argsort(-p, kind="stable"); cum = np.cumsum(y[order]) / np.arange(1, len(p) + 1); ii = np.nonzero(cum >= target)[0]
    return float(p[order][ii.max()]) if len(ii) else None


def fit_w(X, y, w=None):
    """Same model as certlib.fit (identical hyper-parameters); early stopping off so that it never switches on above 10000 rows; optional sample weights."""
    return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, max_leaf_nodes=15, l2_regularization=1.0, random_state=0, early_stopping=False).fit(X, y, sample_weight=w)


def logit(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))


def evaluate(cert, cand, truth):
    """cert: boolean per pair.  Returns dict 'all' and k=1..4: n, ncert, cov %, prec %, Wilson lo/hi, undercounts, overcounts among certified."""
    y = cand == truth; out = {}
    for key, mask in (("all", np.ones(len(truth), bool)),) + tuple((k, truth == k) for k in KS):
        c = mask & cert; n = int(mask.sum()); nc = int(c.sum()); ok = int(y[c].sum()); lo, hi = wilson(ok, nc)
        out[key] = dict(n=n, ncert=nc, cov=100 * nc / n if n else float("nan"), prec=100 * ok / nc if nc else float("nan"), lo=lo, hi=hi, under=int((cand[c] < truth[c]).sum()), over=int((cand[c] > truth[c]).sum()))
    return out


def safe_auc(y, p): return roc_auc_score(y, p) if len(y) and y.min() != y.max() else float("nan")


def mix_eval(cert, cand, truth, pi):
    """Coverage and precision if the pair population had the k-mix pi (dict k -> share) instead of the sampler's k-mix: pairs are weighted by pi[k] / (share of k in the 6000 pairs)."""
    n = len(truth); w = np.zeros(n)
    for k in KS: m = truth == k; w[m] = pi[k] / (m.sum() / n)
    c = cert; return 100 * w[c].sum() / w.sum(), (100 * (w * (cand == truth))[c].sum() / w[c].sum() if w[c].sum() else float("nan"))


def fmt_cell(r): return f"{r['cov']:.1f} / {r['prec']:.1f} ({r['lo']:.1f}-{r['hi']:.1f})" if r["ncert"] else f"{r['cov']:.1f} / -"


# ---------------------------------------------------------------- data
class Data:
    def __init__(self):
        lad = C.load_ladder(F2, F3); typl = (lad["truth"] >= 1) & (lad["truth"] <= 4)
        self.lad = {k: (v[typl] if hasattr(v, "__len__") and len(v) == len(typl) else v) for k, v in lad.items()}
        self.Xl, self.cl = features_nolas(self.lad); self.tl = self.lad["truth"]; self.yl = (self.cl == self.tl).astype(int); self.fold = self.lad["fold"]
        self.T = np.load(P139)["rows"]; self.D = C.rows_to_dict(self.T); self.X, self.cand = features_nolas(self.D); self.truth = self.D["truth"]; self.y = (self.cand == self.truth).astype(int); self.z = self.T[:, 17]; self.n = len(self.y)
        self.W2 = C.rows_to_dict(np.concatenate([np.load(f)["rows"] for f in W2]))
        self.pi_ladder = {k: float(np.mean(self.tl == k)) for k in KS}

    def ladder_model(self, cols):
        return fit_w(self.Xl[:, cols], self.yl)

    def ladder_oof(self, cols):
        p = np.zeros(len(self.yl))
        for f in range(5):
            tr = self.fold != f; p[self.fold == f] = fit_w(self.Xl[tr][:, cols], self.yl[tr]).predict_proba(self.Xl[self.fold == f][:, cols])[:, 1]
        return p


# ---------------------------------------------------------------- blocks
class Blocks:
    def __init__(self, ids, dist, name, labels):
        self.ids = ids; self.dist = dist; self.name = name; self.labels = labels; self.nb = len(dist)


def z_blocks(z):
    ids = np.clip(((z - ZLO) // ZW).astype(int), 0, NB - 1); dist = []
    for j in range(NB):
        lo, hi = ZLO + ZW * j, ZLO + ZW * (j + 1); dist.append(np.maximum(np.maximum(lo - z, z - hi), 0.0))
    return Blocks(ids, dist, "z blocks", [f"z {ZLO + ZW * j}-{ZLO + ZW * (j + 1)}" for j in range(NB)])


def geo_blocks(key, ym, xm, name, unit):
    """4 blocks of equal pair count along `key` (azimuth or radius of the pair midpoint around the centre line).  dist[j] = distance in the y-x plane from each pair midpoint to the nearest midpoint of block j (0 inside)."""
    from scipy.spatial import cKDTree
    edges = np.quantile(key, [0.25, 0.5, 0.75]); ids = np.digitize(key, edges); pts = np.c_[ym, xm]; dist = []; labels = []
    for j in range(4):
        m = ids == j; dist.append(cKDTree(pts[m]).query(pts)[0]); lo = key[m].min(); hi = key[m].max(); labels.append(f"{name} {lo:.0f}-{hi:.0f}{unit}")
    return Blocks(ids, dist, f"{name} blocks", labels)


# ---------------------------------------------------------------- endpoints of the pairs (needed for y, x, radius)
def recover_endpoints(T):
    """Re-run only the endpoint sampling of wfield/truth_pairs.py (BOX=P139, seed 300, RAYLEN 140, KMAX 4).  rng use does not depend on the votes, so the same pairs come out;
    k, z and L are checked against T.  Returns array (n, 8): k, za, ya, xa, zb, yb, xb, radius of the start point (distance to the centre line)."""
    if os.path.exists(CACHE):
        E = np.load(CACHE)["E"]
        if len(E) == len(T) and np.array_equal(E[:, 0], T[:, 0]) and np.abs(E[:, 1] - T[:, 17]).max() < 1e-9: return E
    t0 = time.time(); rng = np.random.default_rng(SEED_PAIRS); ORG_T = np.array([4352, 1664, 1280.]); zlo, zhi = 4352, 5376
    lab = np.load("G:/vesuvius_77f5/p0139/psi_box.npy", mmap_mode="r"); cj = json.load(open("G:/vesuvius_77f5/p0139/centre_w023.json")); oc = np.argsort(cj["z"]); uz, uy, ux = (np.array(cj[t], float)[oc] for t in ("z", "y", "x"))
    L = lab[0:zhi - zlo]; RAYLEN = 140.0; KMAX = 4; rows = []; tries = 0; N = len(T)
    while len(rows) < N and tries < 40 * N:
        tries += 1
        z = int(rng.integers(0, L.shape[0])); y = int(rng.integers(2 + 40, L.shape[1] - 2 - 40)); x = int(rng.integers(2 + 40, L.shape[2] - 2 - 40))
        if L[z, y, x] == 255: continue
        cy, cx = np.interp(zlo + z, uz, uy), np.interp(zlo + z, uz, ux); gy = ORG_T[1] + y - cy; gx = ORG_T[2] + x - cx; r = np.hypot(gy, gx)
        if r < 300: continue
        th = rng.normal(0, np.radians(25)); d = np.array([gy / r * np.cos(th) - gx / r * np.sin(th), gy / r * np.sin(th) + gx / r * np.cos(th)]); steps = np.arange(0, RAYLEN, 0.5); ys = y + d[0] * steps; xs = x + d[1] * steps
        dz = rng.uniform(-0.35, 0.35); zs = z + dz * steps
        inside = (ys >= 0) & (ys <= L.shape[1] - 1) & (xs >= 0) & (xs <= L.shape[2] - 1) & (zs >= 0) & (zs <= L.shape[0] - 1); n_in = int(inside.argmin()) if not inside.all() else len(steps)
        if n_in < 100: continue
        ys = ys[:n_in]; xs = xs[:n_in]; zs = zs[:n_in]; lv = L[np.rint(zs).astype(int), np.rint(ys).astype(int), np.rint(xs).astype(int)]
        bad = np.nonzero(lv == 255)[0]; n_ok = int(bad[0]) if len(bad) else len(lv)
        if n_ok < 60: continue
        lv = lv[:n_ok]; ys = ys[:n_ok]; xs = xs[:n_ok]; zs = zs[:n_ok]
        u = np.unwrap(lv.astype(np.float64) * 2 * np.pi / 255) / (2 * np.pi); sgn = 1.0 if u[-1] >= u[0] else -1.0; u = u * sgn
        if np.any(np.diff(u) < -0.12): continue
        ms = np.arange(np.ceil(u[0]), np.floor(u[-1]) + 1)
        if len(ms) < 2: continue
        pos = np.array([np.interp(m, np.maximum.accumulate(u), np.arange(len(u))) for m in ms]); i0 = int(rng.integers(0, len(ms) - 1)); k = int(rng.integers(1, min(KMAX, len(ms) - 1 - i0) + 1)); i1 = i0 + k
        f = lambda q: (np.interp(q, np.arange(len(ys)), ys), np.interp(q, np.arange(len(xs)), xs)); ya, xa = f(pos[i0]); yb, xb = f(pos[i1]); za = np.interp(pos[i0], np.arange(len(zs)), zs); zb = np.interp(pos[i1], np.arange(len(zs)), zs)
        rows.append((k, zlo + za, ya + ORG_T[1], xa + ORG_T[2], zlo + zb, yb + ORG_T[1], xb + ORG_T[2], r))
    E = np.array(rows); Lg = np.linalg.norm(E[:, [1, 2, 3]] - E[:, [4, 5, 6]], axis=1)
    ok = len(E) == len(T) and np.array_equal(E[:, 0], T[:, 0]) and np.abs(E[:, 1] - T[:, 17]).max() < 1e-9 and np.abs(Lg - T[:, 16]).max() < 1e-9
    print(f"endpoint recovery: {len(E)} pairs, {tries} tries, {time.time() - t0:.0f} s, k / z / L identical to the stored rows: {ok}", flush=True)
    if not ok: raise RuntimeError("endpoint recovery does not reproduce the stored pairs")
    np.savez(CACHE, E=E); return E


def centre_radius(E):
    cj = json.load(open("G:/vesuvius_77f5/p0139/centre_w023.json")); oc = np.argsort(cj["z"]); uz, uy, ux = (np.array(cj[t], float)[oc] for t in ("z", "y", "x"))
    zm = (E[:, 1] + E[:, 4]) / 2; ym = (E[:, 2] + E[:, 5]) / 2; xm = (E[:, 3] + E[:, 6]) / 2; cy = np.interp(zm, uz, uy); cx = np.interp(zm, uz, ux)
    return ym, xm, np.hypot(ym - cy, xm - cx), np.degrees(np.arctan2(ym - cy, xm - cx))


# ---------------------------------------------------------------- the protocol
class Result:
    """Pooled out-of-block scores p, per-block thresholds, certified flags, evaluations."""
    def __init__(self, name, p, taus, blocks, d):
        self.name = name; self.p = p; self.taus = taus; self.blocks = blocks; self.ev = {}; self.orc = {}; self.cert = {}
        for t in TARGETS:
            tb = np.array([np.nan if taus[t][j] is None else taus[t][j] for j in range(blocks.nb)]); tau_pair = tb[blocks.ids]; cert = np.isfinite(tau_pair) & (p >= np.nan_to_num(tau_pair, nan=np.inf))
            self.cert[t] = cert; self.ev[t] = evaluate(cert, d.cand, d.truth)
            to = threshold(p, d.y, t); co = (p >= to) if to is not None else np.zeros(len(p), bool); self.orc[t] = evaluate(co, d.cand, d.truth)
        self.mix = {t: mix_eval(self.cert[t], d.cand, d.truth, d.pi_ladder) for t in TARGETS}
        self.auc = roc_auc_score(d.y, p); self.ll = log_loss(d.y, np.clip(p, 1e-6, 1 - 1e-6)); self.brier = brier_score_loss(d.y, p)
        m2 = d.cand >= 2; to2 = threshold(p[m2], d.y[m2], 0.99); self.orc2 = 100 * np.mean(p[m2] >= to2) if to2 is not None else 0.0; self.auc_k = {k: safe_auc(d.y[d.truth == k], p[d.truth == k]) for k in KS}; self.auc_c = {c: safe_auc(d.y[d.cand == c], p[d.cand == c]) for c in KS}


def protocol(name, score_fn, blocks, d, inner=True, sub=None, rep=0):
    """score_fn(train_mask, eval_mask) -> scores of the eval pairs from a model fitted on the 0139 training pairs (and, depending on the method, the ladders).
    sub = n: only a random subset of n of the training pairs of every test block is available (for the thresholds as well)."""
    p = np.full(d.n, np.nan); taus = {t: [None] * blocks.nb for t in TARGETS}
    for j in range(blocks.nb):
        test = blocks.ids == j; trn = blocks.dist[j] >= GAP
        if sub is not None and sub < trn.sum():
            keep = np.random.default_rng(1000 * rep + 10 * j + 1).choice(np.nonzero(trn)[0], sub, replace=False); trn = np.zeros(d.n, bool); trn[keep] = True
        p[test] = score_fn(trn, test); pi = []; yi = []
        if inner:
            for m in range(blocks.nb):
                if m == j: continue
                ev = (blocks.ids == m) & (blocks.dist[j] >= GAP) & (trn if sub is not None else True); tr2 = trn & (blocks.dist[m] >= GAP)
                if ev.sum() and tr2.sum(): pi.append(score_fn(tr2, ev)); yi.append(d.y[ev])
            pi = np.concatenate(pi); yi = np.concatenate(yi)
            for t in TARGETS: taus[t][j] = threshold(pi, yi, t)
    return Result(name, p, taus, blocks, d)


def ladder_tau_result(name, p, d, ptau_lad, blocks):
    """Unchanged protocol: tau from the ladder out-of-fold scores (ptau_lad), score model trained on ladders only."""
    taus = {t: [threshold(ptau_lad[0], ptau_lad[1], t)] * blocks.nb for t in TARGETS}; return Result(name, p, taus, blocks, d)


def print_overall(results, t, title):
    print(f"\n{title} | target {t * 100:.0f}% overall precision")
    print("| method | certified | coverage % | precision % (Wilson 95%) | over / under | oracle-tau coverage % (precision) | coverage / precision at the ladder k-mix | AUC | log loss |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        a = r.ev[t]["all"]; o = r.orc[t]["all"]
        print(f"| {r.name} | {a['ncert']} | {a['cov']:.1f} | {a['prec']:.2f} ({a['lo']:.1f}-{a['hi']:.1f}) | {a['over']} / {a['under']} | {o['cov']:.1f} ({o['prec']:.1f}) | {r.mix[t][0]:.1f} / {r.mix[t][1]:.1f} | {r.auc:.3f} | {r.ll:.3f} |")


def print_perk(results, t, title, which="ev"):
    print(f"\n{title} | target {t * 100:.0f}% | per k: coverage % / precision % (Wilson 95%)" + (" | ORACLE tau" if which == "orc" else ""))
    print("| method | k=1 | k=2 | k=3 | k=4 |"); print("|---|---|---|---|---|")
    for r in results:
        e = getattr(r, which)[t]; print(f"| {r.name} | " + " | ".join(fmt_cell(e[k]) for k in KS) + " |")


def print_auc(results, title):
    print(f"\n{title} | AUC of the pooled out-of-block score for right vs wrong cand, within truth k and within cand value (0.5 = no information)")
    print("| method | k=1 | k=2 | k=3 | k=4 | cand=1 | cand=2 | cand=3 | cand=4 | oracle share of cand >= 2 pairs certifiable at 99% |"); print("|---|---|---|---|---|---|---|---|---|---|")
    for r in results: print(f"| {r.name} | " + " | ".join(f"{r.auc_k[k]:.3f}" for k in KS) + " | " + " | ".join(f"{r.auc_c[c]:.3f}" for c in KS) + f" | {r.orc2:.1f}% |")


def store(results, tag):
    for r in results:
        RES.setdefault(tag, {})[r.name] = dict(auc=r.auc, auc_k=r.auc_k, auc_c=r.auc_c, logloss=r.ll, brier=r.brier, taus={str(t): r.taus[t] for t in TARGETS}, ev={str(t): {str(k): v for k, v in r.ev[t].items()} for t in TARGETS},
                                               oracle={str(t): {str(k): v for k, v in r.orc[t].items()} for t in TARGETS})


def per_block(results, t, blocks, d):
    print(f"\nper test block ({blocks.name}), target {t * 100:.0f}%: coverage % / precision % (n certified), tau in brackets")
    print("| method | " + " | ".join(f"{blocks.labels[j]} (n={int((blocks.ids == j).sum())})" for j in range(blocks.nb)) + " |"); print("|---|" + "---|" * blocks.nb)
    for r in results:
        cells = []
        for j in range(blocks.nb):
            m = blocks.ids == j; c = r.cert[t] & m; ok = int(d.y[c].sum()); tau = r.taus[t][j]
            cells.append(f"{100 * c.sum() / m.sum():.1f} / {100 * ok / c.sum() if c.sum() else float('nan'):.1f} ({int(c.sum())}) [{'-' if tau is None else format(tau, '.3f')}]")
        print(f"| {r.name} | " + " | ".join(cells) + " |")


# ---------------------------------------------------------------- method builders
def zero_shot(d, cols, name, blocks, with_ladder_tau=True):
    """Ladder-only model (fixed), score on the 0139 pairs.  Returns (result with the ladder-side tau, result with tau chosen from the 0139 training blocks)."""
    m = d.ladder_model(cols); p0 = m.predict_proba(d.X[:, cols])[:, 1]; out = []
    if with_ladder_tau: out.append(ladder_tau_result(name + " [tau from ladders]", p0, d, (d.ladder_oof(cols), d.yl), blocks))
    out.append(protocol(name + " [tau from 0139 train blocks]", lambda tr, ev: p0[ev], blocks, d)); return out, p0


def calib_platt(p0, d, extra=False):
    def f(tr, ev):
        if extra: x = np.c_[logit(p0), d.cand, np.log1p(d.D["L"])]
        else: x = logit(p0)[:, None]
        mdl = make_pipeline(StandardScaler(), LogisticRegression(C=1e4, max_iter=2000)).fit(x[tr], d.y[tr]); return mdl.predict_proba(x[ev])[:, 1]
    return f


def calib_iso(p0, d):
    def f(tr, ev): return IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(p0[tr], d.y[tr]).predict(p0[ev])
    return f


def retrain(d, cols, w):
    Xl = d.Xl[:, cols]
    def f(tr, ev):
        X = np.concatenate([Xl, d.X[tr][:, cols]]); y = np.concatenate([d.yl, d.y[tr]]); ww = np.concatenate([np.ones(len(d.yl)), np.full(int(tr.sum()), float(w))]); return fit_w(X, y, ww).predict_proba(d.X[ev][:, cols])[:, 1]
    return f


def only139(d, cols, kind="hgb"):
    def f(tr, ev):
        if kind == "hgb": return fit_w(d.X[tr][:, cols], d.y[tr]).predict_proba(d.X[ev][:, cols])[:, 1]
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)).fit(d.X[tr][:, cols], d.y[tr]).predict_proba(d.X[ev][:, cols])[:, 1]
    return f


# ---------------------------------------------------------------- sections
def sec_suite(d, blocks, which, tag, show_blocks=True):
    results = []; byname = {}
    def add(rs):
        for r in (rs if isinstance(rs, list) else [rs]): results.append(r); byname[r.name] = r
    t0 = time.time()
    if "base" in which or "a" in which or "d" in which:
        zs, p0 = zero_shot(d, ALL, "ladder-only model", blocks); add(zs)
    if "a" in which:
        add(protocol("(a) Platt on logit(p)", calib_platt(p0, d), blocks, d)); add(protocol("(a) isotonic", calib_iso(p0, d), blocks, d)); add(protocol("(a) Platt on logit(p), cand, log1p(L)", calib_platt(p0, d, extra=True), blocks, d))
    if "b" in which:
        for w in (1, 5, 20): add(protocol(f"(b) ladders + 0139, weight {w}", retrain(d, ALL, w), blocks, d))
    if "c" in which:
        add(protocol("(c) 0139 train blocks only, boosting", only139(d, ALL), blocks, d)); add(protocol("(c) 0139 train blocks only, logistic regression", only139(d, ALL, "lr"), blocks, d))
    if "d" in which:
        for nm, cols in (("no log1p(L)", NO_L), ("no confidences", NO_CONF), ("no L, no confidences", NO_BOTH)):
            add(protocol(f"(d) (b) weight 5, {nm}", retrain(d, cols, 5), blocks, d))
            zs2, _ = zero_shot(d, cols, f"(d0) ladder-only, {nm}", blocks); add(zs2)
            add(protocol(f"(d) (c) boosting, {nm}", only139(d, cols), blocks, d))
    print(f"\n#### suite on {blocks.name} ({tag}), {time.time() - t0:.0f} s")
    for t in TARGETS:
        print_overall(results, t, f"[{tag}]"); print_perk(results, t, f"[{tag}]")
    print_auc(results, f"[{tag}]")
    if show_blocks:
        for t in TARGETS: per_block(results, t, blocks, d)
    store(results, tag); return results, byname


def sec_rules(d):
    """Fixed rules without any training.  They were written down after looking at the 0139 pairs, so the numbers are optimistic; they only show what the retrained models amount to."""
    cand = d.cand; t = d.truth; y = d.y; V = np.rint(d.D["V"]); neq = (V == cand[:, None]).sum(1); L = d.D["L"]
    rules = [("cand = 1", cand == 1), ("cand = 1 and >= 5 of 6 votes equal", (cand == 1) & (neq >= 5)), ("cand = 1 and all 6 votes equal", (cand == 1) & (neq >= 6)), ("cand = 1, all 6 votes equal, L < 30", (cand == 1) & (neq >= 6) & (L < 30)),
             ("cand = 2 and all 6 votes equal", (cand == 2) & (neq >= 6)), ("cand = 2, all 6 votes equal, L < 40", (cand == 2) & (neq >= 6) & (L < 40)), ("cand = 3 and all 6 votes equal", (cand == 3) & (neq >= 6)), ("cand = 4 and all 6 votes equal", (cand == 4) & (neq >= 6))]
    print("\n#### fixed rules (no training; chosen after looking at the data, so optimistic): coverage % of all pairs / precision % (Wilson); per k coverage / precision")
    print("| rule | certified | coverage % | precision % (Wilson 95%) | k=1 | k=2 | k=3 | k=4 |"); print("|---|---|---|---|---|---|---|---|")
    for nm, m in rules:
        e = evaluate(m, cand, t); a = e["all"]; print(f"| {nm} | {a['ncert']} | {a['cov']:.1f} | {a['prec']:.2f} ({a['lo']:.1f}-{a['hi']:.1f}) | " + " | ".join(fmt_cell(e[k]) for k in KS) + " |")


def sec_calibration_table(d, blocks):
    """Reliability of the ladder model on 0139 (raw) and after out-of-block Platt / isotonic: realised accuracy in bins of the score."""
    m = d.ladder_model(ALL); p0 = m.predict_proba(d.X)[:, 1]
    pp = np.full(d.n, np.nan); pi = np.full(d.n, np.nan)
    for j in range(blocks.nb):
        te = blocks.ids == j; tr = blocks.dist[j] >= GAP; pp[te] = calib_platt(p0, d)(tr, te); pi[te] = calib_iso(p0, d)(tr, te)
    edges = [(0, 0.5), (0.5, 0.9), (0.9, 0.98), (0.98, 0.99), (0.99, 0.995), (0.995, 1.01)]
    print("\ncalibration on 0139 (all 6000 pairs): bins of score -> realised accuracy % (n)")
    print("| score | raw ladder model | Platt (out-of-block) | isotonic (out-of-block) |"); print("|---|---|---|---|")
    for lo, hi in edges:
        cells = []
        for p in (p0, pp, pi):
            mm = (p >= lo) & (p < hi); cells.append(f"{100 * d.y[mm].mean():.1f} ({int(mm.sum())})" if mm.any() else "-")
        print(f"| [{lo}, {hi if hi < 1.01 else 1.0}) | " + " | ".join(cells) + " |")
    for t in (0.99, 0.98):
        print(f"nominal threshold score >= {t}: raw precision {100 * d.y[p0 >= t].mean():.2f}% (n={int((p0 >= t).sum())}), Platt {100 * d.y[pp >= t].mean() if (pp >= t).any() else float('nan'):.2f}% (n={int((pp >= t).sum())}), isotonic {100 * d.y[pi >= t].mean() if (pi >= t).any() else float('nan'):.2f}% (n={int((pi >= t).sum())})")
    return p0


def quant(a, q=(10, 25, 50, 75, 90)): return " / ".join(f"{v:.1f}" for v in np.percentile(a, q)) if len(a) else "-"


def sec_e(d, base_res, E):
    ym, xm, rad, azi = centre_radius(E); L = d.D["L"]; truth = d.truth; cand = d.cand; p0 = base_res.p; cert = base_res.cert[0.99]; wrong = cert & (cand != truth); right = cert & (cand == truth)
    print(f"\n#### (e) structure of the overcounts, 0139; errors = pairs certified by the unchanged certifier at the 99% ladder-side threshold and wrong: {int(wrong.sum())} of {int(cert.sum())} certified; over {int((wrong & (cand > truth)).sum())}, under {int((wrong & (cand < truth)).sum())}")
    cw0 = rows_cand(d.W2); print("(e0) precision of the count cand = c (share of pairs with cand = c whose truth is c), % (number of pairs)")
    print("| set | c=1 | c=2 | c=3 | c=4 |"); print("|---|---|---|---|---|")
    for nm_, cc_, tt_ in (("ladders", d.cl, d.tl), ("W2", cw0, d.W2["truth"]), ("0139", cand, truth)): print(f"| {nm_} | " + " | ".join(f"{100 * np.mean(tt_[cc_ == c_] == c_):.1f} ({int((cc_ == c_).sum())})" for c_ in KS) + " |")
    print(f"truth of the 0139 pairs with cand = 2 (wrong ones): truth 1 {int(((cand == 2) & (truth == 1)).sum())}, truth 3 {int(((cand == 2) & (truth == 3)).sum())}, truth 4 {int(((cand == 2) & (truth == 4)).sum())}; z span |za - zb| of the pairs: median {np.median(np.abs(E[:, 1] - E[:, 4])):.1f}, 90% {np.percentile(np.abs(E[:, 1] - E[:, 4]), 90):.1f}, max {np.abs(E[:, 1] - E[:, 4]).max():.1f}; radius range {rad.min():.0f}-{rad.max():.0f}, azimuth range {azi.min():.0f} to {azi.max():.0f} deg")
    print("(e1) length per wrap L / k (L2 voxels), quantiles 10/25/50/75/90")
    print("| k | 0139 all pairs | 0139 certified right | 0139 certified wrong: L/k_truth | 0139 certified wrong: L/k_cand | W2 pairs (Paris 4, A+B) | ladders (Paris 4) |"); print("|---|---|---|---|---|---|---|")
    W = d.W2; wt = W["truth"]; lt = d.tl; Ll = d.lad["L"]
    for k in KS:
        a = truth == k; r_ = right & a; w_ = wrong & a
        print(f"| {k} | n={int(a.sum())}: {quant(L[a] / k)} | n={int(r_.sum())}: {quant(L[r_] / k)} | n={int(w_.sum())}: {quant(L[w_] / k)} | {quant(L[w_] / np.maximum(cand[w_], 1))} | n={int((wt == k).sum())}: {quant(W['L'][wt == k] / k)} | n={int((lt == k).sum())}: {quant(Ll[lt == k] / k)} |")
    a = truth == 1; u, pu = stats.mannwhitneyu(L[wrong & a], L[right & a], alternative="greater") if (wrong & a).any() else (np.nan, np.nan)
    print(f"k=1: certified-wrong pairs are longer than certified-right pairs, Mann-Whitney one-sided p = {pu:.2e}; median length {np.median(L[wrong & a]):.1f} vs {np.median(L[right & a]):.1f}")
    # local spacing test: which count makes the length per wrap look normal for this radius
    rb = np.digitize(rad, np.arange(300, 1500, 100)); sp = {}
    for b in np.unique(rb):
        m = (rb == b) & (cand == truth) & (truth == 1); sp[b] = np.median(L[m]) if m.sum() >= 20 else np.nan
    sloc = np.array([sp[b] for b in rb]); ok = np.isfinite(sloc)
    dk = np.abs(np.log(L / truth / sloc)); dc = np.abs(np.log(L / np.maximum(cand, 1) / sloc)); share = lambda m: 100 * np.mean(dc[m & ok] < dk[m & ok]) if (m & ok).any() else float("nan")
    dk1 = np.abs(np.log(L / truth / sloc)); dk2 = np.abs(np.log(L / (truth + 1) / sloc))
    print("(e1b) which count gives a normal length per wrap?  local spacing s = median L of correct k=1 pairs in the same radius bin (100 voxels) of the centre line")
    print(f"  certified-wrong pairs (cand = k+1 mostly): share with |log(L/cand / s)| < |log(L/k / s)|: {share(wrong):.1f}%  (n={int((wrong & ok).sum())})")
    print(f"  baseline, certified-right pairs, counterfactual cand' = k+1: share with |log(L/(k+1) / s)| < |log(L/k / s)|: {100 * np.mean(dk2[right & ok] < dk1[right & ok]):.1f}%  (n={int((right & ok).sum())})")
    print(f"  baseline for k=1 only: {100 * np.mean(dk2[right & ok & (truth == 1)] < dk1[right & ok & (truth == 1)]):.1f}%;  wrong k=1 only: {share(wrong & (truth == 1)):.1f}%")
    # (e2) single votes
    print("\n(e2) single votes (rounded |phase count|) against the truth k: share over / equal / under, %")
    names = ["v6", "frozen", "+3 line", "-3 line", "path2", "path4"]
    sets = (("0139", d.D, truth), ("W2 (Paris 4)", W, wt), ("ladders (Paris 4)", d.lad, lt))
    print("| set | k | " + " | ".join(names) + " |"); print("|---|---|" + "---|" * 6)
    for nm, DD, tt in sets:
        for k in KS:
            m = tt == k; V = np.rint(DD["V"][m]); cells = [f"{100 * np.mean(V[:, i] > k):.1f} / {100 * np.mean(V[:, i] == k):.1f} / {100 * np.mean(V[:, i] < k):.1f}" for i in range(6)]
            print(f"| {nm} | {k} | " + " | ".join(cells) + " |")
    print("mean (|vote| - k) per vote, 0139: " + ", ".join(f"{n} {np.mean(d.D['V'][:, i] - truth):+.3f}" for i, n in enumerate(names)) + "; W2: " + ", ".join(f"{n} {np.mean(W['V'][:, i] - wt):+.3f}" for i, n in enumerate(names)) + "; ladders: " + ", ".join(f"{n} {np.nanmean(d.lad['V'][:, i] - lt):+.3f}" for i, n in enumerate(names)))
    print("overcount rate of the vote-mode count cand (cand > k), 0139 by k: " + ", ".join(f"k={k} {100 * np.mean(cand[truth == k] > k):.1f}%" for k in KS) + "; W2 by k: " + ", ".join(f"k={k} {100 * np.mean(rows_cand(W)[wt == k] > k):.1f}%" for k in KS))
    fit_ = [np.mean(cand[truth == k] > k) for k in KS]; q = 1 - (1 - fit_[0]) ** (1 / 1)
    print("  if each wrap added a spurious sheet independently with probability q (q from k=1): predicted overcount rate for k=1..4: " + ", ".join(f"{100 * (1 - (1 - q) ** k):.1f}%" for k in KS))
    # unanimous votes
    V = np.rint(d.D["V"]); mode_ok = (cand >= 1) & (cand <= 4); un = mode_ok & (V == cand[:, None]).all(1)
    Vw = np.rint(W["V"]); cw = rows_cand(W); unw = (cw >= 1) & (cw <= 4) & (Vw == cw[:, None]).all(1); Vl = np.rint(d.lad["V"]); cl = d.cl; unl = (cl >= 1) & (cl <= 4) & (Vl == cl[:, None]).all(1)
    print(f"all six votes equal (mode in 1-4): share and accuracy: 0139 {100 * un.mean():.1f}% / {100 * d.y[un].mean():.1f}% (n={int(un.sum())});  W2 {100 * unw.mean():.1f}% / {100 * np.mean(cw[unw] == wt[unw]):.1f}%;  ladders {100 * unl.mean():.1f}% / {100 * np.mean(cl[unl] == lt[unl]):.1f}%")
    # (e3) spatial concentration
    print("\n(e3) where are the errors?  overcount rate of cand (cand > k) among all pairs, and certified-wrong count at the unchanged 99% threshold, per bin; chi-square test of homogeneity of the overcount rate (pairs of one scroll are spatially correlated: p-values are optimistic)")
    over = (cand > truth).astype(int)
    def bins_report(name, key, edges):
        b = np.digitize(key, edges); rows = []
        for u in np.unique(b):
            m = b == u
            if m.sum() < 30: continue
            rows.append((edges[u - 1] if u > 0 else float("-inf"), int(m.sum()), 100 * over[m].mean(), int((wrong & m).sum()), int((cert & m).sum())))
        tab = np.array([[r[1] * r[2] / 100, r[1] - r[1] * r[2] / 100] for r in rows]); chi = stats.chi2_contingency(np.round(tab))[:2]
        print(f"  {name}: " + "; ".join(f"{'<' + str(int(edges[0])) if r[0] == float('-inf') else '>=' + str(int(r[0]))} n={r[1]} over {r[2]:.1f}% wrong-cert {r[3]}/{r[4]}" for r in rows) + f"; chi2 {chi[0]:.1f}, p {chi[1]:.2g} ({len(rows)} bins)")
        rr = np.array([r[2] for r in rows]); print(f"    overcount rate range over bins {rr.min():.1f}% to {rr.max():.1f}%, overall {100 * over.mean():.1f}%")
    bins_report("z (64-layer bins)", d.z, np.arange(ZLO + 64, ZLO + 1024, 64)); bins_report("y of pair midpoint (192-voxel bins)", ym, np.arange(1664 + 192, 3200, 192)); bins_report("x of pair midpoint (192-voxel bins)", xm, np.arange(1280 + 192, 2816, 192))
    bins_report("radius from the centre line (100-voxel bins)", rad, np.arange(400, 1500, 100)); bins_report("azimuth around the centre line (45 degree bins)", azi, np.arange(-135, 180, 45))
    # y-x grid
    gy = np.digitize(ym, np.arange(1664 + 384, 3200, 384)); gx = np.digitize(xm, np.arange(1280 + 384, 2816, 384)); print("  y-x 4 x 4 grid (384-voxel cells), overcount % (n):")
    for iy in range(4):
        print("    y %d-%d: " % (1664 + 384 * iy, 1664 + 384 * (iy + 1)) + ", ".join((f"{100 * over[(gy == iy) & (gx == ix)].mean():.0f}% ({int(((gy == iy) & (gx == ix)).sum())})" if ((gy == iy) & (gx == ix)).sum() >= 20 else "-") for ix in range(4)))
    # (e4) confidences
    print("\n(e4) confidences of the pairs: quantiles 10/25/50/75/90 and AUC (right vs wrong certified-at-99% pairs, and right vs wrong among all pairs with cand 1-4)")
    print("| feature | certified right | certified wrong | all cand-right | all cand-wrong | AUC cert | AUC all | ladders: AUC all (no-lasagna cand) |"); print("|---|---|---|---|---|---|---|---|")
    allm = (cand >= 1) & (cand <= 4); mc = d.y == 1; ml = (d.cl >= 1) & (d.cl <= 4)
    for nm, col, arr, la in (("minc (v6 min confidence)", 13, d.D["minc"], d.lad["minc"]), ("meanc", 14, d.D["meanc"], d.lad["meanc"]), ("pmin (path min confidence)", 15, d.D["pmin"], d.lad["pmin"]), ("L (voxels)", 16, L, d.lad["L"]), ("L / cand", 99, L / np.maximum(cand, 1), d.lad["L"] / np.maximum(d.cl, 1))):
        arr = np.nan_to_num(arr, nan=-1.0); la = np.nan_to_num(la, nan=-1.0)
        auc_c = roc_auc_score(d.y[cert], arr[cert]) if wrong.any() else np.nan; auc_a = roc_auc_score(d.y[allm], arr[allm]); auc_l = roc_auc_score((d.cl == d.tl)[ml], la[ml])
        print(f"| {nm} | {quant(arr[right])} | {quant(arr[wrong])} | {quant(arr[allm & mc])} | {quant(arr[allm & ~mc])} | {auc_c:.3f} | {auc_a:.3f} | {auc_l:.3f} |")
    # (e5) independent image votes
    print("\n(e5) independent image votes on pairs where all six phase votes agree: truth k=1, phase count 1 (right) against 2 (overcount); k=2, 2 against 3.  sp = surface-prediction crossings (straight, path), ct = CT peaks")
    print("| set | k | cand | n | L/k median | L/cand median | sp straight mean | sp path mean | ct mean | sp straight = cand % | sp path = cand % | ct = cand % | minc median |"); print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for nm, DD, tt, cc, uu in (("0139", d.D, truth, cand, un), ("W2", W, wt, cw, unw), ("ladders", d.lad, lt, cl, unl)):
        for k in (1, 2):
            for c_ in (k, k + 1):
                m = uu & (tt == k) & (cc == c_)
                if m.sum() < 5: continue
                print(f"| {nm} | {k} | {c_} | {int(m.sum())} | {np.median(DD['L'][m] / k):.1f} | {np.median(DD['L'][m] / c_):.1f} | {DD['sp1'][m].mean():.2f} | {DD['sp2'][m].mean():.2f} | {DD['ct'][m].mean():.2f} | {100 * np.mean(DD['sp1'][m] == c_):.0f} | {100 * np.mean(DD['sp2'][m] == c_):.0f} | {100 * np.mean(DD['ct'][m] == c_):.0f} | {np.median(DD['minc'][m]):.2f} |")
    print("(e5b) the same comparison at matched pair length (truth k=1, all six phase votes equal): surface-prediction crossings and CT peaks for phase count 1 (right) against 2 (overcount)")
    print("| set | L (voxels) | cand=1: n | sp straight | ct | cand=2: n | sp straight | ct |"); print("|---|---|---|---|---|---|---|---|")
    for nm, DD, tt, cc, uu in (("0139", d.D, truth, cand, un), ("W2", W, wt, cw, unw)):
        for lo, hi in ((0, 15), (15, 20), (20, 25), (25, 30), (30, 40), (40, 200)):
            m = uu & (tt == 1) & (DD["L"] >= lo) & (DD["L"] < hi); a_ = m & (cc == 1); b_ = m & (cc == 2)
            f_ = lambda mm, key: f"{DD[key][mm].mean():.2f}" if mm.sum() >= 5 else "-"
            print(f"| {nm} | {lo}-{hi} | {int(a_.sum())} | {f_(a_, 'sp1')} | {f_(a_, 'ct')} | {int(b_.sum())} | {f_(b_, 'sp1')} | {f_(b_, 'ct')} |")
    print("AUC of every feature for right vs wrong among pairs with all six phase votes equal (mode 1-4); 0.5 = no information")
    print("| feature | ladders | W2 | 0139 |"); print("|---|---|---|---|")
    Xs = {"ladders": (d.Xl[unl], (cl == lt)[unl]), "W2": (features_nolas(W)[0][unw], (cw == wt)[unw]), "0139": (d.X[un], d.y[un])}
    for i, n in enumerate(FNAMES):
        cells = []
        for s in ("ladders", "W2", "0139"):
            X_, y_ = Xs[s]; cells.append(f"{roc_auc_score(y_, X_[:, i]):.3f}" if X_[:, i].std() > 0 and y_.min() != y_.max() else "-")
        print(f"| {n} | " + " | ".join(cells) + " |")
    RES["e"] = dict(n_wrong=int(wrong.sum()), n_cert=int(cert.sum()))


_RC = {}


def rows_cand(W):
    if id(W) not in _RC: _RC[id(W)] = features_nolas(W)[1]
    return _RC[id(W)]


def sec_f(d, blocks):
    """Learning curve: number of 0139 training pairs (random subsets of the training blocks of every test block).  Ranking metrics only: AUC and oracle-tau coverage at 99 / 98% on the pooled test blocks."""
    m = d.ladder_model(ALL); p0 = m.predict_proba(d.X)[:, 1]; sizes = [0, 100, 300, 1000, 2000, None]; reps = 3
    print("\n#### (f) learning curve: pooled over the 4 test blocks, mean over 3 random subsets (min-max in brackets); oracle-tau coverage % at 99% / 98% precision, AUC")
    print("| 0139 training pairs | method | oracle cov 99% | oracle cov 98% | AUC | log loss |"); print("|---|---|---|---|---|---|")
    out = {}
    for n_tr in sizes:
        methods = {"(a) Platt": lambda tr, ev: calib_platt(p0, d)(tr, ev), "(a) isotonic": lambda tr, ev: calib_iso(p0, d)(tr, ev), "(b) weight 5": retrain(d, ALL, 5), "(b) weight 20": retrain(d, ALL, 20), "(c) boosting 0139 only": only139(d, ALL)}
        if n_tr == 0:
            r = Result("zero", p0, {t: [None] * 4 for t in TARGETS}, blocks, d); print(f"| 0 | ladder-only model | {r.orc[0.99]['all']['cov']:.1f} | {r.orc[0.98]['all']['cov']:.1f} | {r.auc:.3f} | {r.ll:.3f} |"); out[0] = dict(auc=r.auc); continue
        for mn, fn in methods.items():
            vals = []
            for rep in range(1 if n_tr is None else reps):
                p = np.full(d.n, np.nan); n_used = []
                for j in range(blocks.nb):
                    te = blocks.ids == j; tr = blocks.dist[j] >= GAP
                    if n_tr is not None and n_tr < tr.sum():
                        rng = np.random.default_rng(1000 * rep + 10 * j + 1); idx = np.nonzero(tr)[0]; keep = rng.choice(idx, n_tr, replace=False); tr = np.zeros(d.n, bool); tr[keep] = True
                    n_used.append(int(tr.sum())); p[te] = fn(tr, te)
                r = Result("x", p, {t: [None] * 4 for t in TARGETS}, blocks, d); vals.append((r.orc[0.99]["all"]["cov"], r.orc[0.98]["all"]["cov"], r.auc, r.ll))
            v = np.array(vals); lab = f"{n_tr}" if n_tr is not None else f"all (about {int(np.mean(n_used))})"
            fm = lambda i, f="{:.1f}": f.format(v[:, i].mean()) + (f" ({f.format(v[:, i].min())}-{f.format(v[:, i].max())})" if len(v) > 1 else "")
            print(f"| {lab} | {mn} | {fm(0)} | {fm(1)} | {fm(2, '{:.3f}')} | {fm(3, '{:.3f}')} |"); out[f"{n_tr}|{mn}"] = v.mean(0).tolist()
    RES["f"] = out
    print("\n#### (f2) training-only threshold with few 0139 training pairs: tau from the inner leave-one-block-out on the same few pairs; pooled test blocks; mean over 3 random subsets (min-max)")
    print("| 0139 training pairs | method | target | coverage % | precision % | k=1 coverage % | k=1 precision % | k>=2 coverage % |"); print("|---|---|---|---|---|---|---|---|")
    for n_tr in (100, 300, 1000):
        for mn, fn in (("(a) Platt", lambda tr, ev: calib_platt(p0, d)(tr, ev)), ("(a) isotonic", lambda tr, ev: calib_iso(p0, d)(tr, ev)), ("(b) weight 20", retrain(d, ALL, 20)), ("(c) boosting 0139 only", only139(d, ALL))):
            rs = [protocol(mn, fn, blocks, d, sub=n_tr, rep=rep) for rep in range(3)]
            for t in TARGETS:
                v = np.array([[r.ev[t]["all"]["cov"], r.ev[t]["all"]["prec"], r.ev[t][1]["cov"], r.ev[t][1]["prec"], 100 * (r.cert[t] & (d.truth >= 2)).sum() / (d.truth >= 2).sum()] for r in rs])
                fm = lambda i: f"{np.nanmean(v[:, i]):.1f} ({np.nanmin(v[:, i]):.1f}-{np.nanmax(v[:, i]):.1f})"
                print(f"| {n_tr} | {mn} | {int(t * 100)}% | " + " | ".join(fm(i) for i in range(5)) + " |")


def sec_bootstrap(byname, d, names, nboot=2000):
    """Bootstrap over the 16 slabs of 64 z layers (resample slabs with replacement, keep the certified flags of the pooled test run): intervals that allow for the spatial correlation of the pairs."""
    slab = np.clip(((d.z - ZLO) // 64).astype(int), 0, 15); idx_by = [np.nonzero(slab == s_)[0] for s_ in range(16)]; rng = np.random.default_rng(7)
    sels = [np.concatenate([idx_by[q] for q in rng.integers(0, 16, 16)]) for _ in range(nboot)]
    print("\n#### slab bootstrap (2000 resamples of the 16 slabs of 64 z layers), point estimate and 95% interval, pooled test blocks")
    print("| method | target | coverage % | precision % | k=1 coverage % | k>=2 coverage % |"); print("|---|---|---|---|---|---|")
    for t in TARGETS:
        for nm in names:
            r = byname[nm]; cert = r.cert[t]; v = []
            for sel in sels:
                c = cert[sel]; kk = d.truth[sel]; v.append((100 * c.mean(), 100 * d.y[sel][c].mean() if c.any() else np.nan, 100 * (c & (kk == 1)).sum() / (kk == 1).sum(), 100 * (c & (kk >= 2)).sum() / max((kk >= 2).sum(), 1)))
            v = np.array(v); pt = (100 * cert.mean(), 100 * d.y[cert].mean() if cert.any() else np.nan, 100 * (cert & (d.truth == 1)).sum() / (d.truth == 1).sum(), 100 * (cert & (d.truth >= 2)).sum() / (d.truth >= 2).sum())
            print(f"| {nm} | {int(t * 100)}% | " + " | ".join(f"{pt[i]:.1f} ({np.nanpercentile(v[:, i], 2.5):.1f}-{np.nanpercentile(v[:, i], 97.5):.1f})" for i in range(4)) + " |")


def sec_groups(d, blocks):
    """Which feature groups does the 0139-only model use?  Drop one group at a time, or keep only a few columns (thresholds from training data only, as in the main suite)."""
    groups = {"agreement counts (cols 0-8)": list(range(0, 9)), "image votes sp/ct (cols 9-12)": [9, 10, 11, 12], "confidences (cols 13-15)": [13, 14, 15], "length log1p(L) (col 16)": [16], "cand (col 17)": [17]}
    res = [protocol("(d2) (c) boosting, all features", only139(d, ALL), blocks, d)]
    for gname, cols in groups.items(): res.append(protocol(f"(d2) (c) boosting without {gname}", only139(d, [i for i in ALL if i not in cols]), blocks, d))
    for nm, cols in (("cand", [17]), ("cand + L", [16, 17]), ("cand + L + agreement counts", list(range(0, 9)) + [16, 17]), ("cand + agreement counts + image votes", list(range(0, 13)) + [17])): res.append(protocol(f"(d2) (c) boosting with only {nm}", only139(d, cols), blocks, d))
    print("\n#### (d2) feature groups of the 0139-only model")
    for t in TARGETS: print_overall(res, t, "[d2]"); print_perk(res, t, "[d2]")
    store(res, "d2")


def main():
    secs = set(sys.argv[1:]) or {"all"}; full = "full" in secs; out_json = "results.json" if "all" in secs else "results_partial.json"
    if "all" in secs: secs = {"base", "a", "b", "c", "d", "e", "f", "g"}
    t0 = time.time(); d = Data(); print(f"ladder pairs (1-4 wraps) {len(d.yl)}, ladder cand accuracy {100 * d.yl.mean():.1f}%; 0139 pairs {d.n}, k=1..4: {[int((d.truth == k).sum()) for k in KS]}, cand accuracy {100 * d.y.mean():.1f}% ({'; '.join(f'k={k} {100 * d.y[d.truth == k].mean():.1f}%' for k in KS)})")
    zb = z_blocks(d.z); print("z blocks: " + "; ".join(f"{zb.labels[j]} n={int((zb.ids == j).sum())} train n={int((zb.dist[j] >= GAP).sum())}" for j in range(NB)))
    suite_secs = secs & {"base", "a", "b", "c", "d"}
    if suite_secs:
        results, byname = sec_suite(d, zb, suite_secs | {"base"}, "z blocks")
        if "a" in secs: sec_calibration_table(d, zb)
        if "base" in secs: sec_rules(d)
        if {"a", "b", "c"} <= secs: sec_bootstrap(byname, d, ["ladder-only model [tau from ladders]", "ladder-only model [tau from 0139 train blocks]", "(a) Platt on logit(p)", "(a) isotonic", "(a) Platt on logit(p), cand, log1p(L)", "(b) ladders + 0139, weight 5", "(b) ladders + 0139, weight 20", "(c) 0139 train blocks only, boosting"])
        if "d" in secs: sec_groups(d, zb)
    if "e" in secs:
        E = recover_endpoints(d.T); base = ladder_tau_result("base", ladder_model_p0(d), d, (d.ladder_oof(ALL), d.yl), zb); sec_e(d, base, E)
    if "f" in secs: sec_f(d, zb)
    if "g" in secs:
        E = recover_endpoints(d.T); ym, xm, rad, azi = centre_radius(E)
        for gb in (geo_blocks(azi, ym, xm, "azimuth", " deg"), geo_blocks(rad, ym, xm, "radius", " vox")):
            print(f"\n{gb.name}: " + "; ".join(f"{gb.labels[j]} n={int((gb.ids == j).sum())} train n={int((gb.dist[j] >= GAP).sum())}" for j in range(4)))
            sec_suite(d, gb, {"base", "a", "b", "c", "d"} if full else {"base", "a", "b", "c"}, gb.name)
    json.dump(RES, open(OUT + out_json, "w"), default=float); print(f"\ndone in {time.time() - t0:.0f} s; numbers in {OUT}{out_json}")


def ladder_model_p0(d): return d.ladder_model(ALL).predict_proba(d.X)[:, 1]


if __name__ == "__main__":
    main()
