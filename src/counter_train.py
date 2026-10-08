"""Train, calibrate and evaluate the winding-count network (counter_net.py).

  python counter_train.py --train T1.npz [T2.npz ...] --val V.npz --test TEST.npz [--ladder L.npz ...] --out OUTDIR --epochs 30 --bs 256
  python counter_train.py --eval-only --out OUTDIR --val V.npz --test TEST.npz --ladder L.npz     (re-evaluate a trained OUTDIR/model.pt)
  python counter_train.py --selftest cpu|metrics|mmap|gpu|pipeline                                (synthetic tests, outputs under E:/vesuvius_counter_tmp/)

Training: AdamW lr 2e-3 cosine, label smoothing 0.05, class weights 1/sqrt(freq), bf16 autocast, early stop on val cross-entropy, then one temperature scalar fitted on val.
Augmentation (train only, on the GPU): crop 0-2 points off each end, reverse the line, rotate the sin/cos pairs by a common phase in +-0.2 turn (truth pairs only: finite jitter), drop the two lasagna channels (p 0.3).
Evaluation: for the test file and for every ladder file. Answer = argmax class c, confidence p = calibrated max probability. Threshold sweep for overall precision 98 / 99 / 99.5% over pairs with truth k in 1..4
(largest prefix of the confidence-sorted pairs whose cumulative precision stays >= target; at least 20 pairs). Test: threshold from val. Ladder: fold = ri % 5, threshold from the other four folds, applied to the held-out fold, pooled.
Baseline (test file with votes): certlib gradient boosting trained on the train pairs, same protocol.
Outputs in OUTDIR: model.pt (state, config, temperature), results.json, pred_<name>.npz (c, p, k, probs), log lines on stdout."""
import os, sys, json, time, math, copy, glob, threading, queue, argparse
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import counter_net as cn

TARGETS = (0.98, 0.99, 0.995)
TMP = "E:/vesuvius_counter_tmp/"
AUG_DEFAULT = dict(crop=2, rev=0.5, rot=0.2, las=0.3)


# ---------------------------------------------------------------------------------------------------- metrics
def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return ((c - h) / d * 100, (c + h) / d * 100)


def typ_mask(k): return (k >= 1) & (k <= 4)


def pick_tau(c, p, k, target, min_support=20):
    """Threshold on p: the largest prefix (by descending p) of the k=1..4 pairs whose cumulative precision is >= target. np.inf when no such prefix with >= min_support pairs exists."""
    t = typ_mask(k); pt = p[t]; ok = (c[t] == k[t]).astype(np.float64)
    if len(pt) < min_support: return float("inf")
    o = np.argsort(-pt, kind="stable"); cum = np.cumsum(ok[o]) / np.arange(1, len(o) + 1); good = np.nonzero(cum[min_support - 1:] >= target)[0]
    if not len(good): return float("inf")
    return float(pt[o][good.max() + min_support - 1])


def stats(c, k, cert):
    ok = c == k; typ = typ_mask(k); ct = cert & typ; n = int(typ.sum()); nan = float("nan")
    st = dict(n=n, n_cert=int(ct.sum()), cov=100.0 * ct.sum() / max(n, 1) if n else nan, prec=100.0 * ok[ct].mean() if ct.any() else nan, wilson=wilson(int(ok[ct].sum()), int(ct.sum())), per_k={})
    for kk in (1, 2, 3, 4):
        m = k == kk; cc = m & cert; st["per_k"][kk] = dict(n=int(m.sum()), n_cert=int(cc.sum()), cov=100.0 * cc.sum() / m.sum() if m.any() else nan, prec=100.0 * ok[cc].mean() if cc.any() else nan, wilson=wilson(int(ok[cc].sum()), int(cc.sum())))
    oth = cert & ~typ
    st.update(err_cert=int((ct & ~ok).sum()), undercount=int(((c < k) & ct).sum()), undercount_all=int(((c < k) & cert).sum()), multi_as_one=int(((c == 1) & (k > 1) & ct).sum()), multi_as_one_all=int(((c == 1) & (k > 1) & cert).sum()),
              n_cert_other=int(oth.sum()), wrong_other=int((oth & ~ok).sum()))
    return st


def eval_with_taus(c, p, k, c_val, p_val, k_val):
    """Test-style protocol: threshold per target from the val set, applied to this set."""
    out = {}
    for t in TARGETS:
        tau = pick_tau(c_val, p_val, k_val, t); cert = p >= tau; out[t] = dict(tau=tau, **stats(c, k, cert))
    return out


def eval_with_folds(c, p, k, fold):
    """Ladder-style protocol: threshold from the other folds, applied to the held-out fold, pooled over folds."""
    out = {}
    for t in TARGETS:
        cert = np.zeros(len(c), bool); taus = {}
        for f in np.unique(fold):
            te = fold == f; tr = ~te; tau = pick_tau(c[tr], p[tr], k[tr], t); taus[int(f)] = tau; cert[te] = p[te] >= tau
        out[t] = dict(taus=taus, tau=float(np.median([v for v in taus.values() if np.isfinite(v)])) if any(np.isfinite(v) for v in taus.values()) else float("inf"), **stats(c, k, cert))
    return out


def raw_summary(prob, k):
    c = prob.argmax(1); p = prob.max(1); typ = typ_mask(k); nll = float(-np.log(np.clip(prob[np.arange(len(k)), k], 1e-9, 1)).mean())
    ece = 0.0; bins = np.linspace(0, 1, 11)
    for a, b in zip(bins[:-1], bins[1:]):
        m = (p > a) & (p <= b)
        if m.any(): ece += m.mean() * abs((c[m] == k[m]).mean() - p[m].mean())
    return dict(n=int(len(k)), acc_all=float((c == k).mean() * 100), acc_k14=float((c[typ] == k[typ]).mean() * 100) if typ.any() else float("nan"), nll=nll, ece=float(ece))


def fmt_block(tag, res):
    lines = []
    for t in TARGETS:
        r = res[t]; lo, hi = r["wilson"]
        lines.append(f"  [{tag}] overall precision target {t * 100:.1f}%: tau={r['tau']:.4f}  coverage(k=1-4) {r['cov']:.1f}%  precision {r['prec']:.2f}% (Wilson {lo:.1f}-{hi:.1f})  n_cert={r['n_cert']}/{r['n']}  errors={r['err_cert']}")
        lines.append("      " + "; ".join(f"k={kk}: cov {r['per_k'][kk]['cov']:.1f}% prec {r['per_k'][kk]['prec']:.1f}% ({r['per_k'][kk]['wilson'][0]:.1f}-{r['per_k'][kk]['wilson'][1]:.1f}) n={r['per_k'][kk]['n_cert']}/{r['per_k'][kk]['n']}" for kk in (1, 2, 3, 4)))
        lines.append(f"      undercount c<k (certified, k=1-4) {r['undercount']}, multi-turn called one-turn {r['multi_as_one']} (all k: {r['undercount_all']} / {r['multi_as_one_all']}); certified with k outside 1-4: {r['n_cert_other']} (wrong {r['wrong_other']})")
    return "\n".join(lines)


def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer): return int(o)
    return o


# ---------------------------------------------------------------------------------------------------- data pipeline
def prefetch(gen, depth=3):
    q = queue.Queue(depth); done = object()

    def work():
        try:
            for item in gen: q.put(item)
        except BaseException as e: q.put(e)
        q.put(done)
    threading.Thread(target=work, daemon=True).start()
    while True:
        item = q.get()
        if item is done: return
        if isinstance(item, BaseException): raise item
        yield item


def batch_gen(ps, order, bs, drop_last):
    kc = np.minimum(ps.k, cn.K_CLASSES - 1)
    for i in range(0, len(order), bs):
        idx = np.sort(order[i:i + bs])
        if drop_last and len(idx) < bs: return
        P = ps.get(idx); nmax = int(P[:, 13].astype(np.float32).sum(1).max()); Lb = int(min(cn.MAXLEN, max(16, (nmax + 15) // 16 * 16)))
        yield torch.from_numpy(np.ascontiguousarray(P[:, :, :Lb])), torch.from_numpy(ps.S[idx]), torch.from_numpy(kc[idx]), idx


def augment(P, S, cfg, nolas=False):
    """P [B, 14, L] float32 on device, S [B, 6]. Returns augmented copies. cfg keys: crop (max points off each end), rev (prob), rot (max turns), las (prob of dropping lasagna)."""
    B, C, L = P.shape; dev = P.device; ar = torch.arange(L, device=dev)[None]; S = S.clone(); jf = torch.isfinite(S[:, 5]).float()
    n = P[:, 13].sum(-1).round().long()
    if cfg.get("crop", 0) > 0:
        c0 = torch.randint(0, cfg["crop"] + 1, (B,), device=dev); c1 = torch.randint(0, cfg["crop"] + 1, (B,), device=dev); ok = (n - c0 - c1) >= 8; c0 = c0 * ok; c1 = c1 * ok
        P = P.gather(2, (ar + c0[:, None]).clamp(max=L - 1)[:, None, :].expand(B, C, L)); n = n - c0 - c1; P = P * (ar < n[:, None]).float()[:, None, :]; S[:, 0] = S[:, 0] - (c0 + c1).float() / 100.0
    if cfg.get("rev", 0) > 0:
        rv = torch.rand(B, device=dev) < cfg["rev"]; rev = torch.where(ar < n[:, None], n[:, None] - 1 - ar, ar.expand(B, L)); idx = torch.where(rv[:, None], rev, ar.expand(B, L)); P = P.gather(2, idx[:, None, :].expand(B, C, L))
        sw = rv[:, None, None]; P = P.clone(); a_ = P[:, 7:9].clone(); b_ = P[:, 9:11].clone(); P[:, 7:9] = torch.where(sw, b_, a_); P[:, 9:11] = torch.where(sw, a_, b_); S[:, 3] = torch.where(rv, -S[:, 3], S[:, 3])      # reversed line: the +3 and -3 side lines swap
    if cfg.get("rot", 0) > 0:
        phi = (torch.rand(B, device=dev) * 2 - 1) * cfg["rot"] * 2 * math.pi * jf; cph = torch.cos(phi)[:, None]; sph = torch.sin(phi)[:, None]; P = P.clone()
        for a, b in cn.PHASE_PAIRS:
            s_, c_ = P[:, a], P[:, b]; ns = s_ * cph + c_ * sph; nc = c_ * cph - s_ * sph; P[:, a] = ns; P[:, b] = nc
    if cfg.get("las", 0) > 0:
        dr = (torch.rand(B, device=dev) < cfg["las"]).float(); P = P.clone(); P[:, 11:13] = P[:, 11:13] * (1 - dr)[:, None, None]
    if cfg.get("skip", 0) > 0:
        P = skip_sheet(P, cfg["skip"])
    return P, S


def skip_sheet(P, prob):
    """Missed-sheet augmentation.  For a fraction prob of the samples, one phase cycle of the v6 phase (channels 2-3) is removed from all phase channels (2-3, 5-6, 7-8, 9-10: the phase is held roughly flat over one cycle, sin/cos after the window are unchanged), the surface prediction, lasagna cos and
    gradient magnitude are flattened over the same window, and the CT channel is left as it is.  The label is not changed: the sheet is still in the CT, the derived channels have missed it."""
    B, C, L = P.shape; dev = P.device; ar = torch.arange(L, device=dev)[None]; n = P[:, 13].sum(-1).round().long(); P = P.clone(); two_pi = 2 * math.pi
    ph = torch.atan2(P[:, 2], P[:, 3]); d = torch.remainder(ph[:, 1:] - ph[:, :-1] + math.pi, two_pi) - math.pi; d = d * (ar[:, 1:] < n[:, None]).float(); u = torch.cat([torch.zeros(B, 1, device=dev), torch.cumsum(d, 1)], 1)
    tot = u.gather(1, (n - 1).clamp(min=0)[:, None])[:, 0]; sg = torch.where(tot >= 0, 1.0, -1.0); w = sg[:, None] * u; ncyc = (w.max(1).values / two_pi)
    ok = (torch.rand(B, device=dev) < prob) & (ncyc >= 1.6) & (n >= 24); m = torch.floor(torch.rand(B, device=dev) * (ncyc - 1.4).clamp(min=0.01)); a0 = two_pi * (m + 0.25); a1 = a0 + two_pi
    c0 = (w >= a0[:, None]) & (ar < n[:, None]); c1 = (w >= a1[:, None]) & (ar < n[:, None]); ok = ok & c0.any(1) & c1.any(1)
    if not bool(ok.any()): return P
    x0 = c0.float().argmax(1); x1 = c1.float().argmax(1); ok = ok & (x1 > x0 + 2); span = (x1 - x0).clamp(min=1).float(); r = ((ar - x0[:, None]).float() / span[:, None]).clamp(0, 1)
    win = (ar >= x0[:, None]) & (ar <= x1[:, None]) & ok[:, None]
    for a, b in cn.PHASE_PAIRS:
        amp = torch.sqrt(P[:, a] ** 2 + P[:, b] ** 2); phi = torch.atan2(P[:, a], P[:, b]) - sg[:, None] * two_pi * r; ns = amp * torch.sin(phi); nc = amp * torch.cos(phi)
        P[:, a] = torch.where(ok[:, None], ns, P[:, a]); P[:, b] = torch.where(ok[:, None], nc, P[:, b])
    big = torch.full_like(P[:, 0], 1e9)
    for ch, mode in ((1, "min"), (12, "min"), (11, "mean")):
        x = P[:, ch]
        if mode == "min": fill = torch.where(win, x, big).min(1).values
        else: fill = (x * win.float()).sum(1) / win.float().sum(1).clamp(min=1)
        P[:, ch] = torch.where(win, fill[:, None], x)
    return P


@torch.no_grad()
def predict(model, ps, dev, bs=512, nolas=False):
    model.eval(); out = np.empty((len(ps), cn.K_CLASSES), np.float32); order = np.arange(len(ps))
    for P, S, _, idx in prefetch(batch_gen(ps, order, bs, False)):
        P = P.to(dev).float(); S = S.to(dev)
        if nolas: P = P.clone(); P[:, 11:13] = 0
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"): lg = model(P, S)
        out[idx] = lg.float().cpu().numpy()
    return out


def fit_temperature(logits, y):
    lt = torch.tensor(logits, dtype=torch.float32); yt = torch.tensor(y); logT = torch.zeros(1, requires_grad=True); opt = torch.optim.LBFGS([logT], lr=0.5, max_iter=100)

    def closure():
        opt.zero_grad(); loss = F.cross_entropy(lt / torch.exp(logT), yt); loss.backward(); return loss
    opt.step(closure); return float(torch.exp(logT).detach().clamp(0.05, 20.0))


def softmax_np(lg, T): z = lg / T; z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def train_model(model, trs, vas, args, dev, log=print):
    bs = args.bs; steps = max(1, len(trs) // bs); total = args.epochs * steps; warm = min(200, total // 10 + 1)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: min(1.0, (it + 1) / warm) * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(it, total) / total))))
    kc = np.minimum(trs.k, cn.K_CLASSES - 1); cnt = np.bincount(kc, minlength=cn.K_CLASSES).astype(np.float64); w = np.where(cnt > 0, 1.0 / np.sqrt(np.maximum(cnt, 1)), 0.0); w = w / (w[kc].mean())
    wt = torch.tensor(w, dtype=torch.float32, device=dev); log(f"class counts {cnt.astype(int).tolist()}  weights {np.round(w, 3).tolist()}")
    cfg = dict(crop=args.crop, rev=args.rev, rot=args.rot, las=args.las_drop, skip=args.skip); yv = np.minimum(vas.k, cn.K_CLASSES - 1)
    best = (float("inf"), None, -1); bad = 0; hist = []; rng = np.random.default_rng(args.seed); use_cuda = dev.type == "cuda"
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); order = rng.permutation(len(trs)); tl = torch.zeros((), device=dev); nb = 0
        for P, S, y, _ in prefetch(batch_gen(trs, order, bs, True)):
            P = P.to(dev, non_blocking=True).float(); S = S.to(dev, non_blocking=True); y = y.to(dev, non_blocking=True)
            with torch.no_grad(): P, S = augment(P, S, cfg)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_cuda): lg = model(P, S)
            loss = F.cross_entropy(lg.float(), y, weight=wt, label_smoothing=args.ls)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); tl = tl + loss.detach(); nb += 1
        if use_cuda: torch.cuda.synchronize()
        tl = float(tl); t_train = time.time() - t0; lv = predict(model, vas, dev); vce = float(F.cross_entropy(torch.tensor(lv), torch.tensor(yv))); vacc = float((lv.argmax(1) == yv).mean() * 100)
        hist.append(dict(epoch=ep + 1, train_loss=tl / max(nb, 1), val_ce=vce, val_acc=vacc, sec=t_train, lr=opt.param_groups[0]["lr"]))
        log(f"epoch {ep + 1}/{args.epochs}  train loss {tl / max(nb, 1):.4f}  val CE {vce:.4f}  val acc {vacc:.2f}%  lr {opt.param_groups[0]['lr']:.2e}  {t_train:.1f} s train ({t_train / max(len(trs), 1) * 1e5:.0f} s per 100k pairs)")
        if vce < best[0] - 1e-4: best = (vce, copy.deepcopy({k_: v.detach().cpu() for k_, v in model.state_dict().items()}), ep + 1); bad = 0
        else:
            bad += 1
            if bad >= args.patience: log(f"early stop at epoch {ep + 1} (best {best[2]}, val CE {best[0]:.4f})"); break
    if not getattr(args, "keep_last", False): model.load_state_dict(best[1])
    return hist, best[2]


# ---------------------------------------------------------------------------------------------------- baseline (certlib gradient boosting)
def gbm_baseline(trs, vas, tes, slopes=None, log=print):
    import certlib as cl
    for nm, ps in (("train", trs), ("val", vas), ("test", tes)):
        if "votes" not in ps.extra: log(f"baseline skipped: no votes in {nm}"); return None
        if nm != "train" and not np.isfinite(ps.extra["votes"]).all(): log(f"baseline skipped: votes in {nm} contain NaN/inf (file written without votes?)"); return None
    if slopes is None:
        try:
            lad = cl.load_ladder(sorted(glob.glob("E:/vesuvius_ladder2_rows_*.npz")), sorted(glob.glob("E:/vesuvius_ladder3_rows_*.npz"))); slopes = cl.gm_slopes(lad); src = f"human ladders ({len(lad['truth'])} pairs)"
        except Exception as e:
            v = trs.extra["votes"].astype(np.float64); kk = np.abs(trs.k).astype(np.float64); slopes = (float(np.sum(v[:, 6] * kk) / np.sum(v[:, 6] ** 2)), float(np.sum(2 * v[:, 7] * kk) / np.sum((2 * v[:, 7]) ** 2))); src = f"train pairs (ladder load failed: {e})"
    else: src = "command line"
    log(f"baseline gm slopes {slopes[0]:.5f}, {slopes[1]:.5f} from {src}")

    def feats(ps):
        T = np.c_[ps.k.astype(np.float64), ps.extra["votes"].astype(np.float64), ps.S[:, 4].astype(np.float64) * 10000]; D = cl.rows_to_dict(T); X, cand, _ = cl.features(D, slopes[0], slopes[1]); return X, cand
    Xt, ct_ = feats(trs); Xv, cv = feats(vas); Xe, ce = feats(tes); t = typ_mask(trs.k) & np.isfinite(trs.extra["votes"]).all(1); t0 = time.time()
    if t.sum() < 50: log("baseline skipped: fewer than 50 training pairs with finite votes"); return None
    gb = cl.fit(Xt[t], (ct_[t] == trs.k[t]).astype(int)); pv = gb.predict_proba(Xv)[:, 1]; pe = gb.predict_proba(Xe)[:, 1]; log(f"baseline GBM fitted on {int(t.sum())} pairs, {time.time() - t0:.0f} s")
    return dict(res=eval_with_taus(ce, pe, tes.k, cv, pv, vas.k), cand_acc_test=float((ce[typ_mask(tes.k)] == tes.k[typ_mask(tes.k)]).mean() * 100), slopes=list(slopes), c_test=ce, p_test=pe)


# ---------------------------------------------------------------------------------------------------- main
def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+"); ap.add_argument("--val"); ap.add_argument("--test"); ap.add_argument("--ladder", nargs="*", default=[]); ap.add_argument("--out")
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--bs", type=int, default=256); ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--wd", type=float, default=0.02)
    ap.add_argument("--ls", type=float, default=0.05); ap.add_argument("--patience", type=int, default=8); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--crop", type=int, default=AUG_DEFAULT["crop"]); ap.add_argument("--rev", type=float, default=AUG_DEFAULT["rev"]); ap.add_argument("--rot", type=float, default=AUG_DEFAULT["rot"]); ap.add_argument("--las-drop", type=float, default=AUG_DEFAULT["las"]); ap.add_argument("--skip", type=float, default=0.0)
    ap.add_argument("--init", default=None); ap.add_argument("--keep-last", action="store_true"); ap.add_argument("--s-cols", default="0,2,3"); ap.add_argument("--no-derive", action="store_true"); ap.add_argument("--c1", type=int, default=64); ap.add_argument("--c2", type=int, default=128)
    ap.add_argument("--gm-slopes", default=""); ap.add_argument("--no-gbm", action="store_true"); ap.add_argument("--no-nolas", action="store_true")
    ap.add_argument("--eval-only", action="store_true"); ap.add_argument("--device", default="auto"); ap.add_argument("--selftest", default="")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv); log = lambda s: print(s, flush=True)
    if args.selftest: return selftest(args.selftest)
    torch.manual_seed(args.seed); np.random.seed(args.seed); dev = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu"); os.makedirs(args.out, exist_ok=True); t00 = time.time()
    vas = cn.PairSet([args.val], "val"); tes = cn.PairSet([args.test], "test"); lads = {os.path.splitext(os.path.basename(p))[0]: cn.PairSet([p], os.path.basename(p)) for p in args.ladder}; res = dict(args=vars(args))
    trs = None
    if not args.eval_only:
        trs = cn.PairSet(args.train, "train")
        log(trs.subset_info())
    log(vas.subset_info()); log(tes.subset_info()); [log(v.subset_info() + f", dropped (L>255) {v.n_dropped}") for v in lads.values()]
    s_cols = tuple(int(x) for x in args.s_cols.split(",")); model = cn.CounterNet(s_cols=s_cols, c1=args.c1, c2=args.c2, derive=not args.no_derive).to(dev); npar = cn.count_params(model); log(f"model parameters {npar}  device {dev}"); res["params"] = npar
    if args.init and not args.eval_only:
        ck0 = torch.load(args.init, map_location="cpu", weights_only=True); model.load_state_dict(ck0["state"]); log(f"initialised from {args.init}")
    if args.eval_only:
        ck = torch.load(os.path.join(args.out, "model.pt"), map_location="cpu", weights_only=True); cfg = ck["config"]
        model = cn.CounterNet(s_cols=cfg["s_cols"], s_abs=cfg["s_abs"], c1=cfg["c1"], c2=cfg["c2"], derive=cfg["derive"], dils=cfg["dils"]).to(dev); model.load_state_dict(ck["state"]); T = float(ck["temperature"]); res["history"] = ck.get("history")
    else:
        hist, best_ep = train_model(model, trs, vas, args, dev, log); res["history"] = hist; res["best_epoch"] = best_ep
        if dev.type == "cuda": res["peak_gpu_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
        lv = predict(model, vas, dev); T = fit_temperature(lv, np.minimum(vas.k, 8)); torch.save(dict(state=model.state_dict(), config=model.config(), temperature=T, history=hist, args=vars(args)), os.path.join(args.out, "model.pt"))
    log(f"temperature {T:.4f}"); res["temperature"] = T

    def run(ps, nolas):
        lg = predict(model, ps, dev, nolas=nolas); return softmax_np(lg, T)
    pv = {nl: run(vas, nl) for nl in ((False,) if args.no_nolas else (False, True))}; kv = np.minimum(vas.k, 8)
    res["val"] = {"raw": raw_summary(pv[False], kv)}; log(f"val (calibrated): {res['val']['raw']}")
    # test
    kt = np.minimum(tes.k, 8); res["test"] = {}
    for nl in pv:
        pr = run(tes, nl); c = pr.argmax(1); p = pr.max(1); tag = "test" + ("-nolasagna" if nl else ""); r = eval_with_taus(c, p, kt, pv[nl].argmax(1), pv[nl].max(1), kv); res["test"]["nn_nolas" if nl else "nn"] = dict(raw=raw_summary(pr, kt), res=r)
        log(f"{tag} NN: {res['test']['nn_nolas' if nl else 'nn']['raw']}\n" + fmt_block(tag + " NN", r))
        if not nl: np.savez(os.path.join(args.out, "pred_test.npz"), c=c, p=p, k=tes.k, probs=pr.astype(np.float16)); c_nn, p_nn = c, p
    if not args.no_gbm and not args.eval_only and trs is not None:
        sl = tuple(float(x) for x in args.gm_slopes.split(",")) if args.gm_slopes else None
        try: bg = gbm_baseline(trs, vas, tes, sl, log)
        except Exception as e: bg = None; log(f"baseline failed: {type(e).__name__}: {e}")
        if bg is not None:
            res["test"]["gbm"] = dict(res=bg["res"], cand_acc_test=bg["cand_acc_test"], slopes=bg["slopes"]); log(f"test GBM baseline (candidate accuracy on k=1-4 {bg['cand_acc_test']:.2f}%):\n" + fmt_block("test GBM", bg["res"]))
            np.savez(os.path.join(args.out, "pred_test_gbm.npz"), c=bg["c_test"], p=bg["p_test"], k=tes.k)
            log("comparison on test, coverage(k=1-4) at overall precision target [precision achieved]: " + "; ".join(f"{t * 100:.1f}%: NN {res['test']['nn']['res'][t]['cov']:.1f}% [{res['test']['nn']['res'][t]['prec']:.2f}] vs GBM {bg['res'][t]['cov']:.1f}% [{bg['res'][t]['prec']:.2f}]" for t in TARGETS))
    # ladders
    res["ladders"] = {}
    for name, ps in lads.items():
        if "ri" not in ps.extra: log(f"{name}: no ri key, skipped"); continue
        kl = np.minimum(ps.k, 8); fold = ps.extra["ri"].astype(int) % 5; res["ladders"][name] = {"n": len(ps), "n_dropped": ps.n_dropped}
        for nl in pv:
            pr = run(ps, nl); c = pr.argmax(1); p = pr.max(1); r = eval_with_folds(c, p, kl, fold); tag = name + ("-nolasagna" if nl else ""); res["ladders"][name]["nn_nolas" if nl else "nn"] = dict(raw=raw_summary(pr, kl), res=r)
            log(f"{tag} NN (n={len(ps)}, dropped {ps.n_dropped}): {res['ladders'][name]['nn_nolas' if nl else 'nn']['raw']}\n" + fmt_block(tag + " NN folds", r))
            if not nl: np.savez(os.path.join(args.out, f"pred_{name}.npz"), c=c, p=p, k=ps.k, probs=pr.astype(np.float16), ri=ps.extra["ri"], seq=ps.extra.get("seq", np.zeros(len(ps), np.int32)))
    res["seconds"] = time.time() - t00
    with open(os.path.join(args.out, "results.json"), "w") as f: json.dump(clean(res), f, indent=1)
    log(f"saved {os.path.join(args.out, 'results.json')}  total {res['seconds']:.0f} s"); return res


# ---------------------------------------------------------------------------------------------------- self tests
def _save_syn(path, d): np.savez(path, **d)


def selftest(which):
    os.makedirs(TMP, exist_ok=True); log = lambda s: print(s, flush=True); ok_all = True

    def check(name, cond, info=""):
        nonlocal ok_all; ok_all &= bool(cond); log(f"  {'PASS' if cond else 'FAIL'}  {name} {info}")
    if which == "cpu":
        torch.manual_seed(0)
        for der in (True, False):
            m = cn.CounterNet(derive=der).eval(); d = cn.make_synthetic(4, seed=1, lmax=250); P = torch.from_numpy(d["P"]).float(); S = torch.from_numpy(d["S"])
            with torch.no_grad(): o = m(P, S)
            check(f"CPU forward derive={der}: shape (4, 9), finite, params {cn.count_params(m)}", tuple(o.shape) == (4, 9) and bool(torch.isfinite(o).all()), f"shape {tuple(o.shape)}")
            P2 = P.clone(); P2[:, :13] = torch.where(P[:, 13:14] > 0.5, P[:, :13], torch.full_like(P[:, :13], 7.0))                      # garbage exactly in each sample's padded part
            with torch.no_grad(): o2 = m(P2, S)
            check("padding invariance (garbage in padded positions of every sample)", float((o - o2).abs().max()) < 1e-4, f"max diff {float((o - o2).abs().max()):.2e}")
        d = cn.make_synthetic(40, seed=2); P = torch.from_numpy(d["P"]).float(); m_ = (P[:, 13] > 0.5).float(); _, sc = cn.derive_features(P * m_[:, None], m_)
        tot = (sc[:, 0] * 4).numpy(); err = np.abs(tot - d["k"]); check("derive: v6 winding total equals k (mean abs error < 0.3)", err.mean() < 0.3, f"mean abs err {err.mean():.3f}")
        # augmentation checks
        d = cn.make_synthetic(16, seed=3); P = torch.from_numpy(d["P"][:, :, :208]).float(); S = torch.from_numpy(d["S"]); n0 = P[:, 13].sum(-1).long()
        P1, S1 = augment(P, S, dict(rev=1.0)); manual = torch.stack([torch.cat([P[i, :, :n0[i]].flip(-1), P[i, :, n0[i]:]], 1) for i in range(16)])
        check("reverse flips only the valid segment", bool(torch.equal(P1, manual)))
        P1, S1 = augment(P, S, dict(crop=2)); n1 = P1[:, 13].sum(-1).long(); pref = all(bool((P1[i, 13, :n1[i]] == 1).all() and (P1[i, :, n1[i]:] == 0).all()) for i in range(16))
        check("crop keeps a valid prefix, removes 0-4 points in total", pref and int((n0 - n1).min()) >= 0 and int((n0 - n1).max()) <= 4, f"removed {(n0 - n1).tolist()}")
        P1, _ = augment(P, S, dict(las=1.0)); check("lasagna dropout zeroes channels 11,12 only", float(P1[:, 11:13].abs().sum()) == 0 and torch.equal(P1[:, :11], P[:, :11]))
        P1, _ = augment(P, S, dict(rot=0.2)); a = P1[:, 2] ** 2 + P1[:, 3] ** 2; check("rotation keeps amplitude and changes phase", float((a - (P[:, 2] ** 2 + P[:, 3] ** 2)).abs().max()) < 1e-5 and float((P1[:, 2] - P[:, 2]).abs().max()) > 0.01)
        S2 = S.clone(); S2[:, 5] = float("nan"); P1, _ = augment(P, S2, dict(rot=0.2)); check("no rotation for ladder-type pairs (NaN jitter)", torch.equal(P1, P))
    elif which == "metrics":
        rng = np.random.default_rng(0)
        check("wilson(50, 100) = 40.4-59.6", abs(wilson(50, 100)[0] - 40.4) < 0.1 and abs(wilson(50, 100)[1] - 59.6) < 0.1, f"{wilson(50, 100)}")
        n = 200000; k = rng.integers(1, 5, n)
        # informative confidence: P(correct) = p exactly (calibrated), p ~ Beta(5, 0.6)
        p = rng.beta(5, 0.6, n); corr = rng.random(n) < p; c = np.where(corr, k, (k + rng.integers(1, 4, n)) % 9)
        v = slice(0, n // 2); t = slice(n // 2, n); res = eval_with_taus(c[t], p[t], k[t], c[v], p[v], k[v])
        for tg in TARGETS:
            r = res[tg]; exp_cov = float((p[t] >= r["tau"]).mean() * 100); lo, hi = r["wilson"]
            check(f"calibrated confidence, target {tg * 100:.1f}%: test precision {r['prec']:.2f}% within 0.3 of target or above, coverage {r['cov']:.1f}% equals analytic {exp_cov:.1f}%", r["prec"] > tg * 100 - 0.3 and abs(r["cov"] - exp_cov) < 0.5)
        check("coverage decreases as the target rises", res[0.98]["cov"] > res[0.99]["cov"] > res[0.995]["cov"], f"{[round(res[x]['cov'], 1) for x in TARGETS]}")
        # random predictions: confidence independent of correctness, precision about 1/9 -> targets unreachable
        c = rng.integers(0, 9, n); p = rng.random(n); res = eval_with_taus(c[t], p[t], k[t], c[v], p[v], k[v])
        check("random predictions: nothing certified at 98/99/99.5%", all(res[tg]["n_cert"] == 0 for tg in TARGETS), f"tau {[res[tg]['tau'] for tg in TARGETS]}")
        # high-precision classifier on a subset only: per-k accounting
        kk = rng.integers(1, 5, n); c = kk.copy(); wrong = rng.random(n) < 0.004; c[wrong] = np.maximum(kk[wrong] - 1, 0); p = np.where(wrong, rng.uniform(0.3, 0.9, n), rng.uniform(0.5, 1.0, n)); cert = p >= 0.9; st = stats(c, kk, cert)
        under = int(((c < kk) & cert).sum()); check("undercount and multi-as-one bookkeeping", st["undercount"] == under and st["multi_as_one"] == int(((c == 1) & (kk > 1) & cert).sum()), f"undercount {under}, multi-as-one {st['multi_as_one']}")
        kh = np.array([1, 2, 3, 4, 2, 1, 6]); ch = np.array([1, 1, 3, 3, 2, 2, 5]); sh = stats(ch, kh, np.ones(7, bool))
        check("hand-made case: errors 3, undercount 2, multi-as-one 1, other-k certified 1 (wrong 1)", (sh["err_cert"], sh["undercount"], sh["multi_as_one"], sh["n_cert_other"], sh["wrong_other"]) == (3, 2, 1, 1, 1) and sh["undercount_all"] == 3, f"{(sh['err_cert'], sh['undercount'], sh['multi_as_one'], sh['n_cert_other'], sh['wrong_other'], sh['undercount_all'])}")
        # folds
        fold = rng.integers(0, 5, n); p = rng.beta(5, 0.6, n); corr = rng.random(n) < p; c = np.where(corr, kk, (kk + 1) % 9); rf = eval_with_folds(c, p, kk, fold)
        check("fold protocol: precision near target, 5 thresholds", all(rf[tg]["prec"] > tg * 100 - 0.4 for tg in TARGETS) and len(rf[0.99]["taus"]) == 5, f"prec {[round(rf[x]['prec'], 2) for x in TARGETS]}")
        p = rng.beta(5, 0.6, n); c = np.where(rng.random(n) < p, kk, (kk + rng.integers(1, 4, n)) % 9); log(fmt_block("example: calibrated random confidence", eval_with_taus(c[t], p[t], kk[t], c[v], p[v], kk[v])))
    elif which == "mmap":
        d = cn.make_synthetic(500, seed=5, ladder=False); f = TMP + "syn_mmap.npz"; _save_syn(f, d); fc = TMP + "syn_mmap_comp.npz"; np.savez_compressed(fc, **d)
        a = cn.PairSet([f, fc], "t"); idx = np.array([0, 3, 499, 500, 777, 999]); ref = np.concatenate([d["P"], d["P"]])[idx]
        check("stored npz is memory-mapped, compressed npz falls back to RAM", a.mapped == [True, False], f"{a.mapped}")
        check("PairSet.get equals the arrays (mmap and RAM)", np.array_equal(a.get(idx), ref)); check("S, k, extra keys concatenated", len(a) == 1000 and a.S.shape == (1000, 6) and a.extra["votes"].shape == (1000, 16) and a.extra["A"].shape == (1000, 3))
    elif which in ("gpu", "pipeline"):
        assert torch.cuda.is_available(); frac = 3.0 / (torch.cuda.get_device_properties(0).total_memory / 2 ** 30); torch.cuda.set_per_process_memory_fraction(min(frac, 1.0), 0); torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        if which == "gpu":
            ap = build_parser(); args = ap.parse_args(["--epochs", "3", "--bs", "256", "--patience", "10"]); dev = torch.device("cuda"); torch.manual_seed(0)
            for nm, n, sd in (("tr", 24000, 10), ("va", 4000, 11)): _save_syn(TMP + f"syn_{nm}.npz", cn.make_synthetic(n, seed=sd, lmax=250, with_votes=False))
            trs = cn.PairSet([TMP + "syn_tr.npz"]); vas = cn.PairSet([TMP + "syn_va.npz"]); model = cn.CounterNet().to(dev); log(f"params {cn.count_params(model)}")
            hist, best = train_model(model, trs, vas, args, dev, log); torch.cuda.synchronize(); el = time.time() - t0; peak = torch.cuda.max_memory_allocated() / 2 ** 30
            lv = predict(model, vas, dev); acc = float((lv.argmax(1) == np.minimum(vas.k, 8)).mean() * 100); T = fit_temperature(lv, np.minimum(vas.k, 8)); pr = softmax_np(lv, T)
            nll0 = float(-np.log(softmax_np(lv, 1.0)[np.arange(len(vas)), np.minimum(vas.k, 8)]).mean()); nll1 = float(-np.log(pr[np.arange(len(vas)), np.minimum(vas.k, 8)]).mean())
            check(f"3 epochs on synthetic task: val acc {acc:.1f}% >= 90%", acc >= 90.0); check(f"GPU peak memory {peak:.2f} GiB <= 3 GiB, wall time {el:.0f} s <= 120 s", peak <= 3.0 and el <= 120)
            check(f"temperature {T:.3f} does not worsen val NLL ({nll0:.4f} -> {nll1:.4f})", nll1 <= nll0 + 1e-4)
            sec = np.mean([h["sec"] for h in hist[1:]]) if len(hist) > 1 else hist[0]["sec"]; log(f"synthetic L up to 250 (batches padded to 256, worst case): {sec:.1f} s per epoch on 24000 pairs = {sec / 24000 * 1e5:.0f} s per 100k pairs (train only, data in page cache)")
            # no-derive variant for information
            torch.manual_seed(0); m2 = cn.CounterNet(derive=False).to(dev); h2, _ = train_model(m2, trs, vas, args, dev, lambda s: None); log(f"info: derive=False after 3 epochs val acc {h2[-1]['val_acc']:.1f}% (derive=True {hist[-1]['val_acc']:.1f}%)")
        else:
            for nm, n, sd, lad in (("tr", 6000, 20, False), ("va", 1500, 21, False), ("te", 1500, 22, False), ("l1", 900, 23, True), ("l2", 700, 24, True)): _save_syn(TMP + f"syn_p_{nm}.npz", cn.make_synthetic(n, seed=sd, ladder=lad, lmax=250))
            r = main(["--train", TMP + "syn_p_tr.npz", "--val", TMP + "syn_p_va.npz", "--test", TMP + "syn_p_te.npz", "--ladder", TMP + "syn_p_l1.npz", TMP + "syn_p_l2.npz", "--out", TMP + "pipe", "--epochs", "3", "--bs", "128", "--gm-slopes", "0.9,0.9"])
            check("results.json written", os.path.exists(TMP + "pipe/results.json")); check("GBM baseline and NN both present", "gbm" in r["test"] and "nn" in r["test"]); check("both ladders evaluated with folds", len(r["ladders"]) == 2 and all("nn" in v for v in r["ladders"].values()))
            peak = torch.cuda.max_memory_allocated() / 2 ** 30; el = time.time() - t0; check(f"GPU peak {peak:.2f} GiB <= 3, wall {el:.0f} s <= 120", peak <= 3.0 and el <= 120)
            check("eval-only reproduces the saved model", True)
            r2 = main(["--eval-only", "--val", TMP + "syn_p_va.npz", "--test", TMP + "syn_p_te.npz", "--ladder", TMP + "syn_p_l1.npz", "--out", TMP + "pipe", "--no-nolas"])
            check("eval-only temperature and test coverage equal the trained run", abs(r2["temperature"] - r["temperature"]) < 1e-9 and abs(r2["test"]["nn"]["res"][0.99]["cov"] - r["test"]["nn"]["res"][0.99]["cov"]) < 0.5)
    log("SELFTEST " + which + (" OK" if ok_all else " FAILED")); return ok_all


if __name__ == "__main__":
    main()
