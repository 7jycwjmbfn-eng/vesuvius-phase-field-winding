"""Slips per wrap on held-out human ladders, for an official fit_spiral checkpoint (box A).

Follows PCU exp/e31b_bootstrap.py (MIT): each ladder point gets the winding whose fitted surface passes
closest (KD-tree over surface vertices, radius 40, nearest vertex per winding, 5x5 vertex window
upsampled 6x linearly; < 10 voxels else unassigned); a pair (i, i+1) of a ladder sorted by wind_a
slips when sg * diff(assigned winding) != diff(wind_a), sg = sign(nanmedian(diff(assigned))) (+1 if 0).
Bootstrap: resample ladders (default_rng(1), 2000), 95% interval = 2.5 / 97.5 percentiles; paired
differences reuse the same resampled ladder indices for both arms.

The fit writes no surface meshes when it has no winding constraint (output range collapses), so the
surfaces are rebuilt from checkpoint_fitted.ckpt exactly as flatten_spiral_checkpoint.py does
(SpiralAndTransform.inv on get_spiral_yxs grids, step 20, windings first..shell_outer_winding_idx,
z clipped to the fit z range). Winding index = the index of the winding in that grid family.

commands (run with the fitter venv python):
  score  RUN_DIR_OR_CKPT --out SCORE.json [--constraints relative_windings.json] [--device cuda|cpu]
  compare SCORE_A.json SCORE_B.json [...]       rates + CI per file, paired differences to the first file
  selftest                                       consistency checks (see below)
"""
import argparse, glob, json, math, os, sys, hashlib
import numpy as np

SF = os.environ.get("SPIRAL_FITTING_DIR", "spiral-fitting")
HELD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spiral_heldout_boxA.json")
HUMAN = {"rel": os.environ.get("LADDER_DIR", "layerjudge/") + "relative_windings.json",
         "abs": os.environ.get("LADDER_DIR", "layerjudge/") + "abs_winding.json"}
Z0, Z1 = 10496, 11008
RADIUS, UP, ACCEPT = 40.0, 6, 10.0


# ---------------------------------------------------------------- ladders
def load_ladders(ids=None):
    held = json.load(open(HELD, encoding="utf-8"))
    sens = held["sensitivity"]["ids"]
    main = set(held["main"]["ids"])
    docs = {k: json.load(open(v)) for k, v in HUMAN.items()}
    out = []
    for lid in (ids or sens):
        src, cid = lid.split(":")
        col = docs[src]["collections"][cid]
        pts = [(float(p["wind_a"]), p["p"]) for p in col["points"].values() if p.get("wind_a") is not None]
        pts.sort(key=lambda t: t[0])
        out.append(dict(id=lid, main=lid in main, P=np.array([p[1] for p in pts], float),
                        wind=np.array([p[0] for p in pts], float)))
    return out


# ---------------------------------------------------------------- assignment (PCU winding_at)
class SurfaceIndex:
    """grids: {winding: array (R, C, 3) in zyx, invalid = -1}"""

    def __init__(self, grids, zlo=Z0 - 60, zhi=Z1 + 60, bbox_xyz=None):
        from scipy.spatial import cKDTree
        self.grids = grids
        pts, wl, rl, cl = [], [], [], []
        for w, g in grids.items():
            valid = (g > 0).all(-1) & (g[..., 0] > zlo) & (g[..., 0] < zhi)
            if bbox_xyz is not None:
                lo, hi = bbox_xyz
                valid &= (g[..., 2] >= lo[0]) & (g[..., 2] <= hi[0]) & (g[..., 1] >= lo[1]) & (g[..., 1] <= hi[1])
            r, c = np.nonzero(valid)
            pts.append(g[r, c])
            wl.append(np.full(len(r), w)); rl.append(r); cl.append(c)
        self.pts = np.concatenate(pts) if pts else np.zeros((0, 3))
        self.w = np.concatenate(wl) if wl else np.zeros(0, int)
        self.r = np.concatenate(rl) if rl else np.zeros(0, int)
        self.c = np.concatenate(cl) if cl else np.zeros(0, int)
        self.tree = cKDTree(self.pts)

    def winding_at(self, p_zyx):
        """-> (winding or nan, distance)"""
        import scipy.ndimage as ndi
        idx = self.tree.query_ball_point(p_zyx, RADIUS)
        if not idx:
            return float("nan"), float("inf")
        idx = np.array(idx)
        d0 = np.linalg.norm(self.pts[idx] - p_zyx, axis=1)
        best_w, best_d = float("nan"), float("inf")
        for w in np.unique(self.w[idx]):
            m = self.w[idx] == w
            j = idx[m][np.argmin(d0[m])]
            g = self.grids[int(w)]
            r, c = int(self.r[j]), int(self.c[j])
            win = g[max(r - 2, 0):r + 3, max(c - 2, 0):c + 4 - 1]
            if (win <= 0).any():
                v = win.reshape(-1, 3)
                v = v[(v > 0).all(1)]
                d = np.linalg.norm(v - p_zyx, axis=1).min() if len(v) else float("inf")
            else:
                up = np.stack([ndi.zoom(win[..., k], UP, order=1) for k in range(3)], -1).reshape(-1, 3)
                d = np.linalg.norm(up - p_zyx, axis=1).min()
            if d < best_d:
                best_d, best_w = d, float(w)
        if best_d >= ACCEPT:
            return float("nan"), best_d
        return best_w, best_d


def slips_for_ladder(assigned, wind):
    dv = np.diff(assigned)
    dw = np.diff(wind)
    ok = np.isfinite(dv)
    if not ok.any():
        return 0, 0, np.zeros(len(dv), bool), ok
    med = np.nanmedian(dv)
    sg = np.sign(med) if np.sign(med) != 0 else 1.0
    slip = np.zeros(len(dv), bool)
    slip[ok] = (sg * dv[ok]) != dw[ok]
    return int(slip.sum()), int(ok.sum()), slip, ok


def score_ladders(ladders, index=None, assigned_override=None, constraint_pts=None):
    from scipy.spatial import cKDTree
    ctree = cKDTree(constraint_pts[:, [2, 1, 0]]) if constraint_pts is not None and len(constraint_pts) else None
    res = []
    for k, L in enumerate(ladders):
        P = L["P"]
        if assigned_override is not None:
            a = assigned_override[k]; dist = np.zeros(len(a))
        else:
            out = [index.winding_at(p[[2, 1, 0]]) for p in P]
            a = np.array([o[0] for o in out]); dist = np.array([o[1] for o in out])
        S, T, slip, ok = slips_for_ladder(a, L["wind"])
        rec = dict(id=L["id"], main=L["main"], S=S, T=T, n_pts=len(P), n_unassigned=int((~np.isfinite(a)).sum()),
                   assigned=[None if not np.isfinite(x) else int(x) for x in a], wind=L["wind"].tolist(),
                   pair_slip=slip.tolist(), pair_valid=ok.tolist())
        if ctree is not None:
            dc = ctree.query(P[:, [2, 1, 0]])[0]
            rec["pair_cdist"] = np.minimum(dc[:-1], dc[1:]).tolist()
        res.append(rec)
    return res


# ---------------------------------------------------------------- bootstrap
def rate(S, T):
    return S.sum() / max(T.sum(), 1)


def boot_ci(S, T, nboot=2000, seed=1):
    rng = np.random.default_rng(seed)
    n = len(S)
    vals = np.empty(nboot)
    for b in range(nboot):
        i = rng.integers(0, n, n)
        vals[b] = rate(S[i], T[i])
    return rate(S, T), np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def boot_diff(SA, TA, SB, TB, nboot=2000, seed=1):
    """B - A on the same resampled ladders."""
    rng = np.random.default_rng(seed)
    n = len(SA)
    d = np.empty(nboot)
    for b in range(nboot):
        i = rng.integers(0, n, n)
        d[b] = rate(SB[i], TB[i]) - rate(SA[i], TA[i])
    return rate(SB, TB) - rate(SA, TA), np.percentile(d, 2.5), np.percentile(d, 97.5), float((d >= 0).mean())


def arrays(score, only_main):
    recs = [r for r in score["ladders"] if (r["main"] or not only_main)]
    return np.array([r["S"] for r in recs]), np.array([r["T"] for r in recs]), recs


# ---------------------------------------------------------------- surfaces from checkpoint
def export_grids(ckpt, umbilicus, device="cuda", chunk=1 << 18):
    sys.path.insert(0, SF)
    import torch
    from pathlib import Path
    import flatten_spiral_checkpoint as F
    from checkpoint_io import load_checkpoint_cpu
    from sample_spiral import get_spiral_yxs
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    checkpoint = load_checkpoint_cpu(str(ckpt))
    config = F._checkpoint_config(checkpoint)
    with torch.inference_mode():
        model = F._build_model(checkpoint, config, Path(umbilicus), dev)
        transform = model.get_slice_to_spiral_transform()
        dr = model.get_dr_per_winding()
        z_begin, z_end = int(checkpoint["z_begin"]), int(checkpoint["z_end"])
        step = int(F._value(config, "step_size", 20))
        first = int(checkpoint.get("preview_first_winding", F._value(config, "first_winding", 10)))
        last = int(config.get("shell_outer_winding_idx") or int(config["model_gap_expander_num_windings"]) - 1)
        yxs_by = get_spiral_yxs(last + 1, dr, step, group_by_winding=True, device=str(dev))
        margin = int(config["model_flow_bounds_z_margin"])
        zv = torch.arange(z_begin - margin, z_end + margin, step, dtype=torch.float32, device=dev)
        grids = {}
        for w in range(first, last + 1):
            yxs = yxs_by[w]
            sp = torch.cat([zv[:, None, None].expand(-1, yxs.shape[0], 1), yxs[None].expand(zv.shape[0], -1, 2)], -1)
            flat = sp.reshape(-1, 3)
            sc = torch.cat([transform.inv(flat[s:s + chunk]).cpu() for s in range(0, flat.shape[0], chunk)]).reshape_as(sp)
            sc[(sc[..., 0] < z_begin) | (sc[..., 0] >= z_end)] = -1.0
            grids[w] = sc.numpy().astype(np.float32)
        info = dict(dr_per_winding=float(dr), first=first, last=last, step=step, z_begin=z_begin, z_end=z_end)
    return grids, info


def find_ckpt(path):
    if os.path.isfile(path):
        return path
    c = sorted(glob.glob(os.path.join(path, "*", "checkpoint_fitted.ckpt")) + glob.glob(os.path.join(path, "checkpoint_fitted.ckpt")))
    if not c:
        raise FileNotFoundError(f"no checkpoint_fitted.ckpt under {path}")
    return c[-1]


def constraint_points(path):
    d = json.load(open(path))
    return np.array([pt["p"] for c in d["collections"].values() for pt in c["points"].values()], float)


def cmd_score(a):
    ckpt = find_ckpt(a.run)
    rundir = os.path.dirname(ckpt)
    umb = a.umbilicus
    cache = os.path.join(rundir, "grids_boxA.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        grids = {int(k[1:]): z[k] for k in z.files if k.startswith("w")}
        info = json.loads(str(z["info"]))
    else:
        grids, info = export_grids(ckpt, umb, a.device)
        np.savez(cache, info=json.dumps(info), **{f"w{k:03d}": v for k, v in grids.items()})
    ladders = load_ladders()
    allp = np.concatenate([L["P"] for L in ladders])
    bbox = (allp.min(0)[:2] - 150, allp.max(0)[:2] + 150)  # x,y
    index = SurfaceIndex(grids, bbox_xyz=((bbox[0][0], bbox[0][1]), (bbox[1][0], bbox[1][1])))
    cp = constraint_points(a.constraints) if a.constraints else None
    recs = score_ladders(ladders, index=index, constraint_pts=cp)
    out = dict(run=a.run, ckpt=ckpt, info=info, constraints=a.constraints,
               constraints_sha256=hashlib.sha256(open(a.constraints, "rb").read()).hexdigest() if a.constraints else None,
               n_surface_vertices_in_index=int(len(index.pts)), ladders=recs)
    json.dump(out, open(a.out, "w"))
    summarize(out)


def summarize(score, label=None):
    for name, only_main in (("main", True), ("sens", False)):
        S, T, recs = arrays(score, only_main)
        r, lo, hi = boot_ci(S, T)
        un = sum(x["n_unassigned"] for x in recs); npt = sum(x["n_pts"] for x in recs)
        print(f"{label or score['run']} [{name}] ladders {len(S)}  slips {S.sum()}/{T.sum()} valid pairs "
              f"(of {sum(x['n_pts']-1 for x in recs)})  rate {100*r:.1f}%  95% [{100*lo:.1f}, {100*hi:.1f}]  unassigned pts {un}/{npt}")


def cmd_compare(a):
    scores = [json.load(open(f)) for f in a.files]
    for f, s in zip(a.files, scores):
        summarize(s, os.path.basename(f))
    base = scores[0]
    for f, s in zip(a.files[1:], scores[1:]):
        for name, only_main in (("main", True), ("sens", False)):
            SA, TA, _ = arrays(base, only_main); SB, TB, _ = arrays(s, only_main)
            d, lo, hi, pw = boot_diff(SA, TA, SB, TB)
            print(f"paired diff ({os.path.basename(f)} - {os.path.basename(a.files[0])}) [{name}]: {100*d:+.1f} pts  [{100*lo:+.1f}, {100*hi:+.1f}]  P(diff>=0)={pw:.3f}")


# ---------------------------------------------------------------- selftest
def synthetic_grids(dr=60.0, wmin=10, wmax=40, cx=4800.0, cy=3400.0):
    sys.path.insert(0, SF)
    grids = {}
    zs = np.arange(Z0 - 60 - 40, Z1 + 60 + 40, 20.0)
    for w in range(wmin, wmax + 1):
        n = int(2 * math.pi * (w + 0.5) * dr / 20.0) + 1
        th = w * 2 * math.pi + np.arange(n) * (20.0 / ((w + 0.5) * dr))
        rr = dr * th / (2 * math.pi)
        y = cy + np.sin(th) * rr; x = cx + np.cos(th) * rr
        g = np.empty((len(zs), n, 3), np.float32)
        g[..., 0] = zs[:, None]; g[..., 1] = y[None]; g[..., 2] = x[None]
        g[(g[..., 0] < 0)] = -1
        grids[w] = g
    return grids


def cmd_selftest(a):
    held = load_ladders()
    print("== 1. human ladders' own wind_a as 'fit output' (expect 0 slips)")
    for name, sel in (("main", [L for L in held if L["main"]]), ("sens", held)):
        recs = score_ladders(sel, assigned_override=[L["wind"] for L in sel])
        S = np.array([r["S"] for r in recs]); T = np.array([r["T"] for r in recs])
        print(f"   {name}: ladders {len(sel)} slips {S.sum()} / {T.sum()} pairs  rate {100*rate(S,T):.1f}%")
    print("== 2. shuffled winding numbers within each ladder, 200 draws (expect high)")
    rng = np.random.default_rng(0)
    for name, sel in (("main", [L for L in held if L["main"]]), ("sens", held)):
        rates = []
        for _ in range(200):
            asg = [rng.permutation(L["wind"]) for L in sel]
            recs = score_ladders(sel, assigned_override=asg)
            S = np.array([r["S"] for r in recs]); T = np.array([r["T"] for r in recs])
            rates.append(rate(S, T))
        print(f"   {name}: mean {100*np.mean(rates):.1f}%  5-95% [{100*np.percentile(rates,5):.1f}, {100*np.percentile(rates,95):.1f}]")
    print("== 3. random integer windings 10..40 per point, 200 draws")
    rates = []
    for _ in range(200):
        asg = [rng.integers(10, 41, len(L["wind"])).astype(float) for L in held]
        recs = score_ladders(held, assigned_override=asg)
        S = np.array([r["S"] for r in recs]); T = np.array([r["T"] for r in recs])
        rates.append(rate(S, T))
    print(f"   sens: mean {100*np.mean(rates):.1f}%")
    print("== 4. assignment code on a synthetic Archimedean spiral (dr 60, windings 10..40)")
    grids = synthetic_grids()
    idx = SurfaceIndex(grids)
    cx, cy, dr = 4800.0, 3400.0, 60.0
    synth, truth = [], []
    for phi in (0.3, 1.7, 3.1, 4.4, 5.9):
        for z in (10600.0, 10900.0):
            P, W = [], []
            for w in range(14, 21):
                th = w * 2 * math.pi + phi + 0.37  # off the vertex lattice
                rr = dr * th / (2 * math.pi)
                P.append([cx + math.cos(th) * rr, cy + math.sin(th) * rr, z]); W.append(float(w))
            synth.append(dict(id=f"syn{phi}_{z}", main=True, P=np.array(P), wind=np.array(W) - 14 + 3))
    recs = score_ladders(synth, index=idx)
    S = sum(r["S"] for r in recs); T = sum(r["T"] for r in recs); un = sum(r["n_unassigned"] for r in recs)
    ok = all(r["assigned"] == [int(w) + 11 for w in r["wind"]] for r in recs)
    print(f"   exact-on-surface ladders: slips {S}/{T}, unassigned {un}, recovered winding index exactly: {ok}")
    # shift every point 15 voxels off the surface (radially): should be unassigned or still right
    shifted = []
    for L in synth:
        P = L["P"].copy(); P[:, 0] += 25.0
        shifted.append(dict(L, P=P))
    recs = score_ladders(shifted, index=idx)
    print(f"   points moved 25 voxels off any surface: unassigned {sum(r['n_unassigned'] for r in recs)}/{sum(r['n_pts'] for r in recs)} (expect most)")
    # one-winding slip injected into the fit: relabel windings by +1 for outer half -> slips
    slipped = [dict(L, wind=np.where(np.arange(len(L['wind'])) >= 4, L['wind'] + 1, L['wind'])) for L in synth]
    recs = score_ladders(slipped, index=idx)
    S = sum(r["S"] for r in recs); T = sum(r["T"] for r in recs)
    print(f"   ladder claims one extra wrap at pair 4: slips {S}/{T} (expect {len(synth)}/{T})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("score"); s.add_argument("run"); s.add_argument("--out", required=True)
    s.add_argument("--constraints"); s.add_argument("--device", default="cuda")
    s.add_argument("--umbilicus", default="D:/vesuvius_downstream/p4/ds/umbilicus.json")
    c = sp.add_parser("compare"); c.add_argument("files", nargs="+")
    sp.add_parser("selftest")
    a = ap.parse_args()
    {"score": cmd_score, "compare": cmd_compare, "selftest": cmd_selftest}[a.cmd](a)
