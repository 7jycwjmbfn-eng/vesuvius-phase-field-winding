"""surfkit_8c: common "surfaces -> ruler A -> deliverables" chain for W2 (Paris4), any method (LG, SPIRAL, Will, ...).

SURF format (npz): for i = 0..n-1 a key "xyz_<i>" = float32 [H, W, 3], level-3 (19.2 um) zyx, NaN = invalid; optional "names" (str [n]).
Grid neighbours must be neighbours on the sheet (any spacing; ~1-4 L3 voxels is fine). One array = one connected sheet piece
that the method claims is ONE papyrus sheet (it may wrap more than one turn). Tifxyz directories (Will style, level-2 coords,
x/y/z.tif, invalid <= 0) are read too.

Subcommands
  convert SRC OUT.npz              SRC = SURF npz or a directory of tifxyz dirs  -> SURF npz
  area SRC                         raw area (quads inside W2 z 5000-5500), largest 4-connected piece per surface;
                                   + label-free clean area, same rules as group_check_arc.py + contiguous_area.py
  jump SRC                         same-sheet check on human-verified patch samples (patch_samples_ua.npz), same rule as compare_will.py
  tifxyz SRC OUTDIR                write one tifxyz dir per surface (level-2 coords, -1 = invalid)
  xsec SRC_A SRC_B Z Y0 Y1 X0 X1 OUT.png   paired cross-section on the L3 CT slice Z (A left, B right)
"""
import glob, json, os, sys
import numpy as np
import tifffile
from numba import njit
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

T = "E:/vesuvius_downstream_tmp/fiberA/"
VOX_CM = 19.2e-4                                   # level-3 voxel edge in cm
ZSPLITS = {"all": (5000.0, 5500.0), "dev": (5000.0, 5250.0), "test": (5250.0, 5500.0)}
ZW = ZSPLITS[os.environ.get("ZSPLIT", "all")]     # W2 in level 3; ZSPLIT=dev / test = half windows
CELL = 26.04                                       # 0.5 mm in L3 voxels (contiguous_area.py)
_cp = sorted(json.load(open(r"D:\vesuvius_downstream\p4\ds\umbilicus.json"))["control_points"], key=lambda c: c["z"])
UZ, UX, UY = (np.array([c[t] for c in _cp], float) / 2 for t in ("z", "x", "y"))


def load(src):
    """-> list of (name, xyz [H, W, 3] float64 L3 zyx with NaN)"""
    out = []
    if src.endswith(".npz"):
        d = np.load(src); n = len([k for k in d.files if k.startswith("xyz_")])
        names = list(d["names"]) if "names" in d.files else [f"s{i}" for i in range(n)]
        for i in range(n):
            out.append((str(names[i]), d[f"xyz_{i}"].astype(np.float64)))
        return out
    for dd in sorted(glob.glob(os.path.join(src, "*"))):
        if os.path.exists(os.path.join(dd, "x.tif")): out += load_one(dd)
    return out


def load_one(dd):
    X, Y, Z = (tifffile.imread(os.path.join(dd, f"{c}.tif")).astype(np.float64) for c in "xyz")
    P = np.stack([Z, Y, X], -1) / 2.0
    P[~((X > 0) & (Y > 0) & (Z > 0))] = np.nan
    return [(os.path.basename(dd), P)]


def save(surfs, path):
    np.savez_compressed(path, names=np.array([n for n, _ in surfs]), **{f"xyz_{i}": s.astype(np.float32) for i, (_, s) in enumerate(surfs)})


@njit(cache=True)
def unwrap_grid(th, valid):
    H, W = th.shape
    out = np.zeros((H, W)); comp = np.full((H, W), -1, np.int64); stack = np.zeros((H * W, 2), np.int64); nc = 0
    for i0 in range(H):
        for j0 in range(W):
            if not valid[i0, j0] or comp[i0, j0] >= 0: continue
            comp[i0, j0] = nc; out[i0, j0] = th[i0, j0]; stack[0, 0] = i0; stack[0, 1] = j0; sp = 1
            while sp > 0:
                sp -= 1; i = stack[sp, 0]; j = stack[sp, 1]
                for q in range(4):
                    ii = i + (q == 0) - (q == 1); jj = j + (q == 2) - (q == 3)
                    if ii < 0 or jj < 0 or ii >= H or jj >= W: continue
                    if not valid[ii, jj] or comp[ii, jj] >= 0: continue
                    d = th[ii, jj] - th[i, j]; d = (d + np.pi) % (2 * np.pi) - np.pi
                    out[ii, jj] = out[i, j] + d; comp[ii, jj] = nc; stack[sp, 0] = ii; stack[sp, 1] = jj; sp += 1
            nc += 1
    return out, comp


def theta(P):
    return np.arctan2(P[..., 1] - np.interp(P[..., 0], UZ, UY), P[..., 2] - np.interp(P[..., 0], UZ, UX))


def radius(P):
    return np.hypot(P[..., 1] - np.interp(P[..., 0], UZ, UY), P[..., 2] - np.interp(P[..., 0], UZ, UX))


def quad_area(P):
    """area (cm2) of each grid quad with 4 valid corners inside W2 z; returns [H-1, W-1] (0 = no quad)"""
    a, b, c, d = P[:-1, :-1], P[:-1, 1:], P[1:, 1:], P[1:, :-1]
    ok = np.isfinite(a).all(-1) & np.isfinite(b).all(-1) & np.isfinite(c).all(-1) & np.isfinite(d).all(-1)
    zc = np.nan_to_num((a[..., 0] + b[..., 0] + c[..., 0] + d[..., 0]) / 4, nan=-1)    # quad centre inside W2 (reproduces Will 4.43)
    ok &= (zc >= ZW[0]) & (zc < ZW[1])
    A = 0.5 * np.linalg.norm(np.cross(c - a, d - b), axis=-1) * VOX_CM ** 2
    return np.where(ok, np.nan_to_num(A), 0.0)


def clean_area(P):
    """group_check_arc + contiguous_area on one surface: dup / stack points, 0.5 mm unrolled cells, largest 8-connected clean piece."""
    valid = np.isfinite(P).all(-1) & (np.nan_to_num(P[..., 0], nan=-1) >= ZW[0]) & (np.nan_to_num(P[..., 0], nan=1e9) < ZW[1])
    if valid.sum() < 50: return 0.0, 0.0, 0.0
    Th, comp = unwrap_grid(np.nan_to_num(theta(P)), valid)
    ij = np.argwhere(valid); X = P[valid]; T_ = Th[valid]; R = radius(X); Z = X[:, 0]; cc = comp[valid]
    best = tot = 0.0; badfrac = []
    for c in np.unique(cc):
        m = cc == c
        if m.sum() < 50: continue
        x, t, r, z = X[m], T_[m], R[m], Z[m]; bad = np.zeros(len(x), bool)
        pr = cKDTree(x).query_pairs(2.0, output_type="ndarray")
        if len(pr): bad[pr[np.abs(t[pr[:, 0]] - t[pr[:, 1]]) > np.pi].ravel()] = True                  # dup
        tb = np.floor(t / 0.02).astype(np.int64); ub, binv = np.unique(tb, return_inverse=True)
        rmed = np.bincount(binv, r) / np.bincount(binv)
        full = np.arange(ub.min(), ub.max() + 1); rf = np.interp(full, ub, rmed); sc = np.r_[0, np.cumsum(rf[:-1] * 0.02)]
        U = np.interp(t / 0.02 - ub.min(), np.arange(len(full)), sc)
        pr2 = cKDTree(np.stack([U / 4.0, z / 4.0], 1)).query_pairs(1.0, output_type="ndarray")
        if len(pr2): bad[pr2[np.linalg.norm(x[pr2[:, 0]] - x[pr2[:, 1]], axis=1) > 8].ravel()] = True     # stack
        ui = np.floor(U / CELL).astype(np.int64); zi = np.floor(z / CELL).astype(np.int64); ui -= ui.min(); zi -= zi.min()
        occ = np.zeros((ui.max() + 1, zi.max() + 1), np.int8); badc = np.zeros_like(occ)
        occ[ui, zi] = 1; np.maximum.at(badc, (ui, zi), bad.astype(np.int8))
        good = (occ == 1) & (badc == 0); lb, n = ndi.label(good, structure=np.ones((3, 3)))
        if n: best = max(best, np.bincount(lb.ravel())[1:].max() * 0.0025)
        tot += good.sum() * 0.0025; badfrac.append(bad.mean())
    return best, tot, float(np.mean(badfrac)) if badfrac else 0.0


def align_components(Th, comp, gap=3):
    """Single-SID mode: shift each grid component's unwrapped angle by a multiple of 2 pi so that it agrees with an already
    aligned component across a NaN gap of <= gap cells. Returns aligned Th and a group id per pixel (components that never come
    within gap cells of the main group start their own group)."""
    Th = Th.copy(); grp = np.full(comp.shape, -1, np.int64)
    ids, cnt = np.unique(comp[comp >= 0], return_counts=True); left = set(ids[np.argsort(-cnt)].tolist()); g = 0
    order = list(ids[np.argsort(-cnt)])
    while left:
        seed = next(c for c in order if c in left); left.discard(seed); al = comp == seed; grp[al] = g
        while True:
            dil = ndi.binary_dilation(al, structure=np.ones((2 * gap + 1, 2 * gap + 1), bool))
            cand = set(np.unique(comp[dil & (comp >= 0) & ~al]).tolist()) & left
            if not cand: break
            _, (iy, ix) = ndi.distance_transform_edt(~al, return_indices=True)
            for c in cand:
                pb = np.argwhere((comp == c) & dil); dd = np.hypot(pb[:, 0] - iy[pb[:, 0], pb[:, 1]], pb[:, 1] - ix[pb[:, 0], pb[:, 1]])
                b = pb[np.argmin(dd)]; a = (iy[b[0], b[1]], ix[b[0], b[1]])
                Th[comp == c] += 2 * np.pi * np.round((Th[a] - Th[b[0], b[1]]) / (2 * np.pi))
                left.discard(c); m = comp == c; al |= m; grp[m] = g
        g += 1
    return Th, grp


def area_rows(surfs):
    rows = []
    for name, P in surfs:
        A = quad_area(P); lb, n = ndi.label(A > 0)
        piece = np.bincount(lb.ravel(), A.ravel())[1:].max() if n else 0.0
        cbest, ctot, bf = clean_area(P) if not os.environ.get("NOCLEAN") else (np.nan, np.nan, np.nan)
        rows.append((name, A.sum(), piece, cbest, ctot, bf))
    return rows


def cmd_area(src):
    surfs = load(src); rows = area_rows(surfs)
    for name, P in surfs:
        A = quad_area(P); lb, n = ndi.label(A > 0)
        piece = np.bincount(lb.ravel(), A.ravel())[1:].max() if n else 0.0
        cbest, ctot, bf = clean_area(P)
        rows.append((name, A.sum(), piece, cbest, ctot, bf))
    rows.sort(key=lambda r: -r[2])
    for r in rows[:15]:
        print(f"  {r[0]:>14s}  raw {r[1]:6.2f} cm2  largest raw piece {r[2]:5.2f}  | clean: largest contiguous {r[3]:5.2f}  total {r[4]:6.2f}  bad pts {r[5] * 100:4.1f}%")
    raw = np.array([r[1] for r in rows]); pc = np.array([r[2] for r in rows]); cb = np.array([r[3] for r in rows])
    print(f"z {ZW[0]:.0f}-{ZW[1]:.0f}; surfaces {len(rows)}: array area sum: total {raw.sum():.2f} cm2, median {np.median(raw):.2f}, max {raw.max():.2f} ({rows[int(np.argmax(raw))][0]}); "
          f"largest single 4-connected piece {pc.max():.2f} ({rows[int(np.argmax(pc))][0]}); largest clean contiguous {np.nanmax(cb) if np.isfinite(cb).any() else float('nan'):.2f} cm2")


def cmd_jump(src):
    return jump(load(src))


def jump(surfs, mode=None, quiet=False):
    """mode: "piece" (default, SID = array x 100000 + grid component) or "array" (single SID per array after 2 pi alignment
    across gaps <= 3 cells). Returns per-array dict name -> [on-samples, switch samples] and totals."""
    mode = mode or os.environ.get("SIDMODE", "piece")
    S = np.load(T + "patch_samples_ua.npz"); p, tu, cid, nrm = S["pts"].astype(np.float64), S["tu"], S["cid"], S["nrm"].astype(np.float64)
    Q, SID, ANG = [], [], []; ngroups = []
    for k, (name, P) in enumerate(surfs):
        valid = np.isfinite(P).all(-1)
        Th, comp = unwrap_grid(np.nan_to_num(theta(P)), valid)
        if mode == "array":
            Th, comp = align_components(Th, comp); ngroups.append(int(comp.max()) + 1 if valid.any() else 0)
        Q.append(P[valid]); ANG.append(Th[valid]); SID.append(k * 100000 + comp[valid])
    Q, SID, ANG = np.concatenate(Q), np.concatenate(SID), np.concatenate(ANG)
    box = (np.array([ZW[0], 576, 896]), np.array([ZW[1] - 1e-6, 1792, 2432])) if os.environ.get("BOX", "fixed") == "fixed" else (Q.min(0), Q.max(0))
    inb = np.all((p >= box[0]) & (p <= box[1]), 1)
    d, j = cKDTree(Q).query(p, distance_upper_bound=3.5, workers=-1); hit = np.isfinite(d) & inb; j = np.minimum(j, len(Q) - 1)
    ii = np.nonzero(inb)[0]; o = np.argsort(cid[ii], kind="stable"); ii = ii[o]; bnd = np.r_[0, np.nonzero(np.diff(cid[ii]))[0] + 1, len(ii)]
    A_, B_ = [], []
    for a0, a1 in zip(bnd[:-1], bnd[1:]):
        s = ii[a0:a1]
        if len(s) < 2: continue
        ia, ib = np.triu_indices(len(s), 1); A_.append(s[ia]); B_.append(s[ib])
    A, B = np.concatenate(A_), np.concatenate(B_); L = np.linalg.norm(p[B] - p[A], axis=1); dtu = np.abs(tu[B] - tu[A])
    given = hit[A] & hit[B] & (SID[j[A]] == SID[j[B]])
    err = ((ANG[j[B]] - ANG[j[A]]) - (tu[B] - tu[A])) / (2 * np.pi); correct = given & (np.abs(err) < 0.5)
    if not quiet:
        print(f"mode {mode}; box z {box[0][0]:.0f}-{box[1][0]:.0f} y {box[0][1]:.0f}-{box[1][1]:.0f} x {box[0][2]:.0f}-{box[1][2]:.0f}; samples in box {inb.sum()}, covered {hit.sum()}; pairs {len(A)}")
        if mode == "array": print(f"  arrays with > 1 alignment group (some piece never within 3 cells of the rest): {sum(g > 1 for g in ngroups)} of {len(ngroups)}")
        for lo, hi in [(5, 100), (100, 200), (200, 400), (400, 800), (800, 5000)]:
            q = (L >= lo) & (L < hi)
            print(f"  {lo:>4d}-{hi:<5d} pairs {q.sum():8d}  judged {given[q].mean() * 100:5.1f}%  correct among judged {correct[q].sum() / max(given[q].sum(), 1) * 100:5.1f}%")
        q = dtu >= np.pi
        print(f"  >= half turn apart along the patch: pairs {q.sum()}, judged {given[q].sum()}, correct {correct[q].sum() / max(given[q].sum(), 1) * 100:.1f}%")
    # sheet-switch check: a surface piece that lies ON a verified patch at sample a (<= 3.5 vox) and runs 5-20 vox beside the
    # same patch at sample b with the same angular progress (|dTheta - dtu| < pi) has moved to a neighbouring sheet = switch.
    # (one full turn later would show |dTheta - dtu| ~ 2 pi and is not counted.)
    pi_ = np.nonzero(inb)[0]; tree = cKDTree(Q)
    nb = tree.query_ball_point(p[pi_], 20.0, workers=-1)
    rec = {}                                                             # (cid, piece) -> list of (sample, dist, Theta)
    for s_, lst in zip(pi_, nb):
        if not lst: continue
        lst = np.asarray(lst); v_ = Q[lst] - p[s_]; dd = np.linalg.norm(v_, axis=1); sid = SID[lst]
        dep = np.abs(v_ @ nrm[s_]); lat = np.sqrt(np.maximum(dd ** 2 - dep ** 2, 0))
        keep = (dd <= 3.5) | (lat < 3.0)                                 # beside = offset along the patch normal, not an edge
        lst, dd, sid = lst[keep], dd[keep], sid[keep]
        if not len(lst): continue
        o = np.lexsort((dd, sid)); sid, dd, lst = sid[o], dd[o], lst[o]; first = np.r_[True, sid[1:] != sid[:-1]]
        for k_, d_, l_ in zip(sid[first], dd[first], lst[first]):
            rec.setdefault((cid[s_], k_), []).append((s_, d_, ANG[l_]))
    groups = sw_groups = on_tot = off_tot = sw_tot = 0; dump = []; per = {n: [0, 0] for n, _ in surfs}
    for (c_, k_), v in rec.items():
        v = np.array(v); on = v[v[:, 1] <= 3.5]; off = v[(v[:, 1] > 5) & (v[:, 1] <= 20)]
        if len(on) < 3: continue
        groups += 1; on_tot += len(on); off_tot += len(off); per[surfs[int(k_) // 100000][0]][0] += len(on)
        if len(off) == 0: continue
        sa = on[:, 0].astype(int); sb = off[:, 0].astype(int)
        jn = cKDTree(p[sa]).query(p[sb])[1]
        e = ((off[:, 2] - on[jn, 2]) - (tu[sb] - tu[sa[jn]])) / (2 * np.pi)
        sw = np.abs(e) < 0.5; nsw = int(sw.sum()); sw_tot += nsw; sw_groups += nsw >= 3; per[surfs[int(k_) // 100000][0]][1] += nsw
        if nsw: dump.append(np.stack([np.full(nsw, c_), np.full(nsw, k_), sb[sw], sa[jn][sw], off[sw, 1]], 1))
    if os.environ.get("DUMP") and dump:                                   # columns: patch cid, surface SID, sample b (beside), sample a (on), dist b
        np.save(os.environ["DUMP"], np.concatenate(dump)); np.save(os.environ["DUMP"].replace(".npy", "_names.npy"), np.array([n for n, _ in surfs]))
    print(f"  sheet switch: (patch, surface) pairs with >= 3 on-samples {groups}; with >= 3 switch samples {sw_groups} ({sw_groups / max(groups, 1) * 100:.1f}%); "
          f"switch samples / on-samples {sw_tot}/{on_tot} = {sw_tot / max(on_tot, 1) * 100:.1f}% (beside-but-one-turn-away samples not counted; off total {off_tot})")
    return per, (groups, sw_groups, on_tot, sw_tot)


def cmd_tifxyz(src, outdir):
    for name, P in load(src):
        d = os.path.join(outdir, name); os.makedirs(d, exist_ok=True)
        P2 = np.where(np.isfinite(P), P * 2.0, -1.0).astype(np.float32)
        for c, k in (("z", 0), ("y", 1), ("x", 2)): tifffile.imwrite(os.path.join(d, f"{c}.tif"), P2[..., k])
        v = np.isfinite(P).all(-1); step = np.nanmedian(np.linalg.norm(np.diff(P, axis=1), axis=-1)) * 2.0
        lo, hi = P[v].min(0) * 2, P[v].max(0) * 2
        json.dump({"area_cm2": float(quad_area(P).sum()), "bbox": [[lo[2], lo[1], lo[0]], [hi[2], hi[1], hi[0]]], "format": "tifxyz",
                   "scale": [1.0 / step, 1.0 / step], "type": "seg", "uuid": name, "source": "surfkit_8c"}, open(os.path.join(d, "meta.json"), "w"), indent=2)
    print("wrote", outdir)


def cmd_xsec(a, b, Z, y0, y1, x0, x1, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ORG = (4992, 640, 896); ct = np.load(T + "ctW2_L3.npy", mmap_mode="r")
    img = np.asarray(ct[int(Z) - ORG[0], int(y0) - ORG[1]:int(y1) - ORG[1], int(x0) - ORG[2]:int(x1) - ORG[2]])
    fig, ax = plt.subplots(1, 2, figsize=(22, 11 * (y1 - y0) / max(x1 - x0, 1) + 1))
    for k, (src, a_) in enumerate(((a, ax[0]), (b, ax[1]))):
        a_.imshow(img, cmap="gray", extent=[x0, x1, y1, y0]); n = 0
        for i, (name, P) in enumerate(load(src)):
            m = np.isfinite(P).all(-1) & (np.abs(np.nan_to_num(P[..., 0], nan=-1e9) - Z) < 0.6)
            q = P[m]; q = q[(q[:, 1] >= y0) & (q[:, 1] < y1) & (q[:, 2] >= x0) & (q[:, 2] < x1)]
            if len(q): a_.scatter(q[:, 2], q[:, 1], s=1.5, color=plt.cm.tab20(i % 20)); n += 1
        a_.set_title(f"{os.path.basename(src.rstrip('/'))}: {n} surfaces cross z={Z:.0f}")
    plt.tight_layout(); plt.savefig(out, dpi=60); print("fig", out)


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "convert": save(load(sys.argv[2]), sys.argv[3]); print("saved", sys.argv[3])
    elif c == "area": cmd_area(sys.argv[2])
    elif c == "jump": cmd_jump(sys.argv[2])
    elif c == "tifxyz": cmd_tifxyz(sys.argv[2], sys.argv[3])
    elif c == "xsec": cmd_xsec(sys.argv[2], sys.argv[3], *(float(v) for v in sys.argv[4:9]), sys.argv[9])
