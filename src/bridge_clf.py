"""Bridge-edge classifier for the psi level-set pieces (box A, pieces_v6g.npz).
Bridge edge (bigeval.py rule): both ends have a truth turn number and n[b]-n[a] != -rint((th[b]-th[a])/2pi).
Features come from the network side only (v6 psi / conf, lasagna, CT, surface prediction, mesh geometry); the truth is used for labels and scoring only.

Stages (run one per command, each keeps memory < 2 GB; scratch files in E:/vesuvius_inv_tmp/bridge_clf/):
  python bridge_clf.py unpack       stream the npz members to plain .npy (mmap-able)
  python bridge_clf.py truth        per-vertex truth turn n and wrapped azimuth th (cached)
  python bridge_clf.py edges        edge list, face->edge map, bridge labels (the bigeval rule)
  python bridge_clf.py baseline     uncut pieces through the evaluator: must give bigeval's 89.6% / 60.9 cm2
  python bridge_clf.py fields [gpsi|vort|ctsp]   derived D=2 volumes (|grad psi|, vortex distance and density, CT mean/std, sp mean)
  python bridge_clf.py feats        per-edge feature matrix (38 features, float16)
  python bridge_clf.py siglabel | minlabel      alternative labels (significant seams; minority-region)
  python bridge_clf.py univariate   single-feature AUC table
  [LABEL=bridge|sig|minor QUICK=0|1 MAX_ITER NEG_RATE NFOLD] python bridge_clf.py train MODE SET TAG
                                   grouped 5-fold CV (MODE piece|sector), AUC / PR, out-of-fold probabilities p_oof_TAG.npy, validation probabilities per fold
  python bridge_clf.py cut METHOD TAG DILATE   cut simulation: METHOD model|conf|bdist|vdist
  python bridge_clf.py select TAG DILATE       operating points whose thresholds come from the validation pieces of each fold
  python bridge_clf.py importance TAG FOLD     permutation importance
  python bridge_clf.py diag_profile | diag_truth | diag_events DUMP.npy TAG | diag_misc tile|ct|second|oracle     diagnostics of the labels (see the report)
  python bridge_clf.py plot OUT.png "label=cut_*.json" ...
  python bridge_clf.py export METHOD TAG DILATE THRESHOLD OUT.npz     SURF export of the cut pieces (ruler: a2/eval_surfs.py)
Convention: a face is retained when none of its three edges is cut; pieces = connected components of the retained faces; retained area counts pieces >= 0.02 cm2;
purity is scored as in bigeval (pieces >= 0.05 cm2 with >= 30% truth coverage, area-weighted share of the largest bridge-free sub-piece).
"""
import os, sys, time, json, shutil, zipfile
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

B = "E:/vesuvius_inv_tmp/bridge_clf/"
SRC_NPZ = "E:/vesuvius_big_hot/pieces_v6g.npz"
ORG = np.array([10496.0, 1920.0, 3328.0])               # box A origin, L2 absolute zyx
VOX_CM = 9.6e-4
AMIN_KEEP = 0.02                                          # extraction minimum piece area (cm2)
AMIN_SCORE = 0.05; COV_MIN = 0.3                          # bigeval scoring set
PSI_F = "E:/vesuvius_big_hot/pred_v6_psi.npy"; CONF_F = "E:/vesuvius_big_hot/pred_v6_conf.npy"
COS_F = "E:/vesuvius_big_hot/las/cos.npy"; GM_F = "E:/vesuvius_big_hot/las/gm.npy"
CT_F = "D:/vesuvius_big_hot/ct_big.npy"; SP_F = "D:/vesuvius_big_hot/sp_big.npy"
FD = B + "fields/"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_meta():
    m = {k: np.load(B + k + ".npy") for k in ("voff", "foff", "area", "k")}
    m["V"] = np.load(B + "V.npy", mmap_mode="r"); m["F"] = np.load(B + "F.npy", mmap_mode="r")
    return m


def piece_VF(m, i):
    V = np.asarray(m["V"][m["voff"][i]:m["voff"][i + 1]]).astype(np.float64)
    F = np.asarray(m["F"][m["foff"][i]:m["foff"][i + 1]]).astype(np.int64)
    return V, F


def face_area_cm2(V, F):
    return 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1) * VOX_CM ** 2


def edges_of(F, nv):
    """unique undirected edges [ne,2] (a<b) and the edge index of the three face sides [nf,3] (side j joins corner j and j+1)"""
    nf = len(F)
    a = np.concatenate([F[:, 0], F[:, 1], F[:, 2]]); b = np.concatenate([F[:, 1], F[:, 2], F[:, 0]])
    key = np.minimum(a, b) * nv + np.maximum(a, b); uk, inv = np.unique(key, return_inverse=True)
    return np.stack([uk // nv, uk % nv], 1), inv.reshape(3, nf).T.astype(np.int32)


def umbilicus():
    cp = sorted(json.load(open(r"D:\vesuvius_downstream\p4\ds\umbilicus.json"))["control_points"], key=lambda c: c["z"])
    return tuple(np.array([c[t] for c in cp], float) for t in ("z", "y", "x"))


# ---------------------------------------------------------------- stage: unpack / truth
def stage_unpack():
    os.makedirs(B, exist_ok=True)
    z = zipfile.ZipFile(SRC_NPZ)
    for n in ["V.npy", "F.npy", "voff.npy", "foff.npy", "k.npy", "area.npy"]:
        with z.open(n) as s, open(B + n, "wb") as d: shutil.copyfileobj(s, d, 1 << 24)
        log("unpacked", n)


def stage_truth():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import isosurf as iso
    m = load_meta(); nvt = int(m["voff"][-1]); n_out = np.lib.format.open_memmap(B + "n.npy", mode="w+", dtype=np.float32, shape=(nvt,))
    th_out = np.lib.format.open_memmap(B + "th.npy", mode="w+", dtype=np.float32, shape=(nvt,)); t0 = time.time()
    for i in range(len(m["area"])):
        V, _ = piece_VF(m, i); sl = slice(m["voff"][i], m["voff"][i + 1])
        n_out[sl] = iso.turn_numbers(V); th_out[sl] = iso.theta_wrapped(V)
        if i % 50 == 0: log(f"piece {i}/{len(m['area'])}  {time.time() - t0:.0f} s")
    n_out.flush(); th_out.flush(); log("truth done")


# ---------------------------------------------------------------- stage: edges
def stage_edges():
    m = load_meta(); n_all = np.load(B + "n.npy", mmap_mode="r"); th_all = np.load(B + "th.npy", mmap_mode="r"); P = len(m["area"])
    fab = open(B + "e_ab.bin", "wb"); flab = open(B + "e_lab.bin", "wb"); ffe = open(B + "f_fe.bin", "wb"); eoff = [0]; t0 = time.time()
    for i in range(P):
        V, F = piece_VF(m, i); nv = len(V); n = np.asarray(n_all[m["voff"][i]:m["voff"][i + 1]]).astype(np.float64); th = np.asarray(th_all[m["voff"][i]:m["voff"][i + 1]]).astype(np.float64)
        E, fe = edges_of(F, nv); a, b = E[:, 0], E[:, 1]
        have = np.isfinite(n[a]) & np.isfinite(n[b]); bridge = have & ((n[b] - n[a]) != -np.rint((th[b] - th[a]) / (2 * np.pi)))
        lab = np.where(have, bridge.astype(np.int8), np.int8(-1))
        fab.write(E.astype(np.int32).tobytes()); flab.write(lab.tobytes()); ffe.write(fe.tobytes()); eoff.append(eoff[-1] + len(E))
        if i % 100 == 0: log(f"piece {i}/{P}  edges so far {eoff[-1]}  {time.time() - t0:.0f} s")
    fab.close(); flab.close(); ffe.close(); np.save(B + "eoff.npy", np.array(eoff, np.int64))
    lab = np.fromfile(B + "e_lab.bin", np.int8); log(f"edges {len(lab)}; labelled {int((lab >= 0).sum())}; bridge {int((lab == 1).sum())} (rate {(lab == 1).sum() / max((lab >= 0).sum(), 1) * 100:.4f}% of labelled)")


SIG_AREA = 0.002                                          # cm2: sub-pieces smaller than this are label noise (single-vertex flips of the truth turn)


def stage_siglabel():
    """significant bridge edges: bridge edges whose two bridge-free sub-pieces both have area >= SIG_AREA (a seam between two real sheet parts; isolated vertex flips do not count)"""
    S = Store(); out = open(B + "e_sig.bin", "wb"); tot = 0
    for i in range(S.P):
        V, F, E, fe, lab, valid = S.piece(i); nv = len(V); fa = face_area_cm2(V, F); keep = lab != 1
        nc, sub = cc(nv, E[keep, 0], E[keep, 1]); sa = np.bincount(sub[F[:, 0]], fa, minlength=nc)
        sig = ((lab == 1) & (sub[E[:, 0]] != sub[E[:, 1]]) & (sa[sub[E[:, 0]]] >= SIG_AREA) & (sa[sub[E[:, 1]]] >= SIG_AREA)).astype(np.int8); sig[lab < 0] = -1; out.write(sig.tobytes()); tot += int((sig == 1).sum())
    out.close(); log("significant bridge edges", tot)


def stage_minlabel():
    """region label: 1 when an end of the edge lies outside the largest bridge-free sub-piece of its piece, 0 when both ends are inside it; -1 for unlabelled edges and
    for pieces whose largest sub-piece holds less than half of the area (the choice of the 'main' part is arbitrary there)"""
    S = Store(); out = open(B + "e_min.bin", "wb"); tot = 0; amb = 0
    for i in range(S.P):
        V, F, E, fe, lab, valid = S.piece(i); nv = len(V); fa = face_area_cm2(V, F); keep = lab != 1
        nc, sub = cc(nv, E[keep, 0], E[keep, 1]); sa = np.bincount(sub[F[:, 0]], fa, minlength=nc); big = int(np.argmax(sa)); mino = sub != big
        y = (mino[E[:, 0]] | mino[E[:, 1]]).astype(np.int8); y[lab < 0] = -1
        if sa[big] < 0.5 * sa.sum(): y[:] = -1; amb += 1
        out.write(y.tobytes()); tot += int((y == 1).sum())
    out.close(); log("region-label edges", tot, "ambiguous pieces", amb)


class Store:
    """memory-mapped edge store"""
    def __init__(self):
        self.m = load_meta(); self.eoff = np.load(B + "eoff.npy"); ne = int(self.eoff[-1]); nf = int(self.m["foff"][-1])
        self.ab = np.memmap(B + "e_ab.bin", np.int32, "r", shape=(ne, 2)); self.lab = np.memmap(B + "e_lab.bin", np.int8, "r", shape=(ne,))
        self.fe = np.memmap(B + "f_fe.bin", np.int32, "r", shape=(nf, 3)); self.P = len(self.m["area"]); self.ne = ne
        self.sig = np.memmap(B + "e_sig.bin", np.int8, "r", shape=(ne,)) if os.path.exists(B + "e_sig.bin") else None
        self.minor = np.memmap(B + "e_min.bin", np.int8, "r", shape=(ne,)) if os.path.exists(B + "e_min.bin") else None
        self.n_all = np.load(B + "n.npy", mmap_mode="r")

    def piece(self, i, geometry=True):
        m = self.m; V, F = piece_VF(m, i) if geometry else (None, np.asarray(m["F"][m["foff"][i]:m["foff"][i + 1]]).astype(np.int64))
        E = np.asarray(self.ab[self.eoff[i]:self.eoff[i + 1]]).astype(np.int64); lab = np.asarray(self.lab[self.eoff[i]:self.eoff[i + 1]])
        fe = np.asarray(self.fe[m["foff"][i]:m["foff"][i + 1]]).astype(np.int64); n = np.asarray(self.n_all[m["voff"][i]:m["voff"][i + 1]])
        return V, F, E, fe, lab, np.isfinite(n)


# ---------------------------------------------------------------- piece evaluation (bigeval rule, extended to cut meshes)
def cc(nv, ea, eb):
    return connected_components(coo_matrix((np.ones(len(ea), np.int8), (ea, eb)), shape=(nv, nv)), directed=False)


def eval_piece(nv, F, fe, E, lab, valid, farea, ret, want_pure=True):
    """Pieces after a cut: a face is retained when `ret` is True; vertices are connected through the edges of retained faces.
    Returns an array [n_comp, 4] = (area cm2, largest bridge-free sub-piece area, truth coverage of the vertices, n vertices) for the components with area > 0."""
    if not ret.any(): return np.zeros((0, 4))
    ne = len(E); active = np.zeros(ne, bool); active[fe[ret].ravel()] = True
    nc, comp = cc(nv, E[active, 0], E[active, 1]); Fr = F[ret]; ar = farea[ret]; c0 = comp[Fr[:, 0]]
    A = np.bincount(c0, ar, minlength=nc); ids = np.nonzero(A > 0)[0]
    used = np.zeros(nv, bool); used[Fr.ravel()] = True
    cnt = np.bincount(comp[used], minlength=nc).astype(float); vcnt = np.bincount(comp[used], weights=valid[used].astype(float), minlength=nc)
    if want_pure:
        keep = active & (lab != 1); nc2, sub = cc(nv, E[keep, 0], E[keep, 1]); sa = np.bincount(sub[Fr[:, 0]], ar, minlength=nc2)
        sid = np.nonzero(sa > 0)[0]; sub_comp = np.zeros(nc2, np.int64); sub_comp[sub] = comp
        mx = np.zeros(nc); np.maximum.at(mx, sub_comp[sid], sa[sid])
    else:
        mx = A
    return np.stack([A[ids], mx[ids], vcnt[ids] / np.maximum(cnt[ids], 1), cnt[ids]], 1)


def summarize(rows):
    """rows: concatenated component rows. Retained = components >= AMIN_KEEP; scored = retained, >= AMIN_SCORE and coverage >= COV_MIN (bigeval)"""
    A, mx, cov = rows[:, 0], rows[:, 1], rows[:, 2]; kept = A >= AMIN_KEEP; sc = (A >= AMIN_SCORE) & (cov >= COV_MIN)
    out = dict(n_pieces=int(kept.sum()), area_kept=float(A[kept].sum()), n_scored=int(sc.sum()), area_scored=float(A[sc].sum()),
               purity=float(mx[sc].sum() / max(A[sc].sum(), 1e-12)), pure_area=float(mx[sc].sum()), pure_area_kept=float(mx[kept].sum()),
               purity_kept=float(mx[kept & (cov >= COV_MIN)].sum() / max(A[kept & (cov >= COV_MIN)].sum(), 1e-12)),
               max_pure=float(mx.max()) if len(mx) else 0.0, max_piece=float(A.max()) if len(A) else 0.0)
    return out


def stage_baseline():
    """the uncut pieces through eval_piece: must give the bigeval numbers (89.6% area-weighted purity, 60.9 cm2)"""
    S = Store(); rows = []; t0 = time.time()
    for i in range(S.P):
        V, F, E, fe, lab, valid = S.piece(i); nv = len(V); fa = face_area_cm2(V, F)
        r = eval_piece(nv, F, fe, E, lab, valid, fa, np.ones(len(F), bool)); rows.append(r)
        if i % 200 == 0: log(i, f"{time.time() - t0:.0f} s")
    R = np.concatenate(rows); s = summarize(R); log(json.dumps(s))
    lab = np.asarray(S.lab); log(f"edges {S.ne}, labelled {(lab >= 0).sum()}, bridge {(lab == 1).sum()} rate {(lab == 1).sum() / (lab >= 0).sum() * 100:.4f}%")


# ---------------------------------------------------------------- stage: fields (derived D=2 volumes, box A grid (256,1536,1536))
def wrap(a):
    return a - np.float32(2 * np.pi) * np.rint(a / np.float32(2 * np.pi))


def plane_slices(shape, a0, a1, i, j):
    return tuple(slice(i if ax == a0 else (j if ax == a1 else 0), (shape[ax] - 1 + i) if ax == a0 else ((shape[ax] - 1 + j) if ax == a1 else shape[ax])) for ax in range(3))


def vortex_mask(psi):
    """grid points that are corners of a unit square whose wrapped phase differences do not sum to zero (three coordinate planes), as in carve.py"""
    vor = np.zeros(psi.shape, bool)
    for (a0, a1) in ((0, 1), (0, 2), (1, 2)):
        sl = lambda i, j: plane_slices(psi.shape, a0, a1, i, j)
        bad = np.rint((wrap(psi[sl(1, 0)] - psi[sl(0, 0)]) + wrap(psi[sl(1, 1)] - psi[sl(1, 0)]) + wrap(psi[sl(0, 1)] - psi[sl(1, 1)]) + wrap(psi[sl(0, 0)] - psi[sl(0, 1)])) / np.float32(2 * np.pi)) != 0
        for da in (0, 1):
            for db in (0, 1): vor[plane_slices(psi.shape, a0, a1, da, db)] |= bad
    return vor


def field_gpsi():
    from numpy.lib.format import open_memmap
    P = np.load(PSI_F, mmap_mode="r"); Z, Y, X = P.shape; out = open_memmap(FD + "gpsi.npy", mode="w+", dtype=np.float16, shape=P.shape); SL = 8; t0 = time.time()
    for a in range(0, Z, SL):
        b = min(a + SL, Z); lo = max(a - 1, 0); hi = min(b + 1, Z); psi = np.asarray(P[lo:hi]).astype(np.float32); g2 = np.zeros(psi.shape, np.float32)
        for ax in range(3):
            d = wrap(np.diff(psi, axis=ax)); pad = [(0, 0)] * 3; pad[ax] = (1, 1); d = np.pad(d, pad, mode="edge")
            sl_a = [slice(None)] * 3; sl_a[ax] = slice(0, -1); sl_b = [slice(None)] * 3; sl_b[ax] = slice(1, None)
            c = 0.5 * (d[tuple(sl_a)] + d[tuple(sl_b)]); g2 += c * c
        out[a:b] = np.sqrt(g2[a - lo:b - lo]).astype(np.float16)
        if (a // SL) % 8 == 0: log(f"gpsi slab {a}-{b}  {time.time() - t0:.0f} s")
    out.flush()


def field_vort(R=24.0, TZ=64, TY=384, TX=384):
    from numpy.lib.format import open_memmap
    from scipy import ndimage as ndi
    P = np.load(PSI_F, mmap_mode="r"); Z, Y, X = P.shape; M = int(R) + 1
    dist = open_memmap(FD + "vdist.npy", mode="w+", dtype=np.uint8, shape=P.shape); den = open_memmap(FD + "vden.npy", mode="w+", dtype=np.uint8, shape=P.shape); t0 = time.time(); nt = 0; nvor = 0; ncell = 0
    for z0 in range(0, Z, TZ):
        for y0 in range(0, Y, TY):
            for x0 in range(0, X, TX):
                z1, y1, x1 = min(z0 + TZ, Z), min(y0 + TY, Y), min(x0 + TX, X); lo = np.maximum([z0 - M, y0 - M, x0 - M], 0); hi = np.minimum([z1 + M, y1 + M, x1 + M], [Z, Y, X])
                psi = np.asarray(P[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]).astype(np.float32); vor = vortex_mask(psi)
                c = (slice(z0 - lo[0], z1 - lo[0]), slice(y0 - lo[1], y1 - lo[1]), slice(x0 - lo[2], x1 - lo[2]))
                d = ndi.distance_transform_edt(~vor) if vor.any() else np.full(vor.shape, 1e3)
                dist[z0:z1, y0:y1, x0:x1] = np.rint(np.minimum(d[c], R) * 10).astype(np.uint8)
                dn = ndi.uniform_filter(vor.astype(np.float32), 7) * 343.0; den[z0:z1, y0:y1, x0:x1] = np.minimum(np.rint(dn[c]), 255).astype(np.uint8)
                nvor += int(vor[c].sum()); ncell += vor[c].size; nt += 1
                if nt % 8 == 0: log(f"vortex tile {nt}  {time.time() - t0:.0f} s")
    dist.flush(); den.flush(); log(f"vortex-marked grid points {nvor / ncell * 100:.3f}%")


def block_stats(vol, z0, z1, y0, y1, x0, x1, sq):
    """mean (and variance) of the L2 volume over 2x2x2 blocks, smoothed by a 3x3x3 box on the D=2 cells, for D=2 cells [z0,z1)x[y0,y1)x[x0,x1)"""
    from scipy import ndimage as ndi
    S = np.array(vol.shape); lo = np.array([z0 - 1, y0 - 1, x0 - 1]) * 2; hi = np.array([z1 + 1, y1 + 1, x1 + 1]) * 2
    lo_c = np.maximum(lo, 0); hi_c = np.minimum(hi, S); a = np.asarray(vol[lo_c[0]:hi_c[0], lo_c[1]:hi_c[1], lo_c[2]:hi_c[2]])
    if np.any(lo_c != lo) or np.any(hi_c != hi): a = np.pad(a, [(int(lo_c[k] - lo[k]), int(hi[k] - hi_c[k])) for k in range(3)], mode="edge")
    a = a.astype(np.float32); s = a.shape; a = a.reshape(s[0] // 2, 2, s[1] // 2, 2, s[2] // 2, 2)
    m1 = ndi.uniform_filter(a.mean((1, 3, 5)), 3, mode="nearest")[1:-1, 1:-1, 1:-1]
    if not sq: return m1
    m2 = ndi.uniform_filter((a * a).mean((1, 3, 5)), 3, mode="nearest")[1:-1, 1:-1, 1:-1]
    return m1, np.sqrt(np.maximum(m2 - m1 * m1, 0))


def field_ctsp(TZ=8, TY=384, TX=384):
    from numpy.lib.format import open_memmap
    ct = np.load(CT_F, mmap_mode="r"); sp = np.load(SP_F, mmap_mode="r"); Z, Y, X = ct.shape[0] // 2, ct.shape[1] // 2, ct.shape[2] // 2
    ctm = open_memmap(FD + "ctm.npy", mode="w+", dtype=np.uint8, shape=(Z, Y, X)); cts = open_memmap(FD + "cts.npy", mode="w+", dtype=np.uint8, shape=(Z, Y, X)); spm = open_memmap(FD + "spm.npy", mode="w+", dtype=np.uint8, shape=(Z, Y, X)); t0 = time.time()
    for z0 in range(0, Z, TZ):
        for y0 in range(0, Y, TY):
            for x0 in range(0, X, TX):
                z1, y1, x1 = min(z0 + TZ, Z), min(y0 + TY, Y), min(x0 + TX, X); sl = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
                m, s = block_stats(ct, z0, z1, y0, y1, x0, x1, True); ctm[sl] = np.clip(np.rint(m), 0, 255).astype(np.uint8); cts[sl] = np.clip(np.rint(s * 3), 0, 255).astype(np.uint8)
                spm[sl] = np.clip(np.rint(block_stats(sp, z0, z1, y0, y1, x0, x1, False)), 0, 255).astype(np.uint8)
        if (z0 // TZ) % 4 == 0: log(f"ct/sp z {z0}-{z1}  {time.time() - t0:.0f} s")
    ctm.flush(); cts.flush(); spm.flush()


def stage_fields():
    os.makedirs(FD, exist_ok=True); which = sys.argv[2] if len(sys.argv) > 2 else "all"
    if which in ("all", "gpsi"): field_gpsi()
    if which in ("all", "vort"): field_vort()
    if which in ("all", "ctsp"): field_ctsp()
    log("fields done", which)


# ---------------------------------------------------------------- stage: feats (edge feature matrix)
FEATS = ["conf_mid", "conf_min", "conf_max", "gpsi_mid", "gpsi_max", "gcell_mean", "galign_min", "galign_mean", "vdist_mid", "vdist_min", "vden_mid",
         "cos_mid", "gm_mid", "ctm_mid", "cts_mid", "spm_mid", "sp0_mean", "spw_mean", "spw_max", "spgap_min", "spgap_mean",
         "elen", "dihedral", "ncos_rad_mean", "ndiff", "e_z", "e_t", "e_r", "bdist_mean", "bdist_min", "log_area", "pconf",
         "conf_r3", "gpsi_r3", "spw_r3", "spgap_r3", "vdist_r3", "galign_r3"]
NF = len(FEATS)
SPS = 16                                                   # sp profile half length along the normal (L2 voxels)


def trilin_setup(g, shape):
    """corner flat indices [8,N] and weights [8,N] for trilinear sampling at index-space points g [N,3]; corner c = dz*4 + dy*2 + dx"""
    sh = np.array(shape, float); g = np.minimum(np.maximum(g, 0.0), sh - 1.001); i0 = np.floor(g).astype(np.int64); f = (g - i0).astype(np.float32)
    base = (i0[:, 0] * shape[1] + i0[:, 1]) * shape[2] + i0[:, 2]; idx = np.empty((8, len(g)), np.int64); w = np.empty((8, len(g)), np.float32)
    for c in range(8):
        dz, dy, dx = (c >> 2) & 1, (c >> 1) & 1, c & 1
        idx[c] = base + dz * shape[1] * shape[2] + dy * shape[2] + dx
        w[c] = (f[:, 0] if dz else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dx else 1 - f[:, 2])
    return idx, w


def gather(flat, idx, w):
    return (flat[idx].astype(np.float32) * w).sum(0)


def vertex_normals(V, F):
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]); N = np.zeros_like(V)
    for k in range(3):
        for c in range(3): N[:, c] += np.bincount(F[:, k], fn[:, c], minlength=len(V))
    return N / (np.linalg.norm(N, axis=1, keepdims=True) + 1e-12), fn


def sp_profile_feats(spflat, shape, V, N, chunk=40000):
    """along the vertex normal: sp at the vertex, width of the sp > 127 run, gaps to the next sp > 127 run on either side (L2 voxels, capped at SPS)"""
    nv = len(V); pos = V - ORG; s = np.arange(-SPS, SPS + 1); out = np.zeros((nv, 4), np.float32); ar = np.arange(SPS)
    for c0 in range(0, nv, chunk):
        c1 = min(c0 + chunk, nv); P = pos[c0:c1, None, :] + s[None, :, None] * N[c0:c1, None, :]; ix = np.rint(P).astype(np.int64)
        ix = np.minimum(np.maximum(ix, 0), np.array(shape) - 1); prof = spflat[(ix[..., 0] * shape[1] + ix[..., 1]) * shape[2] + ix[..., 2]]; ab = prof > 127; cen = ab[:, SPS]
        res = []
        for side in (ab[:, SPS + 1:], ab[:, :SPS][:, ::-1]):
            nf_ = ~side; r = np.where(nf_.any(1), nf_.argmax(1), SPS)                       # run length from s = +-1
            t = side & (ar[None, :] > r[:, None]); gap = np.where(t.any(1), t.argmax(1) - r, SPS); gap = np.where(r >= SPS, SPS, gap); res.append((r, gap))
        out[c0:c1, 0] = prof[:, SPS]; out[c0:c1, 1] = cen + res[0][0] + res[1][0]; out[c0:c1, 2] = np.minimum(res[0][1], res[1][1]); out[c0:c1, 3] = np.maximum(res[0][1], res[1][1])
    return out


class Samplers:
    def __init__(self):
        self.vols = {}
        for k, f in (("conf", CONF_F), ("psi", PSI_F), ("cos", COS_F), ("gpsi", FD + "gpsi.npy"), ("vdist", FD + "vdist.npy"), ("vden", FD + "vden.npy"),
                     ("ctm", FD + "ctm.npy"), ("cts", FD + "cts.npy"), ("spm", FD + "spm.npy")):
            a = np.load(f, mmap_mode="r"); self.vols[k] = (a.reshape(-1), a.shape)
        gm = np.load(GM_F, mmap_mode="r"); self.gm = (gm.reshape(-1), gm.shape); sp = np.load(SP_F, mmap_mode="r"); self.sp = (sp.reshape(-1), sp.shape)
        self.uz, self.uy, self.ux = umbilicus()

    def vertex_feats(self, V, N):
        """per-vertex: conf, gpsi, vdist, gcell, galign, sp profile (4 columns)"""
        g = (V - ORG - 0.5) / 2.0; shp = self.vols["conf"][1]; idx, w = trilin_setup(g, shp); o = {}
        for k in ("conf", "gpsi", "vdist"): o[k] = gather(self.vols[k][0], idx, w) / (10.0 if k == "vdist" else 1.0)
        pc = self.vols["psi"][0][idx].astype(np.float32); c = lambda dz, dy, dx: pc[dz * 4 + dy * 2 + dx]
        gz = sum(wrap(c(1, dy, dx) - c(0, dy, dx)) for dy in (0, 1) for dx in (0, 1)) / 4; gy = sum(wrap(c(dz, 1, dx) - c(dz, 0, dx)) for dz in (0, 1) for dx in (0, 1)) / 4
        gx = sum(wrap(c(dz, dy, 1) - c(dz, dy, 0)) for dz in (0, 1) for dy in (0, 1)) / 4; gn = np.sqrt(gz ** 2 + gy ** 2 + gx ** 2)
        o["gcell"] = gn; o["galign"] = np.abs(gz * N[:, 0] + gy * N[:, 1] + gx * N[:, 2]) / (gn + 1e-6)
        o["sp"] = sp_profile_feats(self.sp[0], self.sp[1], V, N); return o

    def mid_feats(self, Vm, chunk=250000):
        out = np.zeros((len(Vm), 9), np.float32); shp = self.vols["conf"][1]
        for c0 in range(0, len(Vm), chunk):
            c1 = min(c0 + chunk, len(Vm)); g = (Vm[c0:c1] - ORG - 0.5) / 2.0; idx, w = trilin_setup(g, shp)
            for j, k in enumerate(("conf", "gpsi", "vdist", "vden", "cos", "ctm", "cts", "spm")): out[c0:c1, j] = gather(self.vols[k][0], idx, w)
            idx4, w4 = trilin_setup((Vm[c0:c1] - ORG - 1.5) / 4.0, self.gm[1]); out[c0:c1, 8] = gather(self.gm[0], idx4, w4)
        out[:, 2] /= 10.0; return out


def ring_mean(E, nv, x, it=3):
    from scipy.sparse import csr_matrix
    A = csr_matrix((np.ones(2 * len(E), np.float32), (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(nv, nv)); deg = np.maximum(np.asarray(A.sum(1)).ravel(), 1)
    for _ in range(it): x = (A @ x) / deg
    return x


def piece_features(S, Sm, i):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
    V, F, E, fe, lab, valid = S.piece(i); nv, nf, ne = len(V), len(F), len(E); a, b = E[:, 0], E[:, 1]; N, fn = vertex_normals(V, F)
    vf = Sm.vertex_feats(V, N); Vm = 0.5 * (V[a] + V[b]); mf = Sm.mid_feats(Vm)
    ev = V[b] - V[a]; elen = np.linalg.norm(ev, axis=1); cy = np.interp(V[:, 0], Sm.uz, Sm.uy); cx = np.interp(V[:, 0], Sm.uz, Sm.ux)
    ry, rx = V[:, 1] - cy, V[:, 2] - cx; rn = np.hypot(ry, rx) + 1e-9; ry, rx = ry / rn, rx / rn                        # radial unit vector (y, x); tangential = (-rx, ry)
    ncos = np.abs(N[:, 1] * ry + N[:, 2] * rx); ndiff = np.arccos(np.clip((N[a] * N[b]).sum(1), -1, 1))
    evn = ev / (elen[:, None] + 1e-9); e_t = np.abs(-evn[:, 1] * rx[a] + evn[:, 2] * ry[a]); e_r = np.abs(evn[:, 1] * ry[a] + evn[:, 2] * rx[a])
    # dihedral angle of the two faces sharing the edge (0 on the piece boundary)
    fnu = fn / (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12); ei = fe.ravel(); fid = np.repeat(np.arange(nf), 3); order = np.argsort(ei, kind="stable"); cnt = np.bincount(ei, minlength=ne)
    start = np.cumsum(cnt) - cnt; two = np.nonzero(cnt == 2)[0]; f1 = fid[order[start[two]]]; f2 = fid[order[start[two] + 1]]
    dih = np.zeros(ne, np.float32); dih[two] = np.arccos(np.clip((fnu[f1] * fnu[f2]).sum(1), -1, 1))
    bv = np.zeros(nv, bool); be = np.nonzero(cnt == 1)[0]; bv[a[be]] = True; bv[b[be]] = True
    if bv.any(): G = csr_matrix((elen.astype(np.float64), (a, b)), shape=(nv, nv)); bd = dijkstra(G, directed=False, indices=np.nonzero(bv)[0], min_only=True); bd = np.minimum(bd, 1e4) / 2.0
    else: bd = np.full(nv, 1e4)
    X = np.zeros((ne, NF), np.float32); put = lambda name, v: X.__setitem__((slice(None), FEATS.index(name)), v)
    c = vf["conf"]; put("conf_mid", mf[:, 0]); put("conf_min", np.minimum(c[a], c[b])); put("conf_max", np.maximum(c[a], c[b]))
    put("gpsi_mid", mf[:, 1]); put("gpsi_max", np.maximum(vf["gpsi"][a], vf["gpsi"][b])); put("gcell_mean", 0.5 * (vf["gcell"][a] + vf["gcell"][b]))
    put("galign_min", np.minimum(vf["galign"][a], vf["galign"][b])); put("galign_mean", 0.5 * (vf["galign"][a] + vf["galign"][b]))
    put("vdist_mid", mf[:, 2]); put("vdist_min", np.minimum(vf["vdist"][a], vf["vdist"][b])); put("vden_mid", mf[:, 3]); put("cos_mid", mf[:, 4]); put("gm_mid", mf[:, 8])
    put("ctm_mid", mf[:, 5]); put("cts_mid", mf[:, 6]); put("spm_mid", mf[:, 7]); sp = vf["sp"]
    put("sp0_mean", 0.5 * (sp[a, 0] + sp[b, 0])); put("spw_mean", 0.5 * (sp[a, 1] + sp[b, 1])); put("spw_max", np.maximum(sp[a, 1], sp[b, 1]))
    put("spgap_min", np.minimum(sp[a, 2], sp[b, 2])); put("spgap_mean", 0.5 * (sp[a, 2] + sp[b, 2]))
    put("elen", elen); put("dihedral", dih); put("ncos_rad_mean", 0.5 * (ncos[a] + ncos[b])); put("ndiff", ndiff); put("e_z", np.abs(evn[:, 0])); put("e_t", e_t); put("e_r", e_r)
    put("bdist_mean", 0.5 * (bd[a] + bd[b])); put("bdist_min", np.minimum(bd[a], bd[b]))
    ar = face_area_cm2(V, F).sum(); put("log_area", np.log10(ar)); put("pconf", float(c.mean()))
    for nm, v in (("conf_r3", c), ("gpsi_r3", vf["gpsi"]), ("spw_r3", sp[:, 1]), ("spgap_r3", sp[:, 2]), ("vdist_r3", vf["vdist"]), ("galign_r3", vf["galign"])):
        r = ring_mean(E, nv, v.astype(np.float32)); put(nm, 0.5 * (r[a] + r[b]))
    return X


def stage_feats():
    from numpy.lib.format import open_memmap
    S = Store(); Sm = Samplers(); first = int(sys.argv[2]) if len(sys.argv) > 2 else 0; last = int(sys.argv[3]) if len(sys.argv) > 3 else S.P
    path = B + "feats.npy"
    if first == 0 and not os.path.exists(path): out = open_memmap(path, mode="w+", dtype=np.float16, shape=(S.ne, NF))
    else: out = np.load(path, mmap_mode="r+")
    json.dump(FEATS, open(B + "feats.json", "w")); t0 = time.time()
    for i in range(first, last):
        X = piece_features(S, Sm, i); out[S.eoff[i]:S.eoff[i + 1]] = X.astype(np.float16)
        if i % 20 == 0 or i == last - 1: log(f"piece {i}/{S.P} edges {S.eoff[i + 1]}  {time.time() - t0:.0f} s"); out.flush()
    out.flush(); log("feats done", first, last)


# ---------------------------------------------------------------- stage: train (grouped CV)
GROUPS = {"conf": ["conf_mid", "conf_min", "conf_max", "pconf", "conf_r3"],
          "psi": ["gpsi_mid", "gpsi_max", "gcell_mean", "galign_min", "galign_mean", "vdist_mid", "vdist_min", "vden_mid", "gpsi_r3", "vdist_r3", "galign_r3"],
          "img": ["cos_mid", "gm_mid", "ctm_mid", "cts_mid", "spm_mid", "sp0_mean", "spw_mean", "spw_max", "spgap_min", "spgap_mean", "spw_r3", "spgap_r3"],
          "geom": ["elen", "dihedral", "ncos_rad_mean", "ndiff", "e_z", "e_t", "e_r", "bdist_mean", "bdist_min", "log_area"]}
SETS = {"all": sum(GROUPS.values(), []), "conf": GROUPS["conf"], "conf+psi": GROUPS["conf"] + GROUPS["psi"], "conf+psi+img": GROUPS["conf"] + GROUPS["psi"] + GROUPS["img"],
        "no_geom": GROUPS["conf"] + GROUPS["psi"] + GROUPS["img"], "no_conf": GROUPS["psi"] + GROUPS["img"] + GROUPS["geom"], "no_psi": GROUPS["conf"] + GROUPS["img"] + GROUPS["geom"],
        "no_img": GROUPS["conf"] + GROUPS["psi"] + GROUPS["geom"], "geom": GROUPS["geom"], "psi": GROUPS["psi"], "img": GROUPS["img"],
        "piece": ["log_area", "pconf"], "nopiece": [c for c in sum(GROUPS.values(), []) if c not in ("log_area", "pconf")]}
K_FOLD = 5


def piece_az(S):
    path = B + "piece_az.npy"
    if os.path.exists(path): return np.load(path)
    uz, uy, ux = umbilicus(); az = np.zeros(S.P)
    for i in range(S.P):
        V = np.asarray(S.m["V"][S.m["voff"][i]:S.m["voff"][i + 1]:20]).astype(np.float64); ry = V[:, 1] - np.interp(V[:, 0], uz, uy); rx = V[:, 2] - np.interp(V[:, 0], uz, ux); r = np.hypot(ry, rx) + 1e-9
        az[i] = np.arctan2((ry / r).mean(), (rx / r).mean())
    np.save(path, az); return az


def make_folds(S, mode, K=K_FOLD, seed=0):
    area = S.m["area"]; fold = np.zeros(S.P, int)
    if mode == "piece":                                                                    # balanced by area, random tie order
        rng = np.random.default_rng(seed); order = rng.permutation(S.P); order = order[np.argsort(-area[order], kind="stable")]; load = np.zeros(K)
        for p in order: k = int(np.argmin(load)); fold[p] = k; load[k] += area[p]
    else:                                                                                   # azimuth sectors of equal area: whole angular blocks are held out
        az = piece_az(S); o = np.argsort(az); cs = np.cumsum(area[o]) / area.sum(); fold[o] = np.minimum((cs * K).astype(int), K - 1)
    return fold


def edge_piece_id(S):
    return np.repeat(np.arange(S.P, dtype=np.int16), np.diff(S.eoff))


def eval_sample(S, rate=0.1, seed=7, labarr=None):
    """all positive edges + a fixed `rate` sample of the negative edges (weight 1/rate) among the labelled ones; indices sorted"""
    rng = np.random.default_rng(seed); idx = []; labarr = S.lab if labarr is None else labarr
    for i in range(S.P):
        lab = np.asarray(labarr[S.eoff[i]:S.eoff[i + 1]]); pos = np.nonzero(lab == 1)[0]; neg = np.nonzero(lab == 0)[0]; neg = neg[rng.random(len(neg)) < rate]
        idx.append(S.eoff[i] + np.sort(np.r_[pos, neg]))
    return np.concatenate(idx)


def wmetrics(y, p, rate):
    from sklearn.metrics import roc_auc_score, average_precision_score
    w = np.where(y == 1, 1.0, 1.0 / rate); return float(roc_auc_score(y, p, sample_weight=w)), float(average_precision_score(y, p, sample_weight=w))


def load_cols(Xmm, idx, cols):
    """rows idx (sorted) and columns cols of the float16 memmap as float32"""
    out = np.empty((len(idx), len(cols)), np.float32); step = 1 << 20
    for s in range(0, len(idx), step):
        blk = np.asarray(Xmm[idx[s:s + step]]); out[s:s + step] = blk[:, cols].astype(np.float32)
    return out


def truncated(clf, n):
    """shallow copy of a fitted HistGradientBoostingClassifier that uses only its first n boosting iterations"""
    import copy
    c = copy.copy(clf); c._predictors = clf._predictors[:n]; return c


def stage_train():
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score
    mode = sys.argv[2] if len(sys.argv) > 2 else "piece"; fset = sys.argv[3] if len(sys.argv) > 3 else "all"; tag = sys.argv[4] if len(sys.argv) > 4 else f"{mode}_{fset}"
    max_iter = int(os.environ.get("MAX_ITER", "250")); neg_rate = float(os.environ.get("NEG_RATE", "0.04")); lr = float(os.environ.get("LR", "0.1")); leaves = int(os.environ.get("LEAVES", "63"))
    S = Store(); Xmm = np.load(B + "feats.npy", mmap_mode="r"); names = json.load(open(B + "feats.json")); cols = [names.index(c) for c in SETS[fset]]
    LABEL = os.environ.get("LABEL", "bridge"); QUICK = bool(int(os.environ.get("QUICK", "0"))); NFOLD = int(os.environ.get("NFOLD", str(K_FOLD)))
    if LABEL == "sig":                                                               # significant seams: 1 = sig, 0 = any other labelled edge (incl. isolated vertex flips)
        lab_all = np.where(np.asarray(S.lab) >= 0, np.asarray(S.sig) == 1, -1).astype(np.int8)
    elif LABEL == "minor": lab_all = S.minor
    else: lab_all = S.lab
    fold = make_folds(S, mode); area = S.m["area"]; pid = edge_piece_id(S); log(f"mode {mode} features {fset} ({len(cols)}), fold areas {[round(float(area[fold == k].sum()), 2) for k in range(K_FOLD)]}")
    ES = eval_sample(S, 0.1, 7, lab_all); yES = np.asarray(lab_all[ES]); pES = np.full(len(ES), np.nan, np.float32); log(f"eval sample {len(ES)} edges ({int((yES == 1).sum())} bridge)")
    p_oof = np.zeros(S.ne, np.float32); res = {"mode": mode, "set": fset, "label": LABEL, "quick": QUICK, "folds": []}
    for k in range(NFOLD):
        rng = np.random.default_rng(100 + k); rest = np.nonzero(fold != k)[0]; rest = rng.permutation(rest); ca = np.cumsum(area[rest]); isval = np.zeros(S.P, bool); isval[rest[ca > 0.75 * ca[-1]]] = True
        istr = (fold != k) & ~isval; log(f"fold {k}: train {int(istr.sum())} pieces ({area[istr].sum():.1f} cm2), val {int(isval.sum())} ({area[isval].sum():.1f}), test {int((fold == k).sum())} ({area[fold == k].sum():.1f})")
        def sample(mask, rate, seed):
            r = np.random.default_rng(seed); idx = []
            for i in np.nonzero(mask)[0]:
                lab = np.asarray(lab_all[S.eoff[i]:S.eoff[i + 1]]); pos = np.nonzero(lab == 1)[0]; neg = np.nonzero(lab == 0)[0]; neg = neg[r.random(len(neg)) < rate]; idx.append(S.eoff[i] + np.sort(np.r_[pos, neg]))
            return np.concatenate(idx)
        itr = sample(istr, neg_rate, 200 + k); iva = sample(isval, 0.05, 300 + k); ytr = np.asarray(lab_all[itr]) == 1; yva = np.asarray(lab_all[iva]) == 1
        Xtr = load_cols(Xmm, itr, cols); Xva = load_cols(Xmm, iva, cols); wva = np.where(yva, 1.0, 1 / 0.05); log(f"  train rows {len(itr)} (pos {int(ytr.sum())}), val rows {len(iva)} (pos {int(yva.sum())})")
        t0 = time.time(); clf = HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=lr, max_leaf_nodes=leaves, min_samples_leaf=200, l2_regularization=1.0, early_stopping=False, random_state=k)
        clf.fit(Xtr, ytr); log(f"  fit {time.time() - t0:.0f} s")
        best, bap, curve = 0, -1, []
        for it, pv in enumerate(clf.staged_predict_proba(Xva), 1):
            if it % 10 == 0 or it == max_iter:
                ap = average_precision_score(yva, pv[:, 1], sample_weight=wva); curve.append((it, round(float(ap), 4)))
                if ap > bap: bap, best = ap, it
        log(f"  val AP curve {curve[::3]}; best iteration {best} (AP {bap:.4f})"); del Xtr
        joblib.dump((clf, best, cols), B + f"model_{tag}_k{k}.joblib")
        te = np.nonzero(fold == k)[0]; t1 = time.time(); clf_b = truncated(clf, best)
        def predict_pieces(pieces):
            """predict all edges of the given pieces in chunks, with the first `best` iterations"""
            out = {}; pend = []; n = 0
            def flush():
                if not pend: return
                rows = np.concatenate([np.arange(S.eoff[i], S.eoff[i + 1]) for i in pend]); X = load_cols(Xmm, rows, cols)
                out_p = clf_b.predict_proba(X)[:, 1].astype(np.float32); o = 0
                for i in pend: m_ = S.eoff[i + 1] - S.eoff[i]; out[i] = out_p[o:o + m_]; o += m_
                pend.clear()
            for i in pieces:
                pend.append(int(i)); n += S.eoff[i + 1] - S.eoff[i]
                if n >= 1500000: flush(); n = 0
            flush(); return out
        if QUICK:
            m = np.isin(pid[ES], te); rows_ = ES[m]; X = load_cols(Xmm, rows_, cols)
            pES[m] = clf_b.predict_proba(X)[:, 1]; yk = yES[m]; pk = pES[m]; auc, ap = wmetrics(yk, pk, 0.1)
        else:
            pr = predict_pieces(te)
            for i, v in pr.items(): p_oof[S.eoff[i]:S.eoff[i + 1]] = v
            pval = predict_pieces(np.nonzero(isval)[0]); pv_arr = np.full(S.ne, np.nan, np.float16)
            for i, v in pval.items(): pv_arr[S.eoff[i]:S.eoff[i + 1]] = v
            np.save(B + f"pval_{tag}_k{k}.npy", pv_arr); del pv_arr, pval
            m = np.isin(pid[ES], te); yk = yES[m]; pk = p_oof[ES[m]]; pES[m] = pk; auc, ap = wmetrics(yk, pk, 0.1)
        log(f"  fold {k}: test AUC {auc:.4f}  AP {ap:.4f}  (base rate {float((yk == 1).sum() / ((yk == 1).sum() + (yk == 0).sum() / 0.1)) * 100:.2f}%)  predict {time.time() - t1:.0f} s")
        res["folds"].append({"fold": k, "auc": auc, "ap": ap, "best_iter": best, "val_ap": float(bap)})
    done = np.isfinite(pES); auc, ap = wmetrics(yES[done], pES[done], 0.1); base = float((yES == 1).sum() / ((yES == 1).sum() + (yES == 0).sum() / 0.1))
    res.update(auc=auc, ap=ap, base_rate=base); log(f"POOLED out-of-fold: AUC {auc:.4f}  AP {ap:.4f}  base rate {base * 100:.3f}%")
    if not QUICK: np.save(B + f"p_oof_{tag}.npy", p_oof)
    json.dump(res, open(B + f"train_{tag}.json", "w"), indent=1)


def stage_univariate():
    S = Store(); Xmm = np.load(B + "feats.npy", mmap_mode="r"); names = json.load(open(B + "feats.json")); ES = eval_sample(S, 0.03, 11); y = np.asarray(S.lab[ES]); w = np.where(y == 1, 1.0, 1 / 0.03)
    from sklearn.metrics import roc_auc_score, average_precision_score
    X = load_cols(Xmm, ES, list(range(len(names)))); rows = []
    for j, nm in enumerate(names):
        a = roc_auc_score(y, X[:, j], sample_weight=w); rows.append((nm, a, np.median(X[y == 1, j]), np.median(X[y == 0, j])))
    for nm, a, mp, mn in sorted(rows, key=lambda r: -abs(r[1] - 0.5)): print(f"{nm:16s} AUC(bridge high) {a:.3f}   median bridge {mp:9.3f}  other {mn:9.3f}")


# ---------------------------------------------------------------- stage: cut (cut simulation)
P_GRID = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.22, 0.16, 0.11, 0.08, 0.055, 0.04, 0.028, 0.02]
Q_GRID = [0.002, 0.005, 0.01, 0.02, 0.035, 0.05, 0.075, 0.1, 0.15, 0.2]            # fraction of edges cut (heuristic baselines)


def edge_cut_mask(s, t, E, nv, dilate):
    cut = s > t
    if dilate and cut.any():
        vc = np.zeros(nv, bool); vc[E[cut, 0]] = True; vc[E[cut, 1]] = True
        cut = cut | (vc[E[:, 0]] & vc[E[:, 1]] if dilate == 1 else vc[E[:, 0]] | vc[E[:, 1]])
    return cut


def run_sweep(S, getscore, thrs, pieces, dilate=0, want_pure=True, tag=""):
    """for every threshold: component rows [n, 5] = (area, largest bridge-free sub-piece, coverage, n vertices, piece id) of all pieces in `pieces` after the cut"""
    rows = [[] for _ in thrs]; t0 = time.time()
    for c, i in enumerate(pieces):
        V, F, E, fe, lab, valid = S.piece(int(i)); nv = len(V); fa = face_area_cm2(V, F); s = getscore(int(i))
        for j, t in enumerate(thrs):
            cut = edge_cut_mask(s, t, E, nv, dilate); ret = ~cut[fe].any(1); r = eval_piece(nv, F, fe, E, lab, valid, fa, ret, want_pure)
            if len(r): rows[j].append(np.c_[r, np.full(len(r), i)])
        if c % 200 == 0: log(f"  {tag} piece {c}/{len(pieces)}  {time.time() - t0:.0f} s")
    return [np.concatenate(r) if r else np.zeros((0, 5)) for r in rows]


def score_getter(S, method, tag):
    """edge scores (higher = more likely to be cut) per piece"""
    if method == "model":
        p = np.load(B + f"p_oof_{tag}.npy", mmap_mode="r"); return lambda i: np.asarray(p[S.eoff[i]:S.eoff[i + 1]]).astype(np.float64)
    names = json.load(open(B + "feats.json")); Xmm = np.load(B + "feats.npy", mmap_mode="r"); col = {"conf": ("conf_min", -1), "bdist": ("bdist_min", -1), "vdist": ("vdist_min", -1), "gpsi": ("gpsi_max", 1)}[method]
    j = names.index(col[0]); return lambda i: col[1] * np.asarray(Xmm[S.eoff[i]:S.eoff[i + 1], j]).astype(np.float64)


def quantile_thresholds(S, getscore, qs):
    v = np.concatenate([getscore(i)[::40] for i in range(S.P)]); return [float(np.quantile(v, 1 - q)) for q in qs]


def stage_cut():
    method = sys.argv[2]; tag = sys.argv[3] if len(sys.argv) > 3 else "piece_all"; dilate = int(sys.argv[4]) if len(sys.argv) > 4 else 0; S = Store(); gs = score_getter(S, method, tag)
    EXPL = {"conf": [-g for g in (0.28, 0.30, 0.32, 0.35, 0.38, 0.41, 0.45, 0.5, 0.55, 0.6)],        # cut where the minimum vertex confidence of the edge is below g
            "bdist": [-r for r in (0.5, 1.5, 2.5, 4, 6, 9, 13, 18, 25, 35)],                            # cut within r grid cells (geodesic) of the piece boundary: extra erosion
            "vdist": [-r for r in (0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10)]}                                # cut within r grid cells of a phase vortex plaquette
    thrs = P_GRID if method == "model" else EXPL.get(method) or quantile_thresholds(S, gs, Q_GRID); thrs = [float(t) for t in thrs]
    mode = tag.split("_")[0]; fold = make_folds(S, mode); out = {"method": method, "tag": tag, "dilate": dilate, "thresholds": thrs, "points": []}
    # no cut reference + the sweep, all pieces
    rows = run_sweep(S, gs, [1e9] + thrs, np.arange(S.P), dilate, True, f"{method}{dilate}")
    for j, t in enumerate([None] + thrs):
        R = rows[j]; s = summarize(R); s["thr"] = t; s["per_fold"] = [summarize(R[fold[R[:, 4].astype(int)] == k]) for k in range(K_FOLD)]; out["points"].append(s)
        log(f"  thr {t}: kept {s['area_kept']:.2f} cm2 in {s['n_pieces']} pieces, scored {s['area_scored']:.2f}, purity {s['purity'] * 100:.2f}%, pure area (kept) {s['pure_area_kept']:.2f}, largest pure {s['max_pure']:.2f}")
    json.dump(out, open(B + f"cut_{method}_{tag}_d{dilate}.json", "w"), indent=1)


def stage_select():
    """honest operating points: per outer fold, the threshold is chosen on the validation pieces of that fold (its model never saw them) as the most aggressive one that keeps the target area fraction; applied to the held-out fold"""
    tag = sys.argv[2]; dilate = int(sys.argv[3]) if len(sys.argv) > 3 else 0; S = Store(); mode = tag.split("_")[0]; fold = make_folds(S, mode); area = S.m["area"]
    cutj = json.load(open(B + f"cut_model_{tag}_d{dilate}.json")); thrs = cutj["thresholds"]; targets = [0.97, 0.95, 0.925, 0.9, 0.875, 0.85, 0.8, 0.7]
    # retained area fraction on the validation pieces (area only, no purity)
    frac = np.zeros((K_FOLD, len(thrs)))
    for k in range(K_FOLD):
        pv = np.load(B + f"pval_{tag}_k{k}.npy", mmap_mode="r"); gs = lambda i: np.asarray(pv[S.eoff[i]:S.eoff[i + 1]]).astype(np.float64); val = [i for i in range(S.P) if np.isfinite(float(pv[S.eoff[i]]))]
        rows = run_sweep(S, gs, thrs, val, dilate, False, f"val{k}"); tot = area[val].sum()
        for j in range(len(thrs)): R = rows[j]; frac[k, j] = R[R[:, 0] >= AMIN_KEEP, 0].sum() / tot
        log(f"fold {k}: validation pieces {len(val)} ({tot:.1f} cm2); retained fraction by threshold {np.round(frac[k], 3).tolist()}")
    # the per-fold test rows at the chosen thresholds come from a second pass over the held-out pieces
    ops = []
    for f in targets:
        tk = []
        for k in range(K_FOLD):
            ok = [j for j in range(len(thrs)) if frac[k, j] >= f]; tk.append(min(ok, key=lambda j: thrs[j]) if ok else None)       # None = no cut
        ops.append((f, tk))
    thr_of = lambda j: thrs[j] if j is not None else 1e9
    p = np.load(B + f"p_oof_{tag}.npy", mmap_mode="r"); gs = lambda i: np.asarray(p[S.eoff[i]:S.eoff[i + 1]]).astype(np.float64)
    rows_f = []
    for k in range(K_FOLD):
        used = sorted({tk[k] for f, tk in ops}, key=lambda j: -1 if j is None else j); te = np.nonzero(fold == k)[0]; r = run_sweep(S, gs, [thr_of(j) for j in used], te, dilate, True, f"test{k}"); rows_f.append(dict(zip(used, r)))
    res = []
    for f, tk in ops:
        R = np.concatenate([rows_f[k][tk[k]] for k in range(K_FOLD)]); s = summarize(R); s["target_fraction"] = f; s["thresholds_by_fold"] = [thr_of(j) for j in tk]; s["per_fold"] = [summarize(rows_f[k][tk[k]]) for k in range(K_FOLD)]; res.append(s)
        log(f"  target {f:.3f} of the area: thresholds {s['thresholds_by_fold']}  ->  kept {s['area_kept']:.2f} cm2, scored {s['area_scored']:.2f}, purity {s['purity'] * 100:.2f}%, pure area (kept) {s['pure_area_kept']:.2f}")
    json.dump(res, open(B + f"select_{tag}_d{dilate}.json", "w"), indent=1)


def stage_export():
    """SURF npz (bigeval.py / piece2surf.py convention) of the pieces after a cut.
    python bridge_clf.py export METHOD TAG DILATE THRESHOLD OUT.npz [MIN_SURF_AREA]   (THRESHOLD 1e9 = no cut)"""
    import zipfile
    from numpy.lib import format as npf
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import piece2surf as p2s
    method, tag, dilate, thr, out = sys.argv[2], sys.argv[3], int(sys.argv[4]), float(sys.argv[5]), sys.argv[6]; amin = float(sys.argv[7]) if len(sys.argv) > 7 else AMIN_SCORE
    S = Store(); gs = score_getter(S, method, tag); names = []; zf = zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, allowZip64=True); kept_area = 0.0; t0 = time.time(); nsurf_area = 0.0
    for i in range(S.P):
        V, F, E, fe, lab, valid = S.piece(i); nv = len(V); fa = face_area_cm2(V, F); cut = edge_cut_mask(gs(i), thr, E, nv, dilate); ret = ~cut[fe].any(1)
        if not ret.any(): continue
        active = np.zeros(len(E), bool); active[fe[ret].ravel()] = True; nc, comp = cc(nv, E[active, 0], E[active, 1]); A = np.bincount(comp[F[ret, 0]], fa[ret], minlength=nc)
        kept_area += A[A >= AMIN_KEEP].sum()
        for c in np.nonzero(A >= amin)[0]:
            vi = np.nonzero(comp == c)[0]; mp = np.full(nv, -1, np.int64); mp[vi] = np.arange(len(vi)); ec = active & (comp[E[:, 0]] == c); El = np.sort(np.c_[mp[E[ec, 0]], mp[E[ec, 1]]], 1)
            g = p2s.piece_grid(V[vi], El)
            if g is None: continue
            with zf.open(f"xyz_{len(names)}.npy", "w", force_zip64=True) as f: npf.write_array(f, g, allow_pickle=False)
            names.append(f"b_{i}_c{c}"); nsurf_area += A[c]
        if i % 50 == 0: log(f"piece {i}/{S.P}  surfaces {len(names)}  {time.time() - t0:.0f} s")
    with zf.open("names.npy", "w") as f: npf.write_array(f, np.array(names), allow_pickle=False)
    zf.close(); log(f"saved {out}: {len(names)} surfaces >= {amin} cm2, area in them {nsurf_area:.2f} cm2, retained area (pieces >= {AMIN_KEEP}) {kept_area:.2f} cm2")


def stage_importance():
    """permutation importance (drop in weighted AUC) of single features and feature groups for the fold-k model on its held-out pieces"""
    import joblib
    tag = sys.argv[2] if len(sys.argv) > 2 else "piece_all"; k = int(sys.argv[3]) if len(sys.argv) > 3 else 0; S = Store(); Xmm = np.load(B + "feats.npy", mmap_mode="r"); names = json.load(open(B + "feats.json"))
    clf, best, cols = joblib.load(B + f"model_{tag}_k{k}.joblib"); clf = truncated(clf, best); fold = make_folds(S, tag.split("_")[0]); ES = eval_sample(S, 0.1, 7); pid = edge_piece_id(S)
    m = fold[pid[ES]] == k; rows = ES[m]; y = np.asarray(S.lab[rows]); sel = np.random.default_rng(0).random(len(rows)) < 0.35; rows, y = rows[sel], y[sel]
    X = load_cols(Xmm, rows, cols); w = np.where(y == 1, 1.0, 10.0); from sklearn.metrics import roc_auc_score
    base = roc_auc_score(y, clf.predict_proba(X)[:, 1], sample_weight=w); log(f"fold {k} held-out sample {len(rows)} rows, AUC {base:.4f}"); rng = np.random.default_rng(1); out = []
    def drop(idx):
        Xp = X.copy()
        for j in idx: Xp[:, j] = Xp[rng.permutation(len(Xp)), j]
        return base - roc_auc_score(y, clf.predict_proba(Xp)[:, 1], sample_weight=w)
    for j, c in enumerate(cols): out.append((names[c], drop([j])))
    for nm, v in sorted(out, key=lambda r: -r[1])[:20]: print(f"  {nm:16s} AUC drop {v:.4f}")
    for g, lst in GROUPS.items(): print(f"  group {g:5s} AUC drop {drop([cols.index(names.index(c)) for c in lst]):.4f}")
    print("  piece-level pair (log_area, pconf) AUC drop", round(drop([cols.index(names.index(c)) for c in ('log_area', 'pconf')]), 4))


def stage_plot():
    """retained area vs purity curves from the cut_*.json files given as arguments: label=file; right panel: impure area removed vs area lost"""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 5)); sty = ["-o", "-s", "-^", "-v", "-d", "-x", "-*", "-p"]
    for n, spec in enumerate(sys.argv[3:]):
        lab, f = spec.split("="); d = json.load(open(B + f)); pts = d["points"]; k0 = pts[0]; imp0 = k0["area_kept"] - k0["pure_area_kept"]
        ax[0].plot([p["area_kept"] for p in pts], [p["purity"] * 100 for p in pts], sty[n % len(sty)], ms=4, label=lab)
        ax[1].plot([k0["area_kept"] - p["area_kept"] for p in pts], [imp0 - (p["area_kept"] - p["pure_area_kept"]) for p in pts], sty[n % len(sty)], ms=4, label=lab)
    k0 = json.load(open(B + sys.argv[3].split("=")[1]))["points"][0]; imp0 = k0["area_kept"] - k0["pure_area_kept"]
    ax[1].plot([0, 40], [0, 40 * imp0 / k0["area_kept"]], "k:", lw=1, label="random cut (same impure share everywhere)")
    ax[0].axvline(55, color="gray", ls=":", lw=1); ax[0].axhline(93, color="gray", ls=":", lw=1); ax[0].grid(alpha=0.3); ax[1].grid(alpha=0.3)
    ax[0].set_xlabel("retained area (cm2, pieces >= 0.02 cm2)"); ax[0].set_ylabel("area-weighted purity of scored pieces (%)"); ax[0].set_title("purity vs retained area"); ax[0].legend(fontsize=8)
    ax[1].set_xlabel("area lost (cm2)"); ax[1].set_ylabel("impure area removed (cm2)"); ax[1].set_title("impure area removed vs area lost (impure = outside the largest bridge-free sub-piece)", fontsize=9); ax[1].set_xlim(0, 40); ax[1].set_ylim(0, 8); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(B + sys.argv[2], dpi=130); log("saved", B + sys.argv[2])


# ---------------------------------------------------------------- diagnostics (why the labels are hard to learn)
def stage_diag_profile():
    """mean feature value (in sd units, relative to the global mean) of edges binned by geodesic distance to the nearest significant bridge edge"""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
    S = Store(); names = json.load(open(B + "feats.json")); X = np.load(B + "feats.npy", mmap_mode="r"); sig = S.sig; bins = [0, 1, 2, 3, 5, 8, 12, 20, 1e9]
    acc = np.zeros((len(bins) - 1, len(names))); cnt = np.zeros(len(bins) - 1); ref = np.zeros(len(names)); nref = 0; rng = np.random.default_rng(1)
    for i in rng.choice(S.P, 200, replace=False):
        i = int(i); sl = slice(S.eoff[i], S.eoff[i + 1]); sg = np.asarray(sig[sl])
        if (sg == 1).sum() < 20: continue
        V, F, E, fe, lab, valid = S.piece(i); nv = len(V); w = np.linalg.norm(V[E[:, 1]] - V[E[:, 0]], axis=1) / 2.0 + 1e-3
        G = csr_matrix((w, (E[:, 0], E[:, 1])), shape=(nv, nv)); d = dijkstra(G, directed=False, indices=np.unique(E[sg == 1].ravel()), min_only=True)
        dm = 0.5 * (d[E[:, 0]] + d[E[:, 1]]); x = np.asarray(X[sl]).astype(np.float64)
        for j in range(len(bins) - 1):
            m = (dm >= bins[j]) & (dm < bins[j + 1])
            if m.any(): acc[j] += x[m].sum(0); cnt[j] += m.sum()
        ref += x.sum(0); nref += len(x)
    ref /= nref; prof = acc / np.maximum(cnt[:, None], 1); xs = np.asarray(X[::200]).astype(np.float64); sd = np.sqrt(np.maximum((xs ** 2).mean(0) - xs.mean(0) ** 2, 1e-12))
    print("distance bins (grid cells):", bins, "edge counts", cnt.astype(int).tolist())
    for j, nm in enumerate(names): print(f"{nm:14s} z by distance bin: {np.round((prof[:, j] - ref[j]) / sd[j], 2).tolist()}")


def stage_diag_truth():
    """(1) fractional truth turn |delta| at bridge vertices; (2) sp and confidence versus |delta|; (3) bridge rate versus distance to the human-verified patch samples"""
    from scipy.spatial import cKDTree
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import isosurf as iso
    S = Store(); sp = np.load(SP_F, mmap_mode="r"); shp = sp.shape; spf = sp.reshape(-1); conf = np.load(CONF_F, mmap_mode="r").reshape(-1); cs = (256, 1536, 1536)
    rng = np.random.default_rng(3); bins = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5001]; acc = np.zeros((len(bins) - 1, 4)); cnt = np.zeros(len(bins) - 1); pb = []; po = []
    for i in rng.choice(S.P, 60, replace=False):
        i = int(i); V, F, E, fe, lab, valid = S.piece(i)
        ix = np.rint(V / 2.0).astype(np.int64) - iso.TH_ORG; ok = np.all((ix >= 0) & (ix < np.array(iso.TH.shape)), 1); th = np.full(len(V), np.nan); o = np.nonzero(ok)[0]
        srt = o[np.lexsort((ix[o, 2], ix[o, 1], ix[o, 0]))]; th[srt] = iso.TH[ix[srt, 0], ix[srt, 1], ix[srt, 2]]
        q = (th - iso.theta_wrapped(V)) / (2 * np.pi); d = np.abs(q - np.rint(q)); f = np.isfinite(d); vb = np.zeros(len(V), bool); br = lab == 1; vb[E[br, 0]] = True; vb[E[br, 1]] = True
        pb.append(d[vb & f]); po.append(d[~vb & f])
        g = np.minimum(np.maximum(np.rint(V - ORG).astype(np.int64), 0), np.array(shp) - 1); s0 = spf[(g[:, 0] * shp[1] + g[:, 1]) * shp[2] + g[:, 2]].astype(np.float32); smax = np.zeros(len(V), np.float32)
        for dz, dy, dx in ((0, 0, 0), (2, 0, 0), (-2, 0, 0), (0, 2, 0), (0, -2, 0), (0, 0, 2), (0, 0, -2)):
            gg = np.minimum(np.maximum(g + [dz, dy, dx], 0), np.array(shp) - 1); smax = np.maximum(smax, spf[(gg[:, 0] * shp[1] + gg[:, 1]) * shp[2] + gg[:, 2]])
        gc = np.minimum(np.maximum(np.rint((V - ORG - 0.5) / 2).astype(np.int64), 0), np.array(cs) - 1); cf = conf[(gc[:, 0] * cs[1] + gc[:, 1]) * cs[2] + gc[:, 2]].astype(np.float32)
        for j in range(len(bins) - 1):
            m = f & (d >= bins[j]) & (d < bins[j + 1])
            if m.any(): acc[j] += [(s0[m] > 127).sum(), (smax[m] > 127).sum(), cf[m].sum(), m.sum()]; cnt[j] += m.sum()
    b = np.concatenate(pb); o = np.concatenate(po); print("|delta| percentiles (10,50,90,99): bridge vertices", np.round(np.percentile(b, [10, 50, 90, 99]), 3).tolist(), "other vertices", np.round(np.percentile(o, [10, 50, 90, 99]), 3).tolist())
    for j in range(len(bins) - 1): print(f"|delta| {bins[j]:.2f}-{bins[j + 1]:.2f}: vertices {int(cnt[j]):8d}  sp>127 at vertex {acc[j, 0] / cnt[j] * 100:5.1f}%  sp>127 within 2 vox {acc[j, 1] / cnt[j] * 100:5.1f}%  mean conf {acc[j, 2] / cnt[j]:.3f}")
    d = np.load("E:/vesuvius_downstream_tmp/fiberA/patch_samples_ua.npz"); p = d["pts"].astype(np.float64)
    p = p[(p[:, 0] >= 5248) & (p[:, 0] < 5504) & (p[:, 1] >= 960) & (p[:, 1] < 1792) & (p[:, 2] >= 1664) & (p[:, 2] < 2432)]; tree = cKDTree(p)
    bins = [0, 3.5, 6, 10, 20, 40, 80, 1e9]; nb = np.zeros(len(bins) - 1); nt = np.zeros(len(bins) - 1); rng = np.random.default_rng(0)
    for i in range(S.P):
        V, F, E, fe, lab, valid = S.piece(i); Vm = 0.5 * (V[E[:, 0]] + V[E[:, 1]]) / 2.0
        m = (Vm[:, 1] >= 960) & (Vm[:, 1] < 1792) & (Vm[:, 2] >= 1664) & (Vm[:, 2] < 2432) & (lab >= 0)
        if not m.any(): continue
        sel = np.nonzero(m)[0]; sel = sel[rng.random(len(sel)) < 0.2]; dist = tree.query(Vm[sel], workers=2)[0]; y = lab[sel] == 1; nt += np.histogram(dist, bins)[0]; nb += np.histogram(dist[y], bins)[0]
    for j in range(len(bins) - 1): print(f"distance to nearest human patch sample {bins[j]:>5} - {bins[j + 1]:<8} L3 vox: edges {int(nt[j]):9d}  bridge rate {nb[j] / max(nt[j], 1) * 100:.2f}%")


def stage_diag_events():
    """human-verified switch events of the ruler (DUMP file of eval_surfs.py on the uncut SURF): score along the mesh path between the 'on' sample and the 'beside' sample, versus random vertex pairs of the same piece.
    python bridge_clf.py diag_events DUMP.npy TAG"""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
    from scipy.spatial import cKDTree
    S = Store(); dumpf = sys.argv[2]; tag = sys.argv[3] if len(sys.argv) > 3 else "piece_all"; d = np.load(dumpf); names = np.load(dumpf.replace(".npy", "_names.npy"))
    P = np.load("E:/vesuvius_downstream_tmp/fiberA/patch_samples_ua.npz")["pts"].astype(np.float64); poof = np.load(B + f"p_oof_{tag}.npy", mmap_mode="r"); Xmm = np.load(B + "feats.npy", mmap_mode="r"); fn = json.load(open(B + "feats.json"))
    jc = fn.index("conf_min"); jv = fn.index("vdist_min"); rng = np.random.default_rng(0); ev = {}; res = []
    for row in d: ev.setdefault((int(row[0]), int(row[1])), []).append((int(row[2]), int(row[3])))
    print("events", len(ev), "switch samples", len(d))
    for (cid, sid), lst in ev.items():
        i = int(names[sid // 100000].split("_")[1]); V, F, E, fe, lab, valid = S.piece(i); nv = len(V); L3 = V / 2.0; tree = cKDTree(L3); sl = slice(S.eoff[i], S.eoff[i + 1])
        pe = np.asarray(poof[sl]).astype(np.float64); ce = -np.asarray(Xmm[sl, jc]).astype(np.float64); ve = -np.asarray(Xmm[sl, jv]).astype(np.float64)
        w = np.linalg.norm(V[E[:, 1]] - V[E[:, 0]], axis=1) + 1e-3; G = csr_matrix((np.arange(len(E)) + 1, (E[:, 0], E[:, 1])), shape=(nv, nv)); Gs = G + G.T; Gw = csr_matrix((w, (E[:, 0], E[:, 1])), shape=(nv, nv))
        def path_scores(va, vb):
            dist, pred = dijkstra(Gw, directed=False, indices=va, return_predecessors=True)
            if not np.isfinite(dist[vb]): return None
            path = [vb]
            while path[-1] != va: path.append(pred[path[-1]])
            path = np.array(path); ei = np.asarray(Gs[path[:-1], path[1:]]).ravel() - 1
            return pe[ei].max(), (pe[ei] > 0.3).mean(), ce[ei].max(), ve[ei].max(), dist[vb]
        for (sb, sa) in lst[:3]:
            r = path_scores(int(tree.query(P[sa])[1]), int(tree.query(P[sb])[1]))
            if r is None: continue
            sep = np.linalg.norm(P[sb] - P[sa]); ctl = []
            for _ in range(40):
                a = int(rng.integers(nv)); cand = np.nonzero(np.abs(np.linalg.norm(L3 - L3[a], axis=1) - sep) < 0.2 * sep)[0]
                if len(cand) == 0: continue
                rr = path_scores(a, int(rng.choice(cand)))
                if rr is not None: ctl.append(rr)
            res.append((r, np.array(ctl)))
    print("mean rank of the event path among the control paths (0.5 = no signal):", {nm: round(float(np.mean([np.mean(c[:, j] < r[j]) for r, c in res])), 3) for nm, j in (("max p", 0), ("share p>0.3", 1), ("conf cut score", 2), ("vortex cut score", 3))}, "pairs", len(res))


def stage_diag_misc():
    """python bridge_clf.py diag_misc tile|ct|second|oracle
    tile: bridge rate by grid position modulo the sync tile size; ct: CT profile along the normal; second: frozen_8c phase / label disagreement; oracle: whole-piece removal bounds"""
    from sklearn.metrics import roc_auc_score
    S = Store(); which = sys.argv[2]; rng = np.random.default_rng(0)
    if which == "tile":
        for T in (24, 48):
            hb = np.zeros((3, T)); ha = np.zeros((3, T))
            for i in rng.choice(S.P, 150, replace=False):
                V, F, E, fe, lab, valid = S.piece(int(i)); g = (0.5 * (V[E[:, 0]] + V[E[:, 1]]) - ORG - 0.5) / 2.0
                for ax in range(3): ha[ax] += np.bincount(np.floor(g[lab >= 0, ax]).astype(int) % T, minlength=T); hb[ax] += np.bincount(np.floor(g[lab == 1, ax]).astype(int) % T, minlength=T)
            for ax, nm in enumerate("zyx"):
                r = hb[ax] / np.maximum(ha[ax], 1); print(f"T={T} axis {nm}: bridge rate by (cell mod T): min {r.min() * 100:.2f}% max {r.max() * 100:.2f}% mean {hb[ax].sum() / ha[ax].sum() * 100:.2f}%")
    elif which == "ct":
        ct = np.load(CT_F, mmap_mode="r"); shp = ct.shape; ctf = ct.reshape(-1); SPN = 20; s = np.arange(-SPN, SPN + 1); Y = []; Fs = []
        for i in np.random.default_rng(0).choice(S.P, 80, replace=False):
            V, F, E, fe, lab, valid = S.piece(int(i)); N, fn = vertex_normals(V, F); pos = V - ORG; nv = len(V); f = np.zeros((nv, 6), np.float32)
            for c0 in range(0, nv, 40000):
                c1 = min(c0 + 40000, nv); P = pos[c0:c1, None, :] + s[None, :, None] * N[c0:c1, None, :]; ix = np.minimum(np.maximum(np.rint(P).astype(np.int64), 0), np.array(shp) - 1)
                pr = ctf[(ix[..., 0] * shp[1] + ix[..., 1]) * shp[2] + ix[..., 2]].astype(np.float32); pr = np.pad(pr, ((0, 0), (1, 1)), mode="edge"); pr = 0.25 * pr[:, :-2] + 0.5 * pr[:, 1:-1] + 0.25 * pr[:, 2:]
                ismax = np.zeros(pr.shape, bool); ismax[:, 1:-1] = (pr[:, 1:-1] > pr[:, :-2]) & (pr[:, 1:-1] >= pr[:, 2:]); dd = np.abs(np.arange(pr.shape[1]) - SPN)[None, :]; far = ismax & (dd > 4)
                f[c0:c1, 0] = ismax.sum(1); f[c0:c1, 1] = np.minimum(np.where(far, dd, 99).min(1), 40); f[c0:c1, 2] = pr[:, SPN - 4:SPN + 5].max(1) - pr.min(1); f[c0:c1, 3] = pr.std(1); f[c0:c1, 4] = pr[:, SPN]
                f[c0:c1, 5] = np.where(far.any(1), np.where(far, pr, -1).max(1), 0)
            a, b = E[:, 0], E[:, 1]; m = lab >= 0
            ef = np.stack([np.maximum(f[a, 0], f[b, 0]), np.minimum(f[a, 1], f[b, 1]), 0.5 * (f[a, 2] + f[b, 2]), 0.5 * (f[a, 3] + f[b, 3]), 0.5 * (f[a, 4] + f[b, 4]), np.maximum(f[a, 5], f[b, 5])], 1); Y.append(lab[m] == 1); Fs.append(ef[m])
        Y = np.concatenate(Y); Fs = np.concatenate(Fs); print("edges", len(Y), "bridge", Y.sum())
        for j, nm in enumerate(["n maxima", "dist to other max", "centre contrast", "profile std", "ct at vertex", "second max height"]): print(f"{nm:20s} AUC {roc_auc_score(Y, Fs[:, j]):.3f}")
    elif which == "second":
        pf = np.load("D:/vesuvius_big_hot/pred/frozen8c_psi.npy", mmap_mode="r"); uf = np.load("E:/vesuvius_big_hot/u_frozen8c.npy", mmap_mode="r"); shp = pf.shape; pff = pf.reshape(-1); uff = uf.reshape(-1); Y = []; Fs = [[] for _ in range(5)]
        for i in rng.choice(S.P, 120, replace=False):
            i = int(i); V, F, E, fe, lab, valid = S.piece(i); k = S.m["k"][i]; g = np.minimum(np.maximum(np.rint((V - ORG - 0.5) / 2).astype(np.int64), 0), np.array(shp) - 1)
            flat = (g[:, 0] * shp[1] + g[:, 1]) * shp[2] + g[:, 2]; psi_f = pff[flat].astype(np.float32); u_f = uff[flat]; a, b = E[:, 0], E[:, 1]; m = lab >= 0; ru = np.rint(u_f); dk = u_f - k
            vals = [0.5 * (np.abs(wrap(psi_f[a])) + np.abs(wrap(psi_f[b]))), np.abs(ru[a] - ru[b]), np.abs(u_f[a] - u_f[b]), 0.5 * (np.abs(dk - np.rint(dk))[a] + np.abs(dk - np.rint(dk))[b]), np.abs(wrap(psi_f[a] - psi_f[b]))]
            Y.append(lab[m] == 1)
            for j in range(5): Fs[j].append(vals[j][m])
        Y = np.concatenate(Y)
        for nm, f in zip(("frozen |psi| at v6 surface", "frozen label integer jump", "frozen label |du|", "frac(u_frozen - k)", "frozen |dpsi| along edge"), Fs): f = np.concatenate(f); print(f"{nm:30s} AUC {roc_auc_score(Y, f):.3f}")
        Yv = []; X1 = []
        for i in range(S.P):
            V, F, E, fe, lab, valid = S.piece(i); nv = len(V)
            if S.m["area"][i] < 0.05: continue
            keep = lab != 1; nc, sub = cc(nv, E[keep, 0], E[keep, 1]); sa = np.bincount(sub[F[:, 0]], face_area_cm2(V, F), minlength=nc); big = np.argmax(sa)
            if sa[big] / sa.sum() > 0.97: continue
            used = np.zeros(nv, bool); used[F.ravel()] = True; g = np.minimum(np.maximum(np.rint((V - ORG - 0.5) / 2).astype(np.int64), 0), np.array(shp) - 1); u = uff[(g[:, 0] * shp[1] + g[:, 1]) * shp[2] + g[:, 2]]
            d = np.rint(u - S.m["k"][i]); vals_, cnts = np.unique(d, return_counts=True); Yv.append((sub != big)[used]); X1.append((d != vals_[np.argmax(cnts)])[used])
        Yv = np.concatenate(Yv); X1 = np.concatenate(X1); print(f"impure pieces: AUC (frozen label differs from piece mode) vs minority membership {roc_auc_score(Yv, X1):.3f}; share differing: minority {X1[Yv].mean():.3f}, majority {X1[~Yv].mean():.3f}")
    elif which == "oracle":
        rows = []
        for i in range(S.P):
            V, F, E, fe, lab, valid = S.piece(i); r = eval_piece(len(V), F, fe, E, lab, valid, face_area_cm2(V, F), np.ones(len(F), bool)); rows.append(r)
        R = np.concatenate(rows); A, mx, cov = R[:, 0], R[:, 1], R[:, 2]; sc = (A >= AMIN_SCORE) & (cov >= COV_MIN); print("scored pieces", int(sc.sum()), "area", round(float(A[sc].sum()), 2), "purity", round(float(mx[sc].sum() / A[sc].sum()), 4))
        o = np.argsort(-((A - mx) / A)[sc]); Ao, mo = A[sc][o], mx[sc][o]
        for lost in (1, 2, 3, 4, 5, 5.9):
            k = np.searchsorted(np.cumsum(Ao), lost); print(f"oracle whole-piece removal, lose {lost} cm2 ({k} pieces): purity {mo[k:].sum() / Ao[k:].sum() * 100:.2f}%")


if __name__ == "__main__":
    st = sys.argv[1]
    {"unpack": stage_unpack, "truth": stage_truth, "edges": stage_edges, "baseline": stage_baseline, "fields": stage_fields, "feats": stage_feats, "train": stage_train, "univariate": stage_univariate,
     "cut": stage_cut, "select": stage_select, "siglabel": stage_siglabel, "plot": stage_plot, "export": stage_export, "minlabel": stage_minlabel, "importance": stage_importance, "diag_profile": stage_diag_profile, "diag_truth": stage_diag_truth, "diag_events": stage_diag_events, "diag_misc": stage_diag_misc}.get(st, lambda: None)()
