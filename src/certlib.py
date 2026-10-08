"""Shared pieces of the winding-count certificate (box A arrays, D = 2 grid; env BOX=B / BOX=P139 selects box B / the PHerc0139 test box, env P139_STUB=1 is the path test of BOX=P139).
  BoxA        : the arrays (v6 phase/confidence, frozen_8c phase, lasagna cos / grad_mag, surface prediction, CT) and votes(a, b) for two points in L2 coordinates (z, y, x)
  votes(a, b) : tuple (v6, off1, off2, frozen, path_a, path_b, gm, gm_path, cos_peaks, sp_straight, sp_path, ct_peaks, minconf, meanconf, path_minconf, L2_length), same columns as truth_pairs.py rows[:, 1:17]
  load_ladder(): human-ladder rows (ladder_run2 + ladder_run3) joined into the common feature dictionary
  features(D, s1, s2) / fit(X, y): feature matrix of the learned certifier and its gradient-boosting fit"""
import os, json
import numpy as np
from scipy import ndimage as ndi
from skimage.graph import route_through_array
from sklearn.ensemble import HistGradientBoostingClassifier

BOX = os.environ.get("BOX", "A")                                                                                    # env BOX=B: the arrays of box B (origin L2 10496, 640, 1792); BOX=P139: PHerc0139 test box (origin L2 4352, 1664, 1280)
P139_STUB = os.environ.get("P139_STUB") == "1"                                                                      # BOX=P139 only: path test, v6 phase := frozen_8c phase, v6 confidence := 1 (no accuracy meaning)
if BOX == "B": H = "E:/vesuvius_boxB_hot/"; ORG_B = np.array([10496, 640, 1792.]); FROZ = H + "pred/frozen8c_psi.npy"; SPCT = "G:/vesuvius_boxB_cold/"
elif BOX == "P139": H = "E:/vesuvius_p0139_hot/"; ORG_B = np.array([4352, 1664, 1280.]); FROZ = H + "pred/frozen8c_d2.npy"; SPCT = "G:/vesuvius_p0139_cold/"
else: H = "E:/vesuvius_big_hot/"; ORG_B = np.array([10496, 1920, 3328.]); FROZ = "D:/vesuvius_big_hot/pred/frozen8c_psi.npy"; SPCT = "D:/vesuvius_big_hot/"


def peaks(prof, win=25):
    s = ndi.gaussian_filter1d(prof.astype(np.float64), 1.5); d = s - ndi.uniform_filter1d(s, win); sd = d.std() + 1e-6
    return len([q for q in range(1, len(d) - 1) if d[q] >= d[q - 1] and d[q] > d[q + 1] and d[q] > 0.5 * sd])


class BoxA:
    def __init__(self):
        if BOX == "P139":                                                                                                 # no registered lasagna: cos and grad_mag are all-zero read-only views (no memory, no disk), those two votes carry no information
            const = lambda v, shape, dt: np.broadcast_to(np.asarray(v, dt), shape); self.psif = np.load(FROZ, mmap_mode="r")
            if P139_STUB: self.psi6 = self.psif; self.conf6 = const(1, self.psif.shape, np.float16)
            else: self.psi6 = np.load(H + "pred/v6nolas_psi.npy", mmap_mode="r"); self.conf6 = np.load(H + "pred/v6nolas_conf.npy", mmap_mode="r")
            self.cosv = const(0, (512, 768, 768), np.uint8); self.gmv = const(0, (256, 384, 384), np.float32)
        else:
            self.psi6 = np.load(H + "pred_v6_psi.npy", mmap_mode="r"); self.conf6 = np.load(H + "pred_v6_conf.npy", mmap_mode="r"); self.psif = np.load(FROZ, mmap_mode="r")
            self.cosv = np.load(H + "las/cos.npy", mmap_mode="r"); self.gmv = np.asarray(np.load(H + "las/gm.npy", mmap_mode="r")).astype(np.float32) / 1000.0
        self.spv = np.load(SPCT + "sp_big.npy", mmap_mode="r"); self.ctv = np.load(SPCT + "ct_big.npy", mmap_mode="r")

    def votes(self, a, b):
        ga = (np.asarray(a, float) - ORG_B - 0.5) / 2.0; gb = (np.asarray(b, float) - ORG_B - 0.5) / 2.0; Lg = float(np.linalg.norm(gb - ga)); n = int(max(Lg * 2, 8)); line = ga + (gb - ga) * np.linspace(0, 1, n + 1)[:, None]
        lo = np.floor(np.minimum(ga, gb) - 8).astype(int).clip(0); hi = np.ceil(np.maximum(ga, gb) + 9).astype(int)
        sl = tuple(slice(l_, h_) for l_, h_ in zip(lo, hi)); P6 = np.asarray(self.psi6[sl]).astype(np.float32); C6 = np.asarray(self.conf6[sl]).astype(np.float32); PF = np.asarray(self.psif[sl]).astype(np.float32)
        loc = line - lo; samp = lambda vol, c: ndi.map_coordinates(vol, c.T, order=1, mode="nearest")

        def count(P, c):
            a_ = np.unwrap(np.arctan2(samp(np.sin(P), c), samp(np.cos(P), c))); return (a_[-1] - a_[0]) / (2 * np.pi)
        v6 = count(P6, loc); dirv = (gb - ga) / max(Lg, 1e-6); ref = np.array([1.0, 0, 0]) if abs(dirv[0]) < 0.9 else np.array([0, 1.0, 0]); perp = np.cross(dirv, ref); perp /= np.linalg.norm(perp)
        off1 = count(P6, loc + 3 * perp); off2 = count(P6, loc - 3 * perp); fr = count(PF, loc); pa = []
        for power in (2, 4):
            cost = 1.0 / (np.clip(C6, 0.02, 1.2) ** power) + 0.05; ia = tuple(np.rint(ga - lo).astype(int).clip(0, np.array(cost.shape) - 1)); ib = tuple(np.rint(gb - lo).astype(int).clip(0, np.array(cost.shape) - 1))
            path, _ = route_through_array(cost, ia, ib, fully_connected=True, geometric=True); path = np.array(path); an = np.unwrap(np.arctan2(np.sin(P6)[tuple(path.T)], np.cos(P6)[tuple(path.T)]))
            pa.append(((an[-1] - an[0]) / (2 * np.pi), float(C6[tuple(path.T)].min()), path + lo))
        cf = samp(C6, loc); kk = ndi.map_coordinates(np.asarray(self.cosv[sl]).astype(np.float32) / 255.0, loc.T, order=1, mode="nearest"); ks = ndi.uniform_filter1d(kk, 3)
        pk = [q for q in range(1, len(ks) - 1) if ks[q] >= ks[q - 1] and ks[q] > ks[q + 1] and ks[q] > 0.6]
        cnt_cos = len(pk) + (1 if ks[0] > 0.6 and (len(pk) == 0 or pk[0] > 2) else 0) + (1 if ks[-1] > 0.6 and (len(pk) == 0 or pk[-1] < len(ks) - 3) else 0) - 1
        la = np.rint(line * 2.0 + 0.5).astype(int).clip(0, np.array(self.spv.shape) - 1); spl = np.asarray(self.spv[tuple(la.T)]) > 127; sp_s = int((spl[1:] & ~spl[:-1]).sum())
        pp = np.rint(pa[0][2] * 2.0 + 0.5).astype(int).clip(0, np.array(self.spv.shape) - 1); spp = np.asarray(self.spv[tuple(pp.T)]) > 127; sp_p = int((spp[1:] & ~spp[:-1]).sum())
        gmi = float(ndi.map_coordinates(self.gmv, (line / 2.0).T, order=1, mode="nearest").sum() * (Lg * 2.0 / n)); gmp = float(self.gmv[tuple(np.clip(pa[0][2] // 2, 0, np.array(self.gmv.shape) - 1).T)].sum())
        prof = np.asarray(self.ctv[tuple(la.T)]).astype(np.float32); ct_p = peaks(prof)
        return (v6, off1, off2, fr, pa[0][0], pa[1][0], gmi, gmp, cnt_cos, sp_s, sp_p, ct_p, float(cf.min()), float(cf.mean()), pa[0][1], Lg * 2.0)


def rows_to_dict(T, with_truth=True):
    """T columns: k, v6, off1, off2, frozen, path_a, path_b, gm, gm_path, cos, sp_s, sp_p, ct, minc, meanc, pmin, L, z (truth_pairs rows)"""
    return dict(truth=T[:, 0].astype(int), V=np.abs(T[:, [1, 4, 2, 3, 5, 6]]), gm1=T[:, 7], gm2=T[:, 8] * 2.0, cos=T[:, 9], sp1=T[:, 10], sp2=T[:, 11], ct=T[:, 12], minc=T[:, 13], meanc=T[:, 14], pmin=T[:, 15], L=T[:, 16])


def load_ladder(f2, f3, excl_z=True, excl_box=True):
    R2 = np.concatenate([np.load(f)["rows"] for f in f2 if np.load(f)["rows"].size]); R3 = np.concatenate([np.load(f)["rows"] for f in f3 if np.load(f)["rows"].size])
    idx2 = {}; cnt = {}
    for i, (ri, dw) in enumerate(R2[:, :2]):
        ri = int(ri); k = cnt.get(ri, 0); cnt[ri] = k + 1; idx2[(ri, k)] = i
    cnt = {}; s2 = []; s3 = []; sq = []
    for j, (ri, dw) in enumerate(R3[:, :2]):
        ri = int(ri); k = cnt.get(ri, 0); cnt[ri] = k + 1; i = idx2.get((ri, k))
        if i is not None and abs(R2[i, 1] - dw) < 1e-9: s2.append(i); s3.append(j); sq.append(k)
    R2 = R2[s2]; R3 = R3[s3]; sq = np.array(sq, int)
    if excl_z: m = ~((R2[:, 15] >= 9984) & (R2[:, 15] < 10500)); R2 = R2[m]; R3 = R3[m]; sq = sq[m]
    if excl_box:
        regs = json.load(open("E:/vesuvius_ladder_regions.json")); tb_lo = np.array([9984, 2304, 3584]); tb_hi = tb_lo + np.array([1024, 1536, 1536])
        bad = [i for i, r in enumerate(regs) if np.all(np.array(r["lo"]) < tb_hi) and np.all(np.array(r["hi"]) > tb_lo)]; m = ~np.isin(R2[:, 0].astype(int), bad); R2 = R2[m]; R3 = R3[m]; sq = sq[m]
    return dict(truth=np.abs(R2[:, 1]).astype(int), fold=R2[:, 0].astype(int) % 5, ri=R2[:, 0].astype(int), seq=sq, z=R2[:, 15], V=np.abs(np.c_[R2[:, 2], R2[:, 5], R2[:, 7], R2[:, 8], R3[:, 2], R3[:, 3]]), gm1=R2[:, 10], gm2=R3[:, 9], cos=np.abs(R2[:, 9]), sp1=R2[:, 11], sp2=R3[:, 6],
                ct=R3[:, 7], minc=R2[:, 12], meanc=R2[:, 13], pmin=R3[:, 4], L=R2[:, 14])


def gm_slopes(lad):
    ok = np.isfinite(lad["gm1"]); s1 = float(np.sum(lad["gm1"][ok] * lad["truth"][ok]) / np.sum(lad["gm1"][ok] ** 2)); ok2 = np.isfinite(lad["gm2"]); s2 = float(np.sum(lad["gm2"][ok2] * lad["truth"][ok2]) / np.sum(lad["gm2"][ok2] ** 2)); return s1, s2


def features(D, slope1, slope2):
    V = np.rint(np.nan_to_num(D["V"], nan=-1)).astype(int); Vv = np.where(np.isfinite(D["V"]), V, -1); g1 = np.abs(D["gm1"] * slope1); g2 = np.abs(D["gm2"] * slope2); gc1 = np.rint(g1).astype(int)
    mode = np.array([np.bincount(v[v >= 0], minlength=12).argmax() if (v >= 0).any() else 0 for v in Vv]); cand = np.where((V == gc1[:, None]).sum(1) >= (Vv == mode[:, None]).sum(1), gc1, mode)
    X = np.c_[(Vv == cand[:, None]).sum(1), (Vv >= 0).sum(1), (Vv == gc1[:, None]).sum(1), (D["cos"] == cand), (np.rint(np.nan_to_num(g2, nan=-9)) == cand), np.abs(g1 - gc1), np.abs(g1 - cand), np.nan_to_num(np.abs(g2 - cand), nan=-1),
              (D["ct"] == cand), (D["sp1"] == cand), (D["sp2"] == cand), np.abs(D["ct"] - cand), D["minc"], D["meanc"], np.nan_to_num(D["pmin"], nan=-1), np.log1p(D["L"]), np.abs(D["V"][:, :4] - cand[:, None]).mean(1), np.nan_to_num(np.abs(D["V"][:, 4] - cand), nan=-1), cand]
    return X, cand, g1


def fit(X, y): return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, max_leaf_nodes=15, l2_regularization=1.0, random_state=0).fit(X, y)
