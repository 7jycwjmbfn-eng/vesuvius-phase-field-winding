"""CertNet: certificate for the winding count between two points of box A (L2 coordinates z, y, x) from three parts:
  - the counter network (counter_net; trained on truth pairs only) on the 14-channel profile of the segment,
  - the gradient-boosting certifier of certlib (votes, trained on the human ladders),
  - a stacking model (train_stack.py) that combines them; it predicts whether the network count is right.
  The gm slopes are fitted on the ladder pairs with truth 1-4 (as in train_stack.py, which fits them on the training folds); |cos| as in load_ladder.
  Readings reported before 2026-10-08 (final lines, autoladder_prod3 runs) used the slopes of all ladder pairs and the signed cos.
certify(pairs) returns (count, probability) per pair; a pair is certified when 1 <= count <= 4 and probability >= tau[target].
Profiles follow profiles_truth.py exactly (channels, 1 voxel steps from a, perpendicular lines +-3 voxels)."""
import math, pickle
import numpy as np
import torch
import certlib as C
import counter_net as cn
import train_wfield_8c as tw

NMAX = 256


def interp(vol, c, fns):
    hi = np.array(vol.shape) - 1.0; c = np.clip(c, 0.0, hi); i0 = np.minimum(np.floor(c), hi - 1.0).astype(np.intp); f = c - i0; outs = [np.zeros(len(c)) for _ in fns]
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                w = (f[:, 0] if dz else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dx else 1 - f[:, 2]); v = np.asarray(vol[i0[:, 0] + dz, i0[:, 1] + dy, i0[:, 2] + dx]).astype(np.float32)
                for j, fn in enumerate(fns): outs[j] += w * fn(v)
    return [o.astype(np.float32) for o in outs]


ident = lambda v: v


class CertNet:
    def __init__(self, nn_path="E:/vesuvius_counter/run2/model.pt", stack_path="E:/vesuvius_counter/stack_model.pkl", target=0.99, device="cuda"):
        self.box = C.BoxA(); b = self.box; self.gm8 = np.load(C.H + "las/gm.npy", mmap_mode="r"); self.uz, self.uy, self.ux = tw.umbilicus(); self.dev = torch.device(device if torch.cuda.is_available() else "cpu")
        ck = torch.load(nn_path, map_location="cpu", weights_only=True); cfg = ck["config"]; self.net = cn.CounterNet(s_cols=cfg["s_cols"], s_abs=cfg["s_abs"], c1=cfg["c1"], c2=cfg["c2"], derive=cfg["derive"], dils=cfg["dils"]).to(self.dev); self.net.load_state_dict(ck["state"]); self.net.eval(); self.T = float(ck["temperature"])
        st = pickle.load(open(stack_path, "rb")); self.stack = st["model"]; self.tau = st["taus"][target]; self.target = target
        f2 = ["E:/vesuvius_ladder2_rows_0.npz", "E:/vesuvius_ladder2_rows_1.npz"]; f3 = ["E:/vesuvius_ladder3_rows_0.npz", "E:/vesuvius_ladder3_rows_1.npz"]
        lad = C.load_ladder(f2, f3); typ = (lad["truth"] >= 1) & (lad["truth"] <= 4); n = len(typ); self.s1, self.s2 = C.gm_slopes({k: (v[typ] if hasattr(v, "__len__") and len(v) == n else v) for k, v in lad.items()}); X, cand, _ = C.features(lad, self.s1, self.s2); self.gbm = C.fit(X[typ], (cand == lad["truth"])[typ].astype(int))

    def profile(self, a, b, perp_l2=3.0):
        d = b - a; Lv = float(np.linalg.norm(d)); n = int(math.ceil(Lv)) + 1
        if n > NMAX or Lv < 1.0: return None
        bx = self.box; dirv = d / Lv; ref = np.array([1.0, 0, 0]) if abs(dirv[0]) < 0.9 else np.array([0, 1.0, 0]); perp = np.cross(dirv, ref); perp /= np.linalg.norm(perp)
        pts = a + dirv * np.minimum(np.arange(n), Lv)[:, None]; g = (pts - C.ORG_B - 0.5) / 2.0; sh = perp_l2 / 2.0 * perp; Pm = np.zeros((14, NMAX), np.float32)
        s_, c_ = interp(bx.psi6, np.concatenate([g, g + sh, g - sh]), [np.sin, np.cos]); Pm[2, :n], Pm[3, :n] = s_[:n], c_[:n]; Pm[7, :n], Pm[8, :n] = s_[n:2 * n], c_[n:2 * n]; Pm[9, :n], Pm[10, :n] = s_[2 * n:], c_[2 * n:]
        Pm[4, :n] = np.clip(interp(bx.conf6, g, [ident])[0], 0, 1.2); s_, c_ = interp(bx.psif, g, [np.sin, np.cos]); Pm[5, :n], Pm[6, :n] = s_, c_
        Pm[0, :n] = interp(bx.ctv, pts - C.ORG_B, [ident])[0] / 255.0; Pm[1, :n] = interp(bx.spv, pts - C.ORG_B, [ident])[0] / 255.0
        Pm[11, :n] = interp(bx.cosv, g, [ident])[0] / 255.0; Pm[12, :n] = interp(self.gm8, g / 2.0, [ident])[0] / 255.0; Pm[13, :n] = 1.0
        rv = np.array([0.0, a[1] - np.interp(a[0], self.uz, self.uy), a[2] - np.interp(a[0], self.uz, self.ux)]); ra = float(np.linalg.norm(rv)); rv /= max(ra, 1e-6)
        S = np.array([Lv / 100, ra / 1000, abs(float(np.dot(dirv, rv))), (b[0] - a[0]) / Lv, (a[0] + b[0]) / 2 / 10000, np.nan], np.float32)
        return Pm.astype(np.float16), S

    @torch.no_grad()
    def certify(self, pairs):
        """pairs: list of (a, b) numpy arrays.  Returns counts (int array, 0 when the pair cannot be scored) and stack probabilities."""
        m = len(pairs); counts = np.zeros(m, int); probs = np.zeros(m); idx = []; Ps = []; Ss = []; votes = []
        for i, (a, b) in enumerate(pairs):
            r = self.profile(np.asarray(a, float), np.asarray(b, float))
            if r is None: continue
            idx.append(i); Ps.append(r[0]); Ss.append(r[1]); votes.append(self.box.votes(a, b))
        if not idx: return counts, probs
        P = torch.from_numpy(np.stack(Ps).astype(np.float32)).to(self.dev); S = torch.from_numpy(np.stack(Ss)).to(self.dev)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.dev.type == "cuda"): lg = self.net(P, S)
        P9 = torch.softmax(lg.float() / self.T, 1).cpu().numpy(); c_n = P9.argmax(1); p_n = P9.max(1)
        Tm = np.array([[1.0, *v, 0.0] for v in votes]); D = C.rows_to_dict(Tm); D["cos"] = np.abs(D["cos"]); X, cand, _ = C.features(D, self.s1, self.s2); p_g = self.gbm.predict_proba(X)[:, 1]; c_g = cand; L = np.array([v[-1] for v in votes])
        agree = (c_n == c_g); ent = -(P9 * np.log(np.clip(P9, 1e-6, 1))).sum(1); pn_of_g = P9[np.arange(len(c_n)), np.clip(c_g, 0, 8)]
        Xs = np.c_[p_n, p_g, c_n, c_g, agree, pn_of_g, ent, np.log1p(L), np.sort(P9, 1)[:, -2]]; ps = self.stack.predict_proba(Xs)[:, 1]
        for j, i in enumerate(idx): counts[i] = c_n[j]; probs[i] = ps[j]
        return counts, probs
