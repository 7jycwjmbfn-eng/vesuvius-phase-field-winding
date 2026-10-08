"""wfield v6: one-shot design, trained from scratch.
  inputs (8 channels): CT, official surface prediction, outward radial direction (2), official lasagna nx, ny, grad_mag (4x down) and cos (2x down), up-sampled to the crop grid
  model: 3D U-Net with 5 levels (24-48-96-192-288), GroupNorm + SiLU; 2 + 24 output channels (sin/cos regression + 24-bin phase classification); gradient checkpointing
  crops 112^3, batch 2 x 2 accumulation steps; EMA of the weights (decay 0.999) is what is validated and saved
  losses: validity-weighted MSE + soft-label cross-entropy (v3), phase-gradient matching (v5), physical smoothness hinge on all voxels (v5)
  augmentation: flips (radial and normal components change sign), intensity / noise, surface-prediction scale / dropout (v3), obscuration blobs incl. the lasagna channels (v4),
  random drop of the whole lasagna group (15%, robustness for scrolls without it)
  python train_v6.py OUTDIR [STEPS] [WORKERS]
Data: ct_w2.npy / sp_w2.npy / psi truth as in train_wfield_8c.py (z 10000-10380 train, z 10388-10500 validation), lasagna npy from assemble_las.py."""
import os, sys, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from scipy import ndimage as ndi
import train_wfield_8c as tw
import train_v3 as t3
import train_v4 as t4
import train_v5 as t5
import archs as A

NB = t3.NB; CROP = 112; ORG = tw.ORG                                                      # ORG = (9984, 2304, 3584), multiples of 4
LAS = "G:/vesuvius_big/lasagna_oldbox_npy/"; TRAIN_Z = (10000, 10380); VAL_Z = (10388, 10500)
CIN = 8


# ------------------------------------------------------------------ data
class CropsV6(torch.utils.data.Dataset):
    def __init__(self, zr, n, seed=0, aug=True, p_obscure=0.7, force=False):
        self.zr, self.n, self.seed, self.aug, self.p, self.force = zr, n, seed, aug, p_obscure, force; self.ct = None

    def __len__(self): return self.n

    def _open(self):
        self.ct = np.load(tw.D + "ct_w2.npy", mmap_mode="r"); self.sp = np.load(tw.D + "sp_w2.npy", mmap_mode="r"); self.lab = np.load(tw.LAB, mmap_mode="r")
        self.uz, self.uy, self.ux = tw.umbilicus(); self.las = {k: np.load(LAS + f"{k}.npy", mmap_mode="r") for k in ("nx", "ny", "gm", "cos")}

    def _las_crop(self, z, y, x):
        """z, y, x: box-relative origin of the crop (multiples of 4) -> [4, C, C, C] float32: nx, ny (-1..1), grad_mag, cos (0..1) up-sampled to the L2 grid"""
        c4 = CROP // 4; c2 = CROP // 2; out = []
        for k, f, dec in (("nx", 4, lambda a: (a - 128.0) / 127.0), ("ny", 4, lambda a: (a - 128.0) / 127.0), ("gm", 4, lambda a: a / 255.0), ("cos", 2, lambda a: a / 255.0)):
            cc = CROP // f; a = np.asarray(self.las[k][z // f:z // f + cc, y // f:y // f + cc, x // f:x // f + cc]).astype(np.float32)
            if a.shape != (cc, cc, cc): a = np.pad(a, [(0, cc - a.shape[0]), (0, cc - a.shape[1]), (0, cc - a.shape[2])], mode="edge")
            t = torch.from_numpy(dec(a))[None, None]; out.append(F.interpolate(t, scale_factor=f, mode="trilinear", align_corners=False)[0, 0].numpy())
        return np.stack(out)

    def __getitem__(self, i):
        if self.ct is None: self._open()
        rng = np.random.default_rng((self.seed, i, int(time.time() * 1e6) % 2**31) if self.aug else (self.seed, i)); S = self.ct.shape
        for _ in range(60):
            z = int(rng.integers(self.zr[0] - ORG[0], self.zr[1] - ORG[0] - CROP + 1)) // 4 * 4; y = int(rng.integers(0, S[1] - CROP)) // 4 * 4; x = int(rng.integers(0, S[2] - CROP)) // 4 * 4
            if (np.asarray(self.lab[z + CROP // 2, y:y + CROP:4, x:x + CROP:4]) < 255).mean() < 0.2: continue
            ct = np.asarray(self.ct[z:z + CROP, y:y + CROP, x:x + CROP])
            if (ct > 0).mean() > 0.5: break
        sp = np.asarray(self.sp[z:z + CROP, y:y + CROP, x:x + CROP]); zz = ORG[0] + z + CROP / 2
        cy, cx = np.interp(zz, self.uz, self.uy), np.interp(zz, self.uz, self.ux)
        gy = ORG[1] + y + np.arange(CROP)[:, None] - cy; gx = ORG[2] + x + np.arange(CROP)[None, :] - cx; r = np.hypot(gy, gx) + 1e-6
        ry = np.broadcast_to((gy / r)[None], (CROP,) * 3); rx = np.broadcast_to((gx / r)[None], (CROP,) * 3)
        L = np.asarray(self.lab[z:z + CROP, y:y + CROP, x:x + CROP]); m = L < 255; psi = L.astype(np.float32) * (2 * np.pi / 255)
        las = self._las_crop(z, y, x)
        X = np.concatenate([np.stack([ct / 255.0, sp / 255.0, ry, rx]).astype(np.float32), las]).astype(np.float32)               # [8, C, C, C]
        Y = np.stack([np.sin(psi), np.cos(psi)]).astype(np.float32)
        frac = ndi.uniform_filter(m.astype(np.float32), 9, mode="nearest"); w = np.where(m, np.clip(frac, 0.2, 1.0) ** 2, 0.0).astype(np.float32)
        bm = np.zeros(m.shape, bool)
        if self.aug:
            for ax, chs in ((1, ()), (2, (2, 4)), (3, (3, 5))):                                                              # flip along y: ry and ny change sign; along x: rx and nx
                if rng.random() < 0.5:
                    X = np.flip(X, ax).copy(); Y = np.flip(Y, ax).copy(); m = np.flip(m, ax - 1).copy(); w = np.flip(w, ax - 1).copy()
                    for ch in ((2, 5) if ax == 2 else (3, 4) if ax == 3 else ()): X[ch] = -X[ch]
            X[0] = np.clip(X[0] * rng.uniform(0.85, 1.15) + rng.uniform(-0.05, 0.05) + rng.normal(0, 0.02, X[0].shape), 0, 1)
            X[1] = 0 if rng.random() < 0.15 else np.clip(X[1] * rng.uniform(0.8, 1.2), 0, 1)
            if rng.random() < 0.15: X[4:8] = 0                                                                                # drop the whole lasagna group
        if self.force or (self.aug and rng.random() < self.p):
            X, bm = self._obscure(X, rng)
        return torch.from_numpy(np.ascontiguousarray(X)).half(), torch.from_numpy(np.ascontiguousarray(Y)).half(), torch.from_numpy(np.ascontiguousarray(m)), torch.from_numpy(np.ascontiguousarray(w)).half(), torch.from_numpy(bm)

    @staticmethod
    def _obscure(X, rng):
        """obscuration blobs; every filter runs only inside the bounding box of its ellipsoid (about 8x cheaper than the whole crop)"""
        tot = np.zeros(X.shape[1:], np.float32); out = X.copy(); n = int(rng.integers(1, 4)); Z, Y_, X_ = X.shape[1:]
        for _ in range(n):
            c = rng.uniform([0.2 * s for s in (Z, Y_, X_)], [0.8 * s for s in (Z, Y_, X_)]); r = rng.uniform(16, 32, 3) * rng.uniform(0.8, 1.25, 3)
            lo = np.maximum(np.floor(c - r - 6).astype(int), 0); hi = np.minimum(np.ceil(c + r + 6).astype(int), [Z, Y_, X_]); sl = tuple(slice(l, h) for l, h in zip(lo, hi))
            zz, yy, xx = np.ogrid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
            m = np.clip(1.5 * (1 - (((zz - c[0]) / r[0]) ** 2 + ((yy - c[1]) / r[1]) ** 2 + ((xx - c[2]) / r[2]) ** 2)), 0, 1).astype(np.float32); kind = rng.choice(["blur", "blank", "nosp"], p=[0.6, 0.25, 0.15])
            if kind == "blur":
                alt = {0: ndi.gaussian_filter(X[0][sl], 4.0, truncate=2.0), 1: np.clip(ndi.gaussian_filter(X[1][sl], 3.0, truncate=2.0) * 1.8, 0, 1)}
                for k in range(4, 8): alt[k] = ndi.gaussian_filter(X[k][sl], 3.0, truncate=2.0)
            elif kind == "blank":
                ct = X[0][sl]; alt = {0: np.full_like(ct, ct[m > 0.5].mean() if (m > 0.5).any() else ct.mean()) + rng.normal(0, 0.02, ct.shape).astype(np.float32), 1: np.full_like(ct, 0.8)}
                for k in range(4, 8): alt[k] = np.zeros_like(ct)
            else:
                alt = {1: np.zeros_like(X[1][sl])}
            for ch, a_ in alt.items(): out[ch][sl] = out[ch][sl] * (1 - m) + a_ * m
            tot[sl] = np.maximum(tot[sl], m)
        out[0] = np.clip(out[0], 0, 1); out[1] = np.clip(out[1], 0, 1)
        return out, tot > 0.5


# ------------------------------------------------------------------ model
def block(ci, co):
    return nn.Sequential(nn.Conv3d(ci, co, 3, padding=1, bias=False), nn.GroupNorm(8, co), nn.SiLU(inplace=True),
                         nn.Conv3d(co, co, 3, padding=1, bias=False), nn.GroupNorm(8, co), nn.SiLU(inplace=True))


class UNet5(nn.Module):
    def __init__(self, cin=CIN, ch=(24, 48, 96, 192, 288), ckpt=True):
        super().__init__(); self.ckpt = ckpt
        self.enc = nn.ModuleList([block(cin if i == 0 else ch[i - 1], c) for i, c in enumerate(ch)])
        self.down = nn.ModuleList([nn.Conv3d(c, c, 2, stride=2) for c in ch[:-1]])
        self.up = nn.ModuleList([nn.ConvTranspose3d(ch[i + 1], ch[i], 2, stride=2) for i in range(len(ch) - 1)])
        self.dec = nn.ModuleList([block(2 * ch[i], ch[i]) for i in range(len(ch) - 1)]); self.head = nn.Conv3d(ch[0], 2 + NB, 1)

    def _run(self, f, x):
        return cp.checkpoint(f, x, use_reentrant=False) if (self.ckpt and self.training) else f(x)

    def forward(self, x, full=False):
        skips = []
        for i, e in enumerate(self.enc):
            x = self._run(e, x)
            if i < len(self.down): skips.append(x); x = self.down[i](x)
        for i in reversed(range(len(self.up))):
            x = self._run(self.dec[i], torch.cat([self.up[i](x), skips[i]], 1))
        o = self.head(x)
        if full: return o
        p = torch.softmax(o[:, 2:].float(), 1); c = t3.CENT.to(o.device).view(1, NB, 1, 1, 1)
        return torch.stack([(p * torch.sin(c)).sum(1), (p * torch.cos(c)).sum(1)], 1)


# ------------------------------------------------------------------ train
def validate(model, vclean, vobs):
    """returns {readout: [clean core median, clean core acc, clean all acc, obscured core median, obscured core acc]} for readouts reg / cls / avg"""
    model.eval(); res = {k: [] for k in ("reg", "cls", "avg")}
    with torch.no_grad():
        for vs, only_blob in ((vclean, False), (vobs, True)):
            E = {k: [] for k in res}; Wt = []
            for i in range(0, len(vs), 2):
                b = vs[i:i + 2]; x = torch.stack([v[0] for v in b]).cuda().float(); y = torch.stack([v[1] for v in b]).cuda().float(); m = torch.stack([v[2] for v in b]).cuda()
                w = torch.stack([v[3] for v in b]).cuda().float(); bm = torch.stack([v[4] for v in b]).cuda()
                with torch.autocast("cuda", dtype=torch.bfloat16): o = model(x, full=True)
                sel = m & (bm if only_blob else torch.ones_like(m)); Wt.append(w[sel].cpu())
                for k in res:
                    p = A.Net.readout(o.float(), k); d = torch.atan2(p[:, 0], p[:, 1]) - torch.atan2(y[:, 0], y[:, 1]); d = torch.atan2(torch.sin(d), torch.cos(d)).abs(); E[k].append(d[sel].cpu())
            w = torch.cat(Wt); core = w >= 0.9
            for k in res:
                e = torch.cat(E[k]); res[k] += [float(np.degrees(e[core].median())), float((e[core] < np.pi / 2).float().mean())] + ([float((e < np.pi / 2).float().mean())] if not only_blob else [])
    return res


def main(out, steps=25000, workers=5, batch=2, accum=2):
    torch.backends.cudnn.benchmark = True; os.makedirs(out, exist_ok=True)
    ARCH = os.environ.get("ARCH", "plain"); model = A.Net(ARCH).cuda(); ema = copy.deepcopy(model).eval()
    for p in ema.parameters(): p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=1e-4); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1.2e-3, total_steps=steps, pct_start=0.05)
    start = 0; last = os.path.join(out, "last.pt")
    if os.path.exists(last):
        ck = torch.load(last, map_location="cuda"); model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); start = ck["step"]; print(f"resumed from {start}", flush=True)
    dl = torch.utils.data.DataLoader(CropsV6(TRAIN_Z, (steps - start) * batch * accum, seed=start + 21), batch_size=batch, num_workers=workers, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    vclean = [CropsV6(VAL_Z, 48, seed=123, aug=False, p_obscure=0)[i] for i in range(48)]; vobs = [CropsV6(VAL_Z, 48, seed=123, aug=False, force=True)[i] for i in range(48)]
    print(f"arch {ARCH} params {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M, crop {CROP}, batch {batch} x {accum}, steps {steps}, start {start}", flush=True)
    t0 = time.time(); best = -1; model.train(); k = 0; step = start; opt.zero_grad(set_to_none=True)
    for (x, y, m, w, bm) in dl:
        x = x.cuda(non_blocking=True).float(); y = y.cuda(non_blocking=True).float(); m = m.cuda(non_blocking=True); w = w.cuda(non_blocking=True).float()
        with torch.autocast("cuda", dtype=torch.bfloat16): o = model(x, full=True)
        loss, reg = t3.loss_fn(o, y, m, w); gl = t5.grad_loss(o, y, m, w); sl = t5.smooth_hinge(o, x[:, 0]); total = (loss + t5.GW * gl + t5.SW * sl) / accum; total.backward(); k += 1
        if k % accum: continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
        d = min(0.999, (1 + step) / (10 + step))
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()): pe.mul_(d).add_(pm.detach(), alpha=1 - d)
        if step % 100 == 0: print(f"step {step} loss {loss.item():.4f} grad {gl.item():.4f} hinge {sl.item():.4f} reg {reg:.4f} lr {sched.get_last_lr()[0]:.2e} | {k * batch / (time.time() - t0):.1f} crops/s", flush=True)
        if step % 500 == 0 or step == steps:
            R = validate(ema, vclean, vobs); model.train()
            for rk, (mc, ac, aall, mo, ao) in R.items(): print(f"  val step {step} (EMA, {rk}): clean core median {mc:.1f} deg acc<90 {ac * 100:.1f}% | clean all acc {aall * 100:.1f}% | obscured core median {mo:.1f} deg acc<90 {ao * 100:.1f}%", flush=True)
            mc, ac, aall, mo, ao = R["reg"]
            if step % 1000 == 0 or step == steps:
                torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step}, last + ".tmp"); os.replace(last + ".tmp", last)
                torch.save(ema.state_dict(), os.path.join(out, f"ema_{step}.pt"))
            score = 0.5 * ac + 0.5 * (ao if ao == ao else 0)
            if score > best: best = score; torch.save(ema.state_dict(), os.path.join(out, "ema_best.pt"))
        if step >= steps: break
    torch.save(ema.state_dict(), os.path.join(out, "ema_final.pt")); print("done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 25000, int(sys.argv[3]) if len(sys.argv) > 3 else 5)
