"""wfield v4 = v3 + obscuration augmentation (PREREG_wfield_v4_1006.md).
Why: the pieces of the psi = 0 level set merge neighbouring turns mostly in bright adhesion zones that have NO truth (the official segments do not
reach them), so the network was never supervised there.  Here random ellipsoids of the input are made to look stuck together (blurred CT, filled-in
surface prediction, blank CT, or missing surface prediction) while the labels stay those of the clean data, so the network has to infer the hidden sheets
from the surroundings.  Training augmentation only; every evaluation is on real data.
  python train_v4.py OUTDIR [BATCH] [WORKERS] [STEPS]
Everything else (loss weights, 24-bin head, schedule, resumable checkpoints) is train_v3.py."""
import os, sys, time
import numpy as np
import torch
from scipy import ndimage as ndi
import train_wfield_8c as tw
import train_v3 as t3


def blobs(shape, rng, n=None):
    """soft ellipsoid masks [0,1]; returns (mask, list of kinds)"""
    n = int(rng.integers(1, 4)) if n is None else n; Z, Y, X = shape; m = np.zeros(shape, np.float32); kinds = []
    zz, yy, xx = np.ogrid[:Z, :Y, :X]
    for _ in range(n):
        c = rng.uniform([0.2 * s for s in shape], [0.8 * s for s in shape]); r = rng.uniform(16, 32, 3) * rng.uniform(0.8, 1.25, 3)
        d = ((zz - c[0]) / r[0]) ** 2 + ((yy - c[1]) / r[1]) ** 2 + ((xx - c[2]) / r[2]) ** 2
        m = np.maximum(m, np.clip(1.5 * (1 - d), 0, 1)); kinds.append(rng.choice(["blur", "blank", "nosp"], p=[0.6, 0.25, 0.15]))
    return m, kinds


def obscure(X, rng, n=None):
    """X float16 tensor [4, D, H, W] (ct, sp, ry, rx); returns obscured copy and the blob mask"""
    ct = X[0].float().numpy(); sp = X[1].float().numpy(); out_ct, out_sp = ct.copy(), sp.copy(); tot = np.zeros(ct.shape, np.float32)
    for _ in range(int(rng.integers(1, 4)) if n is None else n):
        m, kinds = blobs(ct.shape, rng, 1); kind = kinds[0]
        if kind == "blur":
            alt_ct = ndi.gaussian_filter(ct, 4.0); alt_sp = np.clip(ndi.gaussian_filter(sp, 3.0) * 1.8, 0, 1)
        elif kind == "blank":
            alt_ct = np.full_like(ct, ct[m > 0.5].mean() if (m > 0.5).any() else ct.mean()) + rng.normal(0, 0.02, ct.shape); alt_sp = np.full_like(sp, 0.8)
        else:
            alt_ct = ct; alt_sp = np.zeros_like(sp)
        out_ct = out_ct * (1 - m) + alt_ct * m; out_sp = out_sp * (1 - m) + alt_sp * m; tot = np.maximum(tot, m)
    Y = X.clone(); Y[0] = torch.from_numpy(np.clip(out_ct, 0, 1)).half(); Y[1] = torch.from_numpy(np.clip(out_sp, 0, 1)).half()
    return Y, torch.from_numpy(tot > 0.5)


class CropsV4(t3.CropsV3):
    def __init__(self, zr, n, seed=0, aug=True, p_obscure=0.7, force=False):
        super().__init__(zr, n, seed, aug); self.p = p_obscure; self.force = force

    def __getitem__(self, i):
        X, Y, m, w = super().__getitem__(i)
        rng = np.random.default_rng((self.seed, i, 777, int(time.time() * 1e6) % 2**31 if self.aug else 0))
        if self.force or (self.aug and rng.random() < self.p):
            X, bm = obscure(X, rng)
        else:
            bm = torch.zeros_like(m)
        return X, Y, m, w, bm


def validate(model, vclean, vobs):
    """clean val as in v3, plus accuracy on the obscured voxels of the same crops with obscuration applied"""
    r = t3.validate(model, [v[:4] for v in vclean]); model.eval(); E = []; Wt = []
    with torch.no_grad():
        for i in range(0, len(vobs), 4):
            b = vobs[i:i + 4]; x = torch.stack([v[0] for v in b]).cuda().float(); y = torch.stack([v[1] for v in b]).cuda().float()
            m = torch.stack([v[2] for v in b]).cuda(); w = torch.stack([v[3] for v in b]).cuda().float(); bm = torch.stack([v[4] for v in b]).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = model(x)
            p = p.float(); d = torch.atan2(p[:, 0], p[:, 1]) - torch.atan2(y[:, 0], y[:, 1]); d = torch.atan2(torch.sin(d), torch.cos(d)).abs()
            sel = m & bm & (w >= 0.9); E.append(d[sel].cpu())
    e = torch.cat(E); return r + (float(np.degrees(e.median())) if len(e) else float("nan"), float((e < np.pi / 2).float().mean()) if len(e) else float("nan"))


def main(out, batch=4, workers=6, steps=30000):
    torch.backends.cudnn.benchmark = True; os.makedirs(out, exist_ok=True)
    model = t3.V3Net().cuda(); opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1.5e-3, total_steps=steps, pct_start=0.05)
    start = 0; last = os.path.join(out, "last.pt")
    if os.path.exists(last):
        ck = torch.load(last, map_location="cuda"); model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); start = ck["step"]
        print(f"resumed from step {start}", flush=True)
    dl = torch.utils.data.DataLoader(CropsV4(tw.TRAIN_Z, (steps - start) * batch, seed=start + 1), batch_size=batch, num_workers=workers, pin_memory=True,
                                     persistent_workers=True, prefetch_factor=2)
    vclean = [CropsV4(tw.VAL_Z, 64, seed=123, aug=False, p_obscure=0)[i] for i in range(64)]; vobs = [CropsV4(tw.VAL_Z, 64, seed=123, aug=False, force=True)[i] for i in range(64)]
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M, batch {batch}, workers {workers}, steps {steps}, start {start}", flush=True)
    best = -1; t0 = time.time(); model.train()
    for k, (x, y, m, w, bm) in enumerate(dl):
        step = start + k
        x = x.cuda(non_blocking=True).float(); y = y.cuda(non_blocking=True).float(); m = m.cuda(non_blocking=True); w = w.cuda(non_blocking=True).float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(x, full=True)
        loss, reg = t3.loss_fn(o, y, m, w); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step(); sched.step()
        if step % 100 == 0: print(f"step {step} loss {loss.item():.4f} reg {reg:.4f} lr {sched.get_last_lr()[0]:.2e} | {(k + 1) * batch / (time.time() - t0):.1f} crops/s", flush=True)
        if step % 500 == 499 or step == steps - 1:
            me, a90, mc, ac, mo, ao = validate(model, vclean, vobs); model.train()
            print(f"  val step {step}: clean all median {me:.1f} deg acc<90 {a90 * 100:.1f}% | clean core median {mc:.1f} deg acc<90 {ac * 100:.1f}% | obscured-voxel core median {mo:.1f} deg acc<90 {ao * 100:.1f}%", flush=True)
            if step % 1000 == 999 or step == steps - 1:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step + 1}, last + ".tmp"); os.replace(last + ".tmp", last)
                torch.save(model.state_dict(), os.path.join(out, f"model_{step + 1}.pt"))
            score = 0.5 * ac + 0.5 * (ao if ao == ao else 0)
            if score > best: best = score; torch.save(model.state_dict(), os.path.join(out, "model_best.pt"))
    torch.save(model.state_dict(), os.path.join(out, "model_final.pt")); print("done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 4, int(sys.argv[3]) if len(sys.argv) > 3 else 6, int(sys.argv[4]) if len(sys.argv) > 4 else 30000)
