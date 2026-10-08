"""wfield v3 (PREREG_wfield_v3_1006.md): same task and data as train_wfield_8c.py, four changes backed by the run2 error analysis.
  1. loss weight by local truth validity (truth near the edge of the valid region is unreliable: 25.8% of voxels within 4 voxels of an
     invalid voxel are >90 deg off, 1.2% when >= 30 voxels away)
  2. the full OneCycle schedule is run to the end and checkpoints are resumable (run2 was cut at 11500 of 30000 steps at a high learning rate)
  3. extra head: 24-bin phase classification (soft ordinal labels); the output used downstream is the circular mean of the softmax
  4. augmentation: intensity / noise on CT, scale jitter and random dropout of the surface-prediction channel (cross-scroll robustness)
  python train_v3.py OUTDIR [BATCH] [WORKERS] [STEPS]
Reuses Crops / UNet blocks / paths from train_wfield_8c.py (not modified)."""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage as ndi
import train_wfield_8c as tw

NB = 24                                                                              # phase bins
CENT = torch.arange(NB, dtype=torch.float32) * (2 * np.pi / NB) + np.pi / NB


class CropsV3(tw.Crops):
    def __getitem__(self, i):
        X, Y, m = super().__getitem__(i)
        mn = m.numpy(); frac = ndi.uniform_filter(mn.astype(np.float32), 9, mode="nearest")
        w = np.where(mn, np.clip(frac, 0.2, 1.0) ** 2, 0.0).astype(np.float16)
        if self.aug:
            rng = np.random.default_rng((self.seed, i, int(time.time() * 1e6) % 2**31))
            X = X.clone(); ct = X[0].float() * rng.uniform(0.85, 1.15) + rng.uniform(-0.05, 0.05) + torch.from_numpy(rng.normal(0, 0.02, X[0].shape).astype(np.float32))
            X[0] = ct.clamp(0, 1).half()
            if rng.random() < 0.15: X[1] = 0
            else: X[1] = (X[1].float() * rng.uniform(0.8, 1.2)).clamp(0, 1).half()
        return X, Y, m, torch.from_numpy(w)


class V3Net(tw.UNet):
    def __init__(self, cin=4, ch=(24, 48, 96, 192)):
        super().__init__(cin=cin, cout=2 + NB, ch=ch)

    def forward(self, x, full=False):
        o = super().forward(x)
        if full: return o
        p = torch.softmax(o[:, 2:].float(), 1); c = CENT.to(o.device).view(1, NB, 1, 1, 1)
        return torch.stack([(p * torch.sin(c)).sum(1), (p * torch.cos(c)).sum(1)], 1)       # circular-mean vector (sin, cos), same interface as UNet


def loss_fn(o, y, m, w, sigma=1.0):
    o = o.float(); wm = (w * m).float().unsqueeze(1)
    reg = (((o[:, :2] - y) ** 2) * wm).sum() / (2 * wm.sum() + 1)
    psi = torch.atan2(y[:, 0], y[:, 1]) % (2 * np.pi)                                   # [B, D, H, W]
    d = torch.remainder(psi.unsqueeze(1) - CENT.to(o.device).view(1, NB, 1, 1, 1) + np.pi, 2 * np.pi) - np.pi
    soft = torch.softmax(-(d / (2 * np.pi / NB)) ** 2 / (2 * sigma ** 2), 1)
    ce = -(soft * F.log_softmax(o[:, 2:], 1)).sum(1, keepdim=True)
    return reg + (ce * wm).sum() / (wm.sum() + 1), reg.item()


def validate(model, vset):
    model.eval(); E = []; Wt = []
    with torch.no_grad():
        for i in range(0, len(vset), 4):
            b = vset[i:i + 4]; x = torch.stack([v[0] for v in b]).cuda().float(); y = torch.stack([v[1] for v in b]).cuda().float()
            m = torch.stack([v[2] for v in b]).cuda(); w = torch.stack([v[3] for v in b]).cuda().float()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = model(x)
            p = p.float(); d = torch.atan2(p[:, 0], p[:, 1]) - torch.atan2(y[:, 0], y[:, 1]); d = torch.atan2(torch.sin(d), torch.cos(d)).abs()
            E.append(d[m].cpu()); Wt.append(w[m].cpu())
    e = torch.cat(E); w = torch.cat(Wt); core = w >= 0.9
    return (float(np.degrees(e.median())), float((e < np.pi / 2).float().mean()), float(np.degrees(e[core].median())), float((e[core] < np.pi / 2).float().mean()))


def main(out, batch=4, workers=6, steps=30000):
    torch.backends.cudnn.benchmark = True; os.makedirs(out, exist_ok=True)
    model = V3Net().cuda(); opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1.5e-3, total_steps=steps, pct_start=0.05)
    start = 0; last = os.path.join(out, "last.pt")
    if os.path.exists(last):
        ck = torch.load(last, map_location="cuda"); model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); start = ck["step"]
        print(f"resumed from step {start}", flush=True)
    dl = torch.utils.data.DataLoader(CropsV3(tw.TRAIN_Z, (steps - start) * batch, seed=start + 1), batch_size=batch, num_workers=workers, pin_memory=True,
                                     persistent_workers=True, prefetch_factor=2)
    vset = [CropsV3(tw.VAL_Z, 64, seed=123, aug=False)[i] for i in range(64)]; best = -1
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M, batch {batch}, workers {workers}, steps {steps}, start {start}", flush=True)
    t0 = time.time(); model.train(); acc = 0.0
    for k, (x, y, m, w) in enumerate(dl):
        step = start + k
        x = x.cuda(non_blocking=True).float(); y = y.cuda(non_blocking=True).float(); m = m.cuda(non_blocking=True); w = w.cuda(non_blocking=True).float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(x, full=True)
        loss, reg = loss_fn(o, y, m, w); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step(); sched.step()
        if step % 100 == 0: print(f"step {step} loss {loss.item():.4f} reg {reg:.4f} lr {sched.get_last_lr()[0]:.2e} | {(k + 1) * batch / (time.time() - t0):.1f} crops/s", flush=True)
        if step % 500 == 499 or step == steps - 1:
            me, a90, mc, ac = validate(model, vset); model.train()
            print(f"  val step {step}: all voxels median {me:.1f} deg, acc<90 {a90 * 100:.1f}% | core (weight>=0.9) median {mc:.1f} deg, acc<90 {ac * 100:.1f}%", flush=True)
            if step % 1000 == 999 or step == steps - 1:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step + 1}, last + ".tmp"); os.replace(last + ".tmp", last)
                torch.save(model.state_dict(), os.path.join(out, f"model_{step + 1}.pt"))
            if ac > best: best = ac; torch.save(model.state_dict(), os.path.join(out, "model_best.pt"))
    torch.save(model.state_dict(), os.path.join(out, "model_final.pt")); print("done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 4, int(sys.argv[3]) if len(sys.argv) > 3 else 6, int(sys.argv[4]) if len(sys.argv) > 4 else 30000)
