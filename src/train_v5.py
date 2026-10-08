"""wfield v5 = continued training of 8c's run2 model (warm start from frozen_8c.pt) with the v3 / v4 additions and a phase-gradient matching loss.
  - warm start: the U-Net backbone and the 2-channel (sin, cos) head come from frozen_8c.pt; the 24-bin classification head is new (random init)
  - fresh AdamW, OneCycle (max lr 8e-4) over STEPS, resumable checkpoints
  - validity-weighted loss (v3), 24-bin classification head (v3), obscuration augmentation (v4)
  - NEW phase-gradient loss: for every voxel pair that is neighbours along z, y, x and has truth at both ends, the wrapped phase difference of the prediction must
    match the wrapped phase difference of the truth.  A vortex line (a topological defect that merges neighbouring sheets in the level sets) is a place where
    the wrapped gradient field has a curl; training on the gradient directly penalises it.
  python train_v5.py OUTDIR [BATCH] [WORKERS] [STEPS] [WARM.pt]"""
import os, sys, time
import numpy as np
import torch
import train_wfield_8c as tw
import train_v3 as t3
import train_v4 as t4

GW = 1.0                                                                                  # weight of the gradient loss
SW = 0.5                                                                                  # weight of the physical smoothness hinge (all voxels, labelled or not)
HINGE = 1.2                                                                               # rad per voxel: a sheet phase cannot change faster than this


def wrapped(a):
    return torch.atan2(torch.sin(a), torch.cos(a))


def grad_loss(o, y, m, w):
    """o: [B, 2 + NB, D, H, W] raw outputs; y: [B, 2, D, H, W] truth (sin, cos); m: validity; w: weights.  Phase from the circular mean of the softmax."""
    p = torch.softmax(o[:, 2:].float(), 1); c = t3.CENT.to(o.device).view(1, t3.NB, 1, 1, 1)
    ps = torch.atan2((p * torch.sin(c)).sum(1), (p * torch.cos(c)).sum(1)); pt = torch.atan2(y[:, 0], y[:, 1]); tot = 0.0; den = 0.0
    wm = (w * m).float()
    for ax in (1, 2, 3):                                                                   # D, H, W axes of [B, D, H, W]
        a = [slice(None)] * 4; b = [slice(None)] * 4; a[ax] = slice(0, -1); b[ax] = slice(1, None); a, b = tuple(a), tuple(b)
        dpred = wrapped(ps[b] - ps[a]); dtrue = wrapped(pt[b] - pt[a]); ww = torch.minimum(wm[a], wm[b])
        tot = tot + (ww * (wrapped(dpred - dtrue) ** 2)).sum(); den = den + ww.sum()
    return tot / (den + 1.0)


def smooth_hinge(o, ct):
    """Physical prior on EVERY voxel with CT: the sheet phase (circular mean of the softmax) cannot change faster than HINGE rad per voxel between neighbours.
    Noise-made jumps are what create vortex defects; this acts also where there is no truth."""
    p = torch.softmax(o[:, 2:].float(), 1); c = t3.CENT.to(o.device).view(1, t3.NB, 1, 1, 1)
    ps = torch.atan2((p * torch.sin(c)).sum(1), (p * torch.cos(c)).sum(1)); m = (ct > 0.02).float(); tot = 0.0; den = 0.0
    for ax in (1, 2, 3):
        a = [slice(None)] * 4; b = [slice(None)] * 4; a[ax] = slice(0, -1); b[ax] = slice(1, None); a, b = tuple(a), tuple(b)
        d = wrapped(ps[b] - ps[a]).abs(); mm = torch.minimum(m[a], m[b]); tot = tot + (mm * torch.relu(d - HINGE) ** 2).sum(); den = den + mm.sum()
    return tot / (den + 1.0)


def main(out, batch=4, workers=6, steps=20000, warm="G:/vesuvius_wfield_8c/run2/frozen_8c.pt"):
    torch.backends.cudnn.benchmark = True; os.makedirs(out, exist_ok=True)
    model = t3.V3Net().cuda(); sd = torch.load(warm, map_location="cpu"); own = model.state_dict(); keep = {}
    for k, v in sd.items():
        if k == "head.weight":                                                             # copy the (sin, cos) rows of the old 2-channel head
            nw = own[k].clone(); nw[:2] = v; keep[k] = nw
        elif k == "head.bias":
            nb = own[k].clone(); nb[:2] = v; keep[k] = nb
        elif k in own and own[k].shape == v.shape: keep[k] = v
    model.load_state_dict(keep, strict=False)
    print(f"warm start from {warm}: {len(keep)} of {len(own)} tensors copied", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=8e-4, total_steps=steps, pct_start=0.08)
    start = 0; last = os.path.join(out, "last.pt")
    if os.path.exists(last):
        ck = torch.load(last, map_location="cuda"); model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); start = ck["step"]
        print(f"resumed from step {start}", flush=True)
    dl = torch.utils.data.DataLoader(t4.CropsV4(tw.TRAIN_Z, (steps - start) * batch, seed=start + 11), batch_size=batch, num_workers=workers, pin_memory=True,
                                     persistent_workers=True, prefetch_factor=2)
    vclean = [t4.CropsV4(tw.VAL_Z, 64, seed=123, aug=False, p_obscure=0)[i] for i in range(64)]; vobs = [t4.CropsV4(tw.VAL_Z, 64, seed=123, aug=False, force=True)[i] for i in range(64)]
    best = -1; t0 = time.time(); model.train()
    for k, (x, y, m, w, bm) in enumerate(dl):
        step = start + k
        x = x.cuda(non_blocking=True).float(); y = y.cuda(non_blocking=True).float(); m = m.cuda(non_blocking=True); w = w.cuda(non_blocking=True).float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(x, full=True)
        loss, reg = t3.loss_fn(o, y, m, w); gl = grad_loss(o, y, m, w); sl = smooth_hinge(o, x[:, 0]); total = loss + GW * gl + SW * sl
        opt.zero_grad(set_to_none=True); total.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step(); sched.step()
        if step % 100 == 0: print(f"step {step} loss {loss.item():.4f} grad {gl.item():.4f} hinge {sl.item():.4f} reg {reg:.4f} lr {sched.get_last_lr()[0]:.2e} | {(k + 1) * batch / (time.time() - t0):.1f} crops/s", flush=True)
        if step % 500 == 499 or step == steps - 1:
            me, a90, mc, ac, mo, ao = t4.validate(model, vclean, vobs); model.train()
            print(f"  val step {step}: clean all median {me:.1f} deg acc<90 {a90 * 100:.1f}% | clean core median {mc:.1f} deg acc<90 {ac * 100:.1f}% | obscured-voxel core median {mo:.1f} deg acc<90 {ao * 100:.1f}%", flush=True)
            if step % 1000 == 999 or step == steps - 1:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step + 1}, last + ".tmp"); os.replace(last + ".tmp", last)
                torch.save(model.state_dict(), os.path.join(out, f"model_{step + 1}.pt"))
            score = 0.5 * ac + 0.5 * (ao if ao == ao else 0)
            if score > best: best = score; torch.save(model.state_dict(), os.path.join(out, "model_best.pt"))
    torch.save(model.state_dict(), os.path.join(out, "model_final.pt")); print("done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 4, int(sys.argv[3]) if len(sys.argv) > 3 else 6, int(sys.argv[4]) if len(sys.argv) > 4 else 20000,
         sys.argv[5] if len(sys.argv) > 5 else "G:/vesuvius_wfield_8c/run2/frozen_8c.pt")
