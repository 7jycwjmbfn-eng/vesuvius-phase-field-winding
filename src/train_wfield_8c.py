"""wfield (PREREG_wfield_8c_1005.md): 3D U-Net that predicts the sheet phase psi (sin, cos) from L2 CT + surface prediction +
the outward radial direction. bf16 autocast, multi-process DataLoader reading random 96^3 crops from memmapped npy.
  python train_wfield_8c.py bench [BATCH] [WORKERS]        200-step throughput test (placeholder labels)
  python train_wfield_8c.py train OUTDIR [BATCH] [WORKERS] [STEPS]
Labels: D:/vesuvius_wfield_8c/psi_w2.npy (uint8, 0-254 = psi * 255 / 2pi, 255 = none). Missing -> placeholder from surface
prediction (pipeline / throughput only, never reported as accuracy)."""
import json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LAB = "G:/vesuvius_77f5/psi_w2_8cbox_L2_77f5.npy"
D = "G:/vesuvius_wfield_8c/"; ORG = (9984, 2304, 3584); CROP = 96
TRAIN_Z = (10000, 10380); VAL_Z = (10404, 10500)


def umbilicus():
    cp = sorted(json.load(open(r"D:\vesuvius_downstream\p4\ds\umbilicus.json"))["control_points"], key=lambda c: c["z"])
    return tuple(np.array([c[t] for c in cp], float) for t in ("z", "y", "x"))


class Crops(torch.utils.data.Dataset):
    def __init__(self, zr, n, seed=0, aug=True):
        self.zr, self.n, self.seed, self.aug = zr, n, seed, aug; self.ct = None

    def __len__(self):
        return self.n

    def _open(self):
        self.ct = np.load(D + "ct_w2.npy", mmap_mode="r"); self.sp = np.load(D + "sp_w2.npy", mmap_mode="r")
        self.lab = np.load(LAB, mmap_mode="r") if os.path.exists(LAB) and not os.environ.get("PLACEHOLDER") else None
        self.uz, self.uy, self.ux = umbilicus()

    def __getitem__(self, i):
        if self.ct is None: self._open()
        rng = np.random.default_rng((self.seed, i, int(time.time() * 1e6) % 2**31) if self.aug else (self.seed, i))
        S = self.ct.shape
        for _ in range(60):
            z = rng.integers(self.zr[0] - ORG[0], self.zr[1] - ORG[0] - CROP + 1); y = rng.integers(0, S[1] - CROP); x = rng.integers(0, S[2] - CROP)
            if self.lab is not None and (np.asarray(self.lab[z + CROP // 2, y:y + CROP:4, x:x + CROP:4]) < 255).mean() < 0.2: continue
            ct = np.asarray(self.ct[z:z + CROP, y:y + CROP, x:x + CROP])
            if (ct > 0).mean() > 0.5: break
        sp = np.asarray(self.sp[z:z + CROP, y:y + CROP, x:x + CROP])
        zz = ORG[0] + z + CROP / 2
        cy, cx = np.interp(zz, self.uz, self.uy), np.interp(zz, self.uz, self.ux)
        gy = ORG[1] + y + np.arange(CROP)[:, None] - cy; gx = ORG[2] + x + np.arange(CROP)[None, :] - cx; r = np.hypot(gy, gx) + 1e-6
        ry = np.broadcast_to((gy / r)[None], (CROP, CROP, CROP)); rx = np.broadcast_to((gx / r)[None], (CROP, CROP, CROP))
        if self.lab is not None:
            L = np.asarray(self.lab[z:z + CROP, y:y + CROP, x:x + CROP]); m = L < 255; psi = L.astype(np.float32) * (2 * np.pi / 255)
        else:                                                                        # placeholder: phase-like target from the surface prediction
            m = np.ones_like(sp, bool); psi = sp.astype(np.float32) / 255 * np.pi
        X = np.stack([ct / 255.0, sp / 255.0, ry, rx]).astype(np.float16)             # float16 in shared memory (Windows commit budget)
        Y = np.stack([np.sin(psi), np.cos(psi)]).astype(np.float16)
        if self.aug:                                                                 # flips: the radial-direction channels flip sign with the axis
            for ax, ch in ((1, None), (2, 2), (3, 3)):
                if rng.random() < 0.5:
                    X = np.flip(X, ax); Y = np.flip(Y, ax); m = np.flip(m, ax - 1)
                    if ch is not None: X[ch] = -X[ch]
        return torch.from_numpy(np.ascontiguousarray(X)), torch.from_numpy(np.ascontiguousarray(Y)), torch.from_numpy(np.ascontiguousarray(m))


def block(ci, co):
    return nn.Sequential(nn.Conv3d(ci, co, 3, padding=1, bias=False), nn.GroupNorm(8, co), nn.SiLU(inplace=True),
                         nn.Conv3d(co, co, 3, padding=1, bias=False), nn.GroupNorm(8, co), nn.SiLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, cin=4, cout=2, ch=(24, 48, 96, 192)):
        super().__init__()
        self.enc = nn.ModuleList([block(cin if i == 0 else ch[i - 1], c) for i, c in enumerate(ch)])
        self.down = nn.ModuleList([nn.Conv3d(c, c, 2, stride=2) for c in ch[:-1]])
        self.up = nn.ModuleList([nn.ConvTranspose3d(ch[i + 1], ch[i], 2, stride=2) for i in range(len(ch) - 1)])
        self.dec = nn.ModuleList([block(2 * ch[i], ch[i]) for i in range(len(ch) - 1)])
        self.head = nn.Conv3d(ch[0], cout, 1)

    def forward(self, x):
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x)
            if i < len(self.down): skips.append(x); x = self.down[i](x)
        for i in reversed(range(len(self.up))):
            x = self.dec[i](torch.cat([self.up[i](x), skips[i]], 1))
        return self.head(x)


def loss_fn(p, y, m):
    m = m.unsqueeze(1).float(); return ((p.float() - y) ** 2 * m).sum() / (2 * m.sum() + 1)


def validate(model, vset):
    model.eval(); L = []; E = []
    with torch.no_grad():
        for i in range(0, len(vset), 4):
            x = torch.stack([v[0] for v in vset[i:i + 4]]).cuda().float(); y = torch.stack([v[1] for v in vset[i:i + 4]]).cuda().float(); m = torch.stack([v[2] for v in vset[i:i + 4]]).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = model(x)
            p = p.float(); L.append(loss_fn(p, y, m).item())
            d = torch.atan2(p[:, 0], p[:, 1]) - torch.atan2(y[:, 0], y[:, 1]); d = torch.atan2(torch.sin(d), torch.cos(d)).abs()
            E.append(d[m].float().cpu())
    e = torch.cat(E); return float(np.mean(L)), float(np.degrees(e.median().item())), float((e < np.pi / 2).float().mean().item())


def run(mode, out=None, batch=8, workers=8, steps=200):
    torch.backends.cudnn.benchmark = True; dev = "cuda"
    model = UNet().to(dev); model = model if os.environ.get("NOCL") else model.to(memory_format=torch.channels_last_3d)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.05)
    dl = torch.utils.data.DataLoader(Crops(TRAIN_Z, steps * batch), batch_size=batch, num_workers=workers, pin_memory=True,
                                     persistent_workers=workers > 0, prefetch_factor=2 if workers else None)
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M, batch {batch}, workers {workers}, labels {'77f5 psi' if os.path.exists(LAB) and not os.environ.get('PLACEHOLDER') else 'PLACEHOLDER'}", flush=True)
    vset = [Crops(VAL_Z, 64, seed=123, aug=False)[i] for i in range(64)] if mode == "train" else None; best = 1e9
    t0 = time.time(); tl = None; n = 0; wait = 0.0; tw = time.time()
    for step, (x, y, m) in enumerate(dl):
        wait += time.time() - tw
        x = x.to(dev, non_blocking=True).float(); x = x if os.environ.get("NOCL") else x.to(memory_format=torch.channels_last_3d); y = y.to(dev, non_blocking=True).float(); m = m.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = model(x)
        loss = loss_fn(p, y, m); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if step == 20: torch.cuda.synchronize(); tl = time.time(); n = 0; wait = 0.0
        n += x.shape[0]
        if step % int(os.environ.get('LOGEVERY', 50)) == 0 or step == steps - 1:
            torch.cuda.synchronize(); el = time.time() - (tl or t0)
            print(f"step {step} loss {loss.item():.4f} | {n / max(el, 1e-6):.1f} crops/s, data wait {wait / max(el, 1e-6) * 100:.0f}% of time, "
                  f"max mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
        if mode == "train" and out and (step % 500 == 499 or step == steps - 1):
            vl, ve, va = validate(model, vset); model.train()
            print(f"  val step {step}: loss {vl:.4f}, circular median error {ve:.1f} deg, sheet-level acc (<90 deg) {va * 100:.1f}%", flush=True)
            os.makedirs(out, exist_ok=True); torch.save(model.state_dict(), os.path.join(out, f"model_{step + 1}.pt"))
            if vl < best: best = vl; torch.save(model.state_dict(), os.path.join(out, "model_best.pt"))
        tw = time.time()
    return model


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "bench":
        run("bench", batch=int(sys.argv[2]) if len(sys.argv) > 2 else 8, workers=int(sys.argv[3]) if len(sys.argv) > 3 else 8, steps=int(sys.argv[4]) if len(sys.argv) > 4 else 200)
    else:
        run("train", out=sys.argv[2], batch=int(sys.argv[3]), workers=int(sys.argv[4]), steps=int(sys.argv[5]))
