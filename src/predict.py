"""Sliding-window prediction of psi (radians, float16) and its confidence R = |circular-mean vector| (float16) for a V3Net / UNet checkpoint over a z range of
the W2 box (default) or the 0139 box (BOX=p0139).  Same tiling as eval_wfield_8c.predict_chunk (96^3 tiles, 16-voxel overlap, centres kept).
  python predict.py MODEL.pt Z0 Z1 OUT_PREFIX          (absolute z of the box's voxel grid; writes OUT_PREFIX_psi.npy and OUT_PREFIX_conf.npy)"""
import os, sys
import numpy as np
import torch
import train_wfield_8c as tw
import eval_wfield_8c as ev

TILE, OV = 96, 16


def predict_chunk2(model, za0, za1, ct, sp, um, ORG):
    uz, uy, ux = um; S = ct.shape; psi_o = np.zeros((za1 - za0, S[1], S[2]), np.float16); r_o = np.zeros((za1 - za0, S[1], S[2]), np.float16); st = TILE - 2 * OV
    zstarts = sorted({int(np.clip(z, 0, S[0] - TILE)) for z in range(za0 - OV, za1 - OV, st)} | {int(np.clip(za0 - OV, 0, S[0] - TILE))})
    for z in zstarts:
        for y in range(0, S[1] - 2 * OV, st):
            ya = min(y, S[1] - TILE); batch, pos = [], []
            for x in range(0, S[2] - 2 * OV, st):
                xa = min(x, S[2] - TILE)
                c_ = np.asarray(ct[z:z + TILE, ya:ya + TILE, xa:xa + TILE]) / 255.0
                if (c_ > 0).mean() < 0.05: continue
                s_ = np.asarray(sp[z:z + TILE, ya:ya + TILE, xa:xa + TILE]) / 255.0
                zz = ORG[0] + z + TILE / 2; cy, cx = np.interp(zz, uz, uy), np.interp(zz, uz, ux)
                gy = ORG[1] + ya + np.arange(TILE)[:, None] - cy; gx = ORG[2] + xa + np.arange(TILE)[None, :] - cx; r = np.hypot(gy, gx) + 1e-6
                batch.append(np.stack([c_, s_, np.broadcast_to((gy / r)[None], c_.shape), np.broadcast_to((gx / r)[None], c_.shape)])); pos.append(xa)
            for i in range(0, len(batch), 4):
                X = torch.from_numpy(np.stack(batch[i:i + 4]).astype(np.float32)).cuda()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    p = model(X).float().cpu().numpy()
                for k, xa in enumerate(pos[i:i + 4]):
                    psi = np.arctan2(p[k, 0], p[k, 1]); R = np.hypot(p[k, 0], p[k, 1])
                    lo_z = z + (OV if z > 0 else 0); hi_z = z + TILE - (OV if z + TILE < S[0] else 0)
                    a0, a1 = max(lo_z, za0), min(hi_z, za1)
                    if a1 <= a0: continue
                    yl, yh = (OV if ya > 0 else 0), TILE - (OV if ya + TILE < S[1] else 0); xl, xh = (OV if xa > 0 else 0), TILE - (OV if xa + TILE < S[2] else 0)
                    sl = (slice(a0 - za0, a1 - za0), slice(ya + yl, ya + yh), slice(xa + xl, xa + xh))
                    psi_o[sl] = psi[a0 - z:a1 - z, yl:yh, xl:xh]; r_o[sl] = R[a0 - z:a1 - z, yl:yh, xl:xh]
    return psi_o, r_o


if __name__ == "__main__":
    mp, z0, z1, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    if os.environ.get("BOX") == "big":                                           # the big box centred on the umbilicus (dl_big.py), no truth
        HOT = os.environ.get("BOX_HOT", "D:/vesuvius_big_hot/"); ORG = tuple(int(v) for v in os.environ.get("BOX_ORG", "10496,1920,3328").split(","))
        from catarr import Cat                                                 # BOX_HOT may list several z-stacked dirs separated by ";" (lowest z first)
        ct = Cat([np.load(h + "ct_big.npy", mmap_mode="r") for h in HOT.split(";")]); sp = Cat([np.load(h + "sp_big.npy", mmap_mode="r") for h in HOT.split(";")]); lab = None; um = tw.umbilicus(); tag = "big"
    else:
        ORG, ct, sp, lab, um, tag = ev.box()
    S = ct.shape; zs, ze = z0 - ORG[0], z1 - ORG[0]
    if any(t in mp for t in ("v3", "v4", "v5")):
        import train_v3 as t3; model = t3.V3Net()
    else:
        model = tw.UNet()
    model.load_state_dict(torch.load(mp, map_location="cpu")); model = model.cuda().eval()
    DN = int(os.environ.get("DOWN", "1"))                                          # DOWN=2: store the 2x vector-averaged psi and the mean confidence (8x smaller)
    P = np.lib.format.open_memmap(out + "_psi.npy", mode="w+", dtype=np.float16, shape=((ze - zs) // DN, S[1] // DN, S[2] // DN))
    Rm = np.lib.format.open_memmap(out + "_conf.npy", mode="w+", dtype=np.float16, shape=((ze - zs) // DN, S[1] // DN, S[2] // DN))
    tot = acc = 0
    for a in range(zs, ze, 64):
        b = min(a + 64, ze); p, r = predict_chunk2(model, a, b, ct, sp, um, ORG)
        if DN == 1: P[a - zs:b - zs] = p; Rm[a - zs:b - zs] = r
        else:
            for k in range(0, p.shape[0], 16):                                    # down-sample in z sub-chunks to bound the memory
                q = p[k:k + 16].astype(np.float32); sh = (q.shape[0] // DN, DN, S[1] // DN, DN, S[2] // DN, DN)
                sn = np.sin(q).reshape(sh).mean((1, 3, 5)); cs = np.cos(q).reshape(sh).mean((1, 3, 5)); o0 = (a - zs + k) // DN
                P[o0:o0 + sn.shape[0]] = np.arctan2(sn, cs).astype(np.float16); Rm[o0:o0 + sn.shape[0]] = r[k:k + 16].astype(np.float32).reshape(sh).mean((1, 3, 5)).astype(np.float16)
        if lab is None: print(f"  chunk z {ORG[0] + a}-{ORG[0] + b} done", flush=True); continue
        L = np.asarray(lab[a:b]); m = L < 255; e = np.abs(np.angle(np.exp(1j * (p[m].astype(np.float32) - L[m] * (2 * np.pi / 255))))); tot += m.sum(); acc += (e < np.pi / 2).sum()
        print(f"  chunk z {ORG[0] + a}-{ORG[0] + b}: sheet-level acc {(e < np.pi / 2).mean() * 100 if len(e) else float('nan'):.1f}%", flush=True)
    P.flush(); Rm.flush(); print(f"{tag} z {z0}-{z1}: " + (f"sheet-level accuracy {acc / max(tot, 1) * 100:.1f}% on {tot} truth voxels" if lab is not None else "done"), flush=True)
