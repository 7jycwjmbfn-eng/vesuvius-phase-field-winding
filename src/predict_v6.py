"""Sliding-window prediction of psi for the 8-channel v6 models (ct, surface prediction, radial unit vector, lasagna nx, ny, grad_mag, cos) over a z range of the old W2 box.
Same tiling as predict.py (96^3 tiles, 16-voxel overlap, centres kept); tile origins are multiples of 4 so that the lasagna up-sampling matches training.
  python predict_v6.py MODEL.pt ARCH Z0 Z1 OUT_PREFIX      (writes OUT_PREFIX_psi.npy and OUT_PREFIX_conf.npy, float16, full L2 resolution; env READOUT=reg|cls|avg)"""
import os, sys
import numpy as np
import torch
import torch.nn.functional as F
import train_wfield_8c as tw
import train_v6 as t6
import archs as A

TILE, OV = 96, 16
FLIPS = {1: [(0, 0, 0)], 4: [(0, 0, 0), (0, 1, 0), (0, 0, 1), (0, 1, 1)], 8: [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]}[int(os.environ.get("TTA", "1"))]
LASF = (("nx", 4, lambda a: (a - 128.0) / 127.0), ("ny", 4, lambda a: (a - 128.0) / 127.0), ("gm", 4, lambda a: a / 255.0), ("cos", 2, lambda a: a / 255.0))


def las_tile(las, z, y, x):
    out = []
    for k, f, dec in LASF:
        cc = TILE // f; a = np.asarray(las[k][z // f:z // f + cc, y // f:y // f + cc, x // f:x // f + cc]).astype(np.float32)
        if a.shape != (cc, cc, cc): a = np.pad(a, [(0, cc - a.shape[0]), (0, cc - a.shape[1]), (0, cc - a.shape[2])], mode="edge")
        out.append(F.interpolate(torch.from_numpy(dec(a))[None, None], scale_factor=f, mode="trilinear", align_corners=False)[0, 0].numpy())
    return np.stack(out)


def predict_chunk(model, mode, za0, za1, ct, sp, las, um, ORG):
    uz, uy, ux = um; S = ct.shape; psi_o = np.zeros((za1 - za0, S[1], S[2]), np.float16); r_o = np.zeros_like(psi_o); st = TILE - 2 * OV
    zstarts = sorted({int(np.clip(z, 0, S[0] - TILE)) // 4 * 4 for z in range(za0 - OV, za1 - OV, st)} | {int(np.clip(za0 - OV, 0, S[0] - TILE)) // 4 * 4})
    for z in zstarts:
        for y in range(0, S[1] - 2 * OV, st):
            ya = min(y, S[1] - TILE) // 4 * 4; batch, pos = [], []
            for x in range(0, S[2] - 2 * OV, st):
                xa = min(x, S[2] - TILE) // 4 * 4
                c_ = np.asarray(ct[z:z + TILE, ya:ya + TILE, xa:xa + TILE]) / 255.0
                if (c_ > 0).mean() < 0.05: continue
                s_ = np.asarray(sp[z:z + TILE, ya:ya + TILE, xa:xa + TILE]) / 255.0
                zz = ORG[0] + z + TILE / 2; cy, cx = np.interp(zz, uz, uy), np.interp(zz, uz, ux)
                gy = ORG[1] + ya + np.arange(TILE)[:, None] - cy; gx = ORG[2] + xa + np.arange(TILE)[None, :] - cx; r = np.hypot(gy, gx) + 1e-6
                base = np.stack([c_, s_, np.broadcast_to((gy / r)[None], c_.shape), np.broadcast_to((gx / r)[None], c_.shape)]).astype(np.float32)
                batch.append(np.concatenate([base, las_tile(las, z, ya, xa)])); pos.append(xa)
            for i in range(0, len(batch), 4):
                X = torch.from_numpy(np.stack(batch[i:i + 4]).astype(np.float32)).cuda()
                p = 0
                for fz, fy, fx in FLIPS:                                                  # flip the input (radial and lasagna components change sign), average the output vectors
                    Xf = X.clone(); dims = [d for d, f in ((2, fz), (3, fy), (4, fx)) if f]
                    if fy: Xf[:, [2, 5]] *= -1
                    if fx: Xf[:, [3, 4]] *= -1
                    if dims: Xf = torch.flip(Xf, dims)
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        o = model(Xf, full=True)
                    q = A.Net.readout(o.float(), mode)
                    p = p + (torch.flip(q, dims) if dims else q)
                p = (p / len(FLIPS)).cpu().numpy()
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
    mp, arch, z0, z1, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]; mode = os.environ.get("READOUT", "reg"); DN = int(os.environ.get("DOWN", "1"))
    if os.environ.get("BOX") == "big":                                           # a big box centred on the umbilicus (no truth): env BOX_HOT (ct_big.npy, sp_big.npy), BOX_ORG, LAS_DIR
        HOT = os.environ.get("BOX_HOT", "D:/vesuvius_big_hot/"); ORG = tuple(int(v) for v in os.environ.get("BOX_ORG", "10496,1920,3328").split(","))
        from catarr import Cat; ct = Cat([np.load(h + "ct_big.npy", mmap_mode="r") for h in HOT.split(";")]); sp = Cat([np.load(h + "sp_big.npy", mmap_mode="r") for h in HOT.split(";")]); lab = None; LD = os.environ.get("LAS_DIR", "")
    else:
        ORG = tw.ORG; ct = np.load(tw.D + "ct_w2.npy", mmap_mode="r"); sp = np.load(tw.D + "sp_w2.npy", mmap_mode="r"); lab = np.load(tw.LAB, mmap_mode="r"); LD = t6.LAS
    um = tw.umbilicus()
    if os.environ.get("CENTRE_JSON"):                                             # another scroll: centre line {"z": [...], "y": [...], "x": [...]} in the voxel units of the box
        import json; _c = json.load(open(os.environ["CENTRE_JSON"])); um = tuple(np.array(_c[k], float) for k in ("z", "y", "x"))
    if os.environ.get("LAS_ZERO"):                                                # no lasagna for this scan: the lasagna group is fed as zeros (nx = ny = 128 encodes 0), as in the training augmentation that drops the group
        _S = ct.shape; las = {"nx": np.full((-(-_S[0] // 4), -(-_S[1] // 4), -(-_S[2] // 4)), 128, np.uint8), "ny": np.full((-(-_S[0] // 4), -(-_S[1] // 4), -(-_S[2] // 4)), 128, np.uint8),
                              "gm": np.zeros((-(-_S[0] // 4), -(-_S[1] // 4), -(-_S[2] // 4)), np.uint8), "cos": np.zeros((-(-_S[0] // 2), -(-_S[1] // 2), -(-_S[2] // 2)), np.uint8)}
    else: las = {k: np.load(LD + f"{k}.npy", mmap_mode="r") for k in ("nx", "ny", "gm", "cos")}
    model = A.Net(arch).cuda().eval(); model.load_state_dict(torch.load(mp, map_location="cpu")); S = ct.shape; zs, ze = z0 - ORG[0], z1 - ORG[0]
    shp = ((ze - zs) // DN, S[1] // DN, S[2] // DN)
    P = np.lib.format.open_memmap(out + "_psi.npy", mode="w+", dtype=np.float16, shape=shp); Rm = np.lib.format.open_memmap(out + "_conf.npy", mode="w+", dtype=np.float16, shape=shp)
    for a in range(zs, ze, 64):
        b = min(a + 64, ze); p, r = predict_chunk(model, mode, a, b, ct, sp, las, um, ORG)
        if DN == 1: P[a - zs:b - zs] = p; Rm[a - zs:b - zs] = r
        else:
            for k in range(0, p.shape[0], 16):                                    # down-sample in z sub-chunks to bound the memory
                q = p[k:k + 16].astype(np.float32); sh = (q.shape[0] // DN, DN, S[1] // DN, DN, S[2] // DN, DN)
                sn = np.sin(q).reshape(sh).mean((1, 3, 5)); cs = np.cos(q).reshape(sh).mean((1, 3, 5)); o0 = (a - zs + k) // DN
                P[o0:o0 + sn.shape[0]] = np.arctan2(sn, cs).astype(np.float16); Rm[o0:o0 + sn.shape[0]] = r[k:k + 16].astype(np.float32).reshape(sh).mean((1, 3, 5)).astype(np.float16)
        if lab is None: print(f"  chunk z {ORG[0] + a}-{ORG[0] + b} done", flush=True); continue
        L = np.asarray(lab[a:b]); m = L < 255; e = np.abs(np.angle(np.exp(1j * (p[m].astype(np.float32) - L[m] * (2 * np.pi / 255))))) if m.any() else np.array([])
        print(f"  chunk z {ORG[0] + a}-{ORG[0] + b}: sheet-level acc {(e < np.pi / 2).mean() * 100 if len(e) else float('nan'):.1f}% on {m.sum()} truth voxels", flush=True)
    P.flush(); Rm.flush(); print("done", flush=True)
