"""wfield readouts (PREREG_wfield_8c_1005.md), memory-light: prediction is written chunk by chunk to a float16 memmap on G:,
statistics accumulate per chunk; R1 lines read only a small crop around each line.
  (a) sheet-level accuracy |dpsi| < 90 deg on truth voxels; R3 circular median error
  (b) around Will switch points (willtruth=1, +-5 L2 voxels) vs control (willtruth=0)            [W2 box only]
  R1  sheet count along the truth phase gradient, 12-60 voxels: prediction vs truth vs surface-prediction ridges
  python eval_wfield_8c.py MODEL.pt Z0 Z1 [NLINES]          (absolute z of the box's volume; BOX=p0139 for the 0139 box)"""
import json, os, sys
import numpy as np
import torch
from scipy import ndimage as ndi
import train_wfield_8c as tw

TILE, OV = 96, 16


def box():
    if os.environ.get("BOX") == "p0139":
        G = "G:/vesuvius_77f5/p0139/"; cen = json.load(open(G + "centre_w023.json"))
        return ((4352, 1664, 1280), np.load(G + "box_ct.npy", mmap_mode="r"), np.load(G + "box_surf.npy", mmap_mode="r"),
                np.load(G + "psi_box.npy", mmap_mode="r"), tuple(np.array(cen[k], float) for k in ("z", "y", "x")), "p0139")
    return (tw.ORG, np.load(tw.D + "ct_w2.npy", mmap_mode="r"), np.load(tw.D + "sp_w2.npy", mmap_mode="r"),
            np.load(tw.LAB, mmap_mode="r"), tw.umbilicus(), "w2")


def predict_chunk(model, za0, za1, ct, sp, um, ORG):
    """psi for box-relative z rows [za0, za1), float32; tiles overlap by OV and only their centres are kept"""
    uz, uy, ux = um; S = ct.shape; out = np.zeros((za1 - za0, S[1], S[2]), np.float32); st = TILE - 2 * OV
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
                    psi = np.arctan2(p[k, 0], p[k, 1])
                    lo_z = z + (OV if z > 0 else 0); hi_z = z + TILE - (OV if z + TILE < S[0] else 0)
                    a0, a1 = max(lo_z, za0), min(hi_z, za1)
                    if a1 <= a0: continue
                    yl, yh = (OV if ya > 0 else 0), TILE - (OV if ya + TILE < S[1] else 0); xl, xh = (OV if xa > 0 else 0), TILE - (OV if xa + TILE < S[2] else 0)
                    out[a0 - za0:a1 - za0, ya + yl:ya + yh, xa + xl:xa + xh] = psi[a0 - z:a1 - z, yl:yh, xl:xh]
    return out


def count_line(vals, kind):
    if kind == "phase":
        return int(np.rint(np.angle(np.exp(1j * np.diff(vals))).sum() / (2 * np.pi)))
    v = vals; pk = (v[1:-1] >= 0.5) & (v[1:-1] >= v[:-2]) & (v[1:-1] >= v[2:]); n = 0; last = -99
    for i in np.nonzero(pk)[0]:
        if i - last >= 8: n += 1; last = i
    return n


if __name__ == "__main__":
    mp, z0, z1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); NL = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
    ORG, ct, sp, lab, um, tag = box()
    model = tw.UNet().cuda(); model.load_state_dict(torch.load(mp)); model.eval()
    zs, ze = z0 - ORG[0], z1 - ORG[0]; S = ct.shape
    pf = mp.replace(".pt", f"_pred_{tag}_{z0}_{z1}.npy")
    pred = np.lib.format.open_memmap(pf, mode="w+", dtype=np.float16, shape=(ze - zs, S[1], S[2]))
    tot = acc = 0; errs = []
    for a in range(zs, ze, 64):
        b = min(a + 64, ze); p = predict_chunk(model, a, b, ct, sp, um, ORG); pred[a - zs:b - zs] = p
        L = np.asarray(lab[a:b]); m = L < 255; e = np.abs(np.angle(np.exp(1j * (p[m] - L[m] * (2 * np.pi / 255)))))
        tot += m.sum(); acc += (e < np.pi / 2).sum()
        errs.append(np.random.default_rng(a).choice(e, min(len(e), 200000), replace=False) if len(e) else e)
        print(f"  chunk z {ORG[0] + a}-{ORG[0] + b}: truth voxels {m.sum()}, sheet-level acc {(e < np.pi / 2).mean() * 100 if len(e) else float('nan'):.1f}%", flush=True)
    pred.flush()
    print(f"(a) {tag} z {z0}-{z1}: sheet-level accuracy (|dpsi| < 90 deg) {acc / max(tot, 1) * 100:.1f}% on {tot} truth voxels; "
          f"R3 circular median error {np.degrees(np.median(np.concatenate(errs))):.1f} deg (sampled)", flush=True)
    if tag == "w2":
        W = np.load("E:/vesuvius_downstream_tmp/fiberA/p9comp_w2u.npz")["pts"].astype(np.float64) * 2.0 - np.array(ORG) - np.array([zs, 0, 0])
        wt = np.load("E:/vesuvius_downstream_tmp/fiberA/willtruth.npz")["truth"]
        inb = np.all((W >= 6) & (W < np.array(pred.shape) - 7), 1); rng = np.random.default_rng(1)
        for lab_, name in ((1, "Will switch points (willtruth=1)"), (0, "control: Will points with willtruth=0")):
            q = np.rint(W[inb & (wt == lab_)]).astype(int)
            if lab_ == 0 and len(q) > 20000: q = q[rng.choice(len(q), 20000, replace=False)]
            ok = ac = 0; seen = set()
            for z_, y_, x_ in q:
                key = (z_ // 5, y_ // 5, x_ // 5)
                if key in seen: continue
                seen.add(key)
                L = np.asarray(lab[zs + z_ - 5:zs + z_ + 6, y_ - 5:y_ + 6, x_ - 5:x_ + 6]); m = L < 255
                P = np.asarray(pred[z_ - 5:z_ + 6, y_ - 5:y_ + 6, x_ - 5:x_ + 6]).astype(np.float32)
                e = np.abs(np.angle(np.exp(1j * (P[m] - L[m] * (2 * np.pi / 255))))); ok += m.sum(); ac += (e < np.pi / 2).sum()
            print(f"(b) {name}: {len(q)} points ({len(seen)} distinct 5-voxel cells), {ok} truth voxels within +-5: "
                  f"sheet-level accuracy {ac / max(ok, 1) * 100:.1f}%", flush=True)
    rng = np.random.default_rng(0); res = {"t": [], "p": [], "b": []}; tries = 0
    while len(res["t"]) < NL and tries < NL * 30:
        tries += 1
        p0 = np.array([rng.integers(70, ze - zs - 70), rng.integers(70, S[1] - 70), rng.integers(70, S[2] - 70)]); c0 = p0 - 64
        Lc = np.asarray(lab[zs + c0[0]:zs + c0[0] + 129, c0[1]:c0[1] + 129, c0[2]:c0[2] + 129]); mc = Lc < 255
        if not mc[63:66, 63:66, 63:66].all(): continue
        tc = Lc.astype(np.float32) * (2 * np.pi / 255); zc = np.exp(1j * tc[63:66, 63:66, 63:66])
        g = np.array([np.angle(zc[2, 1, 1] / zc[0, 1, 1]), np.angle(zc[1, 2, 1] / zc[1, 0, 1]), np.angle(zc[1, 1, 2] / zc[1, 1, 0])]) / 2
        if np.linalg.norm(g) < 1e-3: continue
        d = g / np.linalg.norm(g); Lr = rng.uniform(12, 60); t = np.arange(0, Lr + 0.25, 0.5); P = (64 + t[:, None] * d[None]).T
        if not ndi.map_coordinates(mc.astype(np.uint8), P, order=0).all(): continue
        Pc = np.asarray(pred[c0[0]:c0[0] + 129, c0[1]:c0[1] + 129, c0[2]:c0[2] + 129]).astype(np.float32)
        Sc = np.asarray(sp[zs + c0[0]:zs + c0[0] + 129, c0[1]:c0[1] + 129, c0[2]:c0[2] + 129]).astype(np.float32) / 255
        tv = ndi.map_coordinates(tc, P, order=0); pv = ndi.map_coordinates(Pc, P, order=0); sv = ndi.map_coordinates(Sc, P, order=1)
        res["t"].append(count_line(tv, "phase")); res["p"].append(count_line(pv, "phase")); res["b"].append(count_line(sv, "ridge"))
    T, Pp, B = (np.array(res[k]) for k in ("t", "p", "b"))
    print(f"R1 {len(T)} lines: exact sheet count prediction {np.mean(Pp == T) * 100:.1f}%, surface-prediction ridges {np.mean(B == T) * 100:.1f}%")
    for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 99)):
        q = (T >= lo) & (T < hi)
        if q.any(): print(f"   truth n in [{lo},{hi}): {q.sum()} lines, prediction {np.mean(Pp[q] == T[q]) * 100:.1f}%, ridges {np.mean(B[q] == T[q]) * 100:.1f}%")
