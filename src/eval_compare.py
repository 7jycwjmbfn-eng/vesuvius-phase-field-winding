"""Same validation crops, same metric, for models with different inputs.  4-channel models (frozen_8c, v3, v4, v5) take channels 0-3 of the 8-channel crop.
  python eval_compare.py          (edit MODELS below)"""
import sys, torch, numpy as np
sys.path.insert(0, ".")
import train_wfield_8c as tw, train_v3 as t3, train_v6 as t6, archs as A

def load_models(extra):
    M = {}
    m = tw.UNet().cuda().eval(); m.load_state_dict(torch.load("G:/vesuvius_wfield_8c/run2/frozen_8c.pt", map_location="cpu")); M["frozen_8c (4ch, regression)"] = (m, 4, "uvec")
    m = t3.V3Net().cuda().eval(); m.load_state_dict(torch.load("G:/vesuvius_wfield_8c/v4/model_final.pt", map_location="cpu")); M["v4 final (4ch, regression head)"] = (m, 4, "reg"); M["v4 final (4ch, classification head)"] = (m, 4, "cls")
    for name, arch, path in extra:
        m = A.Net(arch).cuda().eval(); m.load_state_dict(torch.load(path, map_location="cpu")); M[f"{name} (8ch, regression head)"] = (m, 8, "reg"); M[f"{name} (8ch, classification head)"] = (m, 8, "cls")
    return M

def run(M, vs, only_blob):
    out = {k: ([], []) for k in M}
    with torch.no_grad():
        for i in range(0, len(vs), 2):
            b = vs[i:i + 2]; x = torch.stack([v[0] for v in b]).cuda().float(); y = torch.stack([v[1] for v in b]).cuda().float(); mm = torch.stack([v[2] for v in b]).cuda(); w = torch.stack([v[3] for v in b]).cuda().float(); bm = torch.stack([v[4] for v in b]).cuda()
            sel = mm & (bm if only_blob else torch.ones_like(mm))
            for k, (m, cin, mode) in M.items():
                with torch.autocast("cuda", dtype=torch.bfloat16): o = m(x[:, :cin], full=True) if mode != "uvec" else m(x[:, :cin])
                o = o.float(); p = o if mode == "uvec" else A.Net.readout(o, mode)
                d = torch.atan2(p[:, 0], p[:, 1]) - torch.atan2(y[:, 0], y[:, 1]); d = torch.atan2(torch.sin(d), torch.cos(d)).abs()
                out[k][0].append(d[sel].cpu()); out[k][1].append(w[sel].cpu())
    return out

if __name__ == "__main__":
    extra = [("v6 resenc 8000 steps", "resenc", "G:/vesuvius_wfield_8c/v6_resenc_8kb/ema_final.pt")]
    import os
    M = load_models(extra); vc = [t6.CropsV6(t6.VAL_Z, 48, seed=7, aug=False, p_obscure=0)[i] for i in range(48)]; vo = [t6.CropsV6(t6.VAL_Z, 48, seed=7, aug=False, force=True)[i] for i in range(48)]
    rc = run(M, vc, False); ro = run(M, vo, True)
    print(f"{'model':42s} | clean core acc / median | clean all acc | obscured core acc / median   (same {len(vc)} crops of 112^3, z 10388-10500)")
    for k in M:
        e = torch.cat(rc[k][0]); w = torch.cat(rc[k][1]); core = w >= 0.9; eo = torch.cat(ro[k][0]); wo = torch.cat(ro[k][1]); coreo = wo >= 0.9
        print(f"{k:42s} | {float((e[core] < np.pi / 2).float().mean()) * 100:5.1f}% / {float(np.degrees(e[core].median())):5.1f} deg | {float((e < np.pi / 2).float().mean()) * 100:5.1f}% | {float((eo[coreo] < np.pi / 2).float().mean()) * 100:5.1f}% / {float(np.degrees(eo[coreo].median())):5.1f} deg")
