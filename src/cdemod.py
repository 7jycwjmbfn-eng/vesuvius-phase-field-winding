"""Carrier-compensated complex filtering of the predicted phase field on the GPU (route 1 of the vortex-loss survey).
u = conf * exp(i psi) averaged 2x2x2 to the D=2 grid; a least-squares carrier phi is fitted to the wrapped edge differences (weighted, Jacobi-preconditioned CG), the field is demodulated,
smoothed with a Gaussian of SIGMA grid cells and modulated back: psi_hat = wrap(phi + arg v_f), gate amplitude |v_f|.  Dipoles of phase vortices closer than SIGMA vanish; single vortices stay.
  python cdemod.py PSI_FULL.npy CONF_FULL.npy OUT_PREFIX SIGMA [SIGMA2 ...]     (writes OUT_PREFIX_s<SIGMA>_psi.npy and _conf.npy at the D=2 grid, float16)"""
import sys, time
import numpy as np
import torch
import torch.nn.functional as F

TWO = 2 * torch.pi
dev = "cuda"


def wrap(a): return a - TWO * torch.round(a / TWO)


def pairs():
    for ax in range(3):
        a = [slice(None)] * 3; b = [slice(None)] * 3; a[ax] = slice(0, -1); b[ax] = slice(1, None); yield tuple(a), tuple(b)


@torch.no_grad()
def ls_carrier(u, iters=3000, rtol=1e-6, eps=1e-4):
    au = u.abs(); P = list(pairs())
    G = [torch.angle(u[a].conj() * u[b]) for a, b in P]; W = [au[a] * au[b] + eps for a, b in P]
    def Dt(r, k, out): a, b = P[k]; out[b] += r; out[a] -= r
    rhs = torch.zeros_like(au); diag = torch.zeros_like(au)
    for k, (a, b) in enumerate(P): Dt(W[k] * G[k], k, rhs); diag[a] += W[k]; diag[b] += W[k]
    del G
    def A(p):
        out = torch.zeros_like(p)
        for k, (a, b) in enumerate(P): Dt(W[k] * (p[b] - p[a]), k, out)
        return out
    x = torch.zeros_like(au); r = rhs.clone(); z = r / diag; p = z.clone(); rz = (r * z).sum(); n0 = rhs.norm(); t0 = time.time()
    for it in range(iters):
        Ap = A(p); al = rz / (p * Ap).sum(); x += al * p; r -= al * Ap
        if it % 250 == 0: print(f"    CG {it}: residual {float(r.norm() / n0):.2e}, {time.time() - t0:.0f} s", flush=True)
        if r.norm() < rtol * n0: break
        z = r / diag; rz2 = (r * z).sum(); p = z + (rz2 / rz) * p; rz = rz2
    print(f"    CG done after {it + 1} iterations, residual {float(r.norm() / n0):.2e}", flush=True)
    return x


def gauss3d(x, s):
    r = int(3 * s + 0.5); t = torch.arange(-r, r + 1, dtype=x.dtype, device=x.device); k = torch.exp(-t ** 2 / (2 * s * s)); k = k / k.sum(); y = x[None, None]
    for ax in range(3):
        shp = [1, 1, 1, 1, 1]; shp[2 + ax] = -1; pad = [0] * 6; pad[2 * (2 - ax)] = pad[2 * (2 - ax) + 1] = r
        y = F.conv3d(F.pad(y, pad, mode="replicate"), k.view(shp))
    return y[0, 0]


def vortex_per_million(psi):
    tot = 0; bad = 0
    for a0, a1 in ((0, 1), (0, 2), (1, 2)):
        def sl(i, j):
            idx = [slice(None)] * 3
            for ax, o in ((a0, i), (a1, j)): idx[ax] = slice(o, psi.shape[ax] - 1 + o)
            return psi[tuple(idx)]
        circ = wrap(sl(1, 0) - sl(0, 0)) + wrap(sl(1, 1) - sl(1, 0)) + wrap(sl(0, 1) - sl(1, 1)) + wrap(sl(0, 0) - sl(0, 1)); k_ = torch.round(circ / TWO); tot += k_.numel(); bad += int((k_ != 0).sum())
    return bad / tot * 1e6


def main():
    pf, cf, out = sys.argv[1], sys.argv[2], sys.argv[3]; sigmas = [float(v) for v in sys.argv[4:]]; t0 = time.time()
    P = np.load(pf, mmap_mode="r"); C = np.load(cf, mmap_mode="r"); Z, Y, X = P.shape; Z2, Y2, X2 = Z // 2, Y // 2, X // 2
    u = torch.zeros(Z2, Y2, X2, dtype=torch.complex64, device=dev)
    for k in range(0, Z, 32):
        p = torch.from_numpy(np.asarray(P[k:k + 32]).astype(np.float32)).to(dev); c = torch.from_numpy(np.asarray(C[k:k + 32]).astype(np.float32)).to(dev)
        n = p.shape[0] // 2; z = torch.complex(c * torch.cos(p), c * torch.sin(p)); z = torch.view_as_real(z).permute(3, 0, 1, 2)                      # [2, z, y, x]
        z = F.avg_pool3d(z[:, :2 * n], 2); u[k // 2:k // 2 + n] = torch.complex(z[0], z[1])
    print(f"D=2 field {tuple(u.shape)}, vortex {vortex_per_million(torch.angle(u)):.0f} per million, mean |u| {float(u.abs().mean()):.3f}, {time.time() - t0:.0f} s", flush=True)
    phi = ls_carrier(u)
    v = u * torch.exp(-1j * phi)
    for s in sigmas:
        vf = torch.complex(gauss3d(v.real, s), gauss3d(v.imag, s)); psi_hat = wrap(phi + torch.angle(vf)); amp = vf.abs()
        print(f"sigma {s}: vortex {vortex_per_million(psi_hat):.0f} per million, mean |v_f| {float(amp.mean()):.3f}, {time.time() - t0:.0f} s", flush=True)
        np.save(f"{out}_s{s:g}_psi.npy", psi_hat.cpu().numpy().astype(np.float16)); np.save(f"{out}_s{s:g}_conf.npy", amp.cpu().numpy().astype(np.float16))
    np.save(f"{out}_s0_psi.npy", torch.angle(u).cpu().numpy().astype(np.float16)); np.save(f"{out}_s0_conf.npy", u.abs().cpu().numpy().astype(np.float16)); print("saved", out, flush=True)


if __name__ == "__main__":
    main()
