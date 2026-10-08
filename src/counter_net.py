"""Winding-count network: model, npz loading (zero-copy mmap of stored npz members) and synthetic data for tests.

Input  P [B, 14, L]  float16/float32, channels: 0 CT/255, 1 surface prediction/255, 2 sin psi_v6, 3 cos psi_v6, 4 v6 confidence, 5 sin psi_frozen, 6 cos psi_frozen,
                     7,8 sin/cos v6 on the +3 voxel parallel line, 9,10 same on the -3 line, 11 lasagna cos/255, 12 lasagna grad_mag/255, 13 validity mask (prefix of ones).
       S [B, 6]      float32: L/100, r_a/1000, |cos(line, radial)|, dz/L, z_mid/10000, jitter (jitter is NOT used by the model: it is not known at deployment).
Output logits [B, 9]: classes k = 0..7 and "8 or more".

Design choices
  * every convolution and normalisation is mask aware: padded positions are zero after each block, GroupNorm statistics use valid positions only
  * derive=True adds, inside the model, the per-step phase increment of the four sin/cos pairs (atan2 of the cross and dot product of neighbouring unit vectors, in turns), its running sum,
    and the total winding of each phase field as head inputs (sign canonicalised by the median of the four totals). The network is free to ignore them; they make summation of 100+ small
    increments to an integer easy to learn. The increments are invariant to the phase-offset augmentation.
  * S columns used: default 0..3 (L, r_a, |cos|, |dz/L|). z_mid (col 4) is dropped because the truth pairs cover one narrow z slab and the ladders do not; jitter (col 5) is dropped because it is a label-side quantity.
"""
import math, struct, zipfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

K_CLASSES = 9
N_CH = 14
MAXLEN = 256
PHASE_PAIRS = ((2, 3), (5, 6), (7, 8), (9, 10))
LAS_CH = (11, 12)


# ---------------------------------------------------------------------------------------------------- model
def derive_features(x, m):
    """x [B, C, L] float32 (masked), m [B, L] float32 -> extra [B, 8, L], scalars [B, 4]."""
    s = torch.stack([x[:, a] for a, _ in PHASE_PAIRS], 1); c = torch.stack([x[:, b] for _, b in PHASE_PAIRS], 1)                 # [B, 4, L]
    cross = s[..., :-1] * c[..., 1:] - c[..., :-1] * s[..., 1:]; dot = c[..., :-1] * c[..., 1:] + s[..., :-1] * s[..., 1:]
    d = torch.atan2(cross, dot) / (2 * math.pi); d = F.pad(d, (1, 0))                                                          # d[..., t] = phase step t-1 -> t, in turns; d[..., 0] = 0
    valid = m[:, None, :] * F.pad(m[:, None, :-1], (1, 0)); d = d * valid
    cum = d.cumsum(-1); tot = cum[..., -1]                                                                                    # [B, 4]
    med = torch.sort(tot, 1).values[:, 1:3].mean(1); sgn = torch.where(med >= 0, 1.0, -1.0)                                    # canonical direction: median winding >= 0
    extra = torch.cat([d * sgn[:, None, None] * 16.0, cum * sgn[:, None, None] * 0.5], 1) * m[:, None, :]
    scal = (tot * sgn[:, None] / 4.0).clamp(-3.0, 3.0)
    return extra, scal


class MaskedGroupNorm(nn.Module):
    def __init__(self, groups, ch, eps=1e-5):
        super().__init__(); self.g = groups; self.eps = eps; self.weight = nn.Parameter(torch.ones(ch)); self.bias = nn.Parameter(torch.zeros(ch))

    def forward(self, x, m):                                                                                                    # x [B, C, L] (bf16 or fp32), m [B, 1, L]; statistics accumulate in fp32 over valid positions
        B, C, L = x.shape; g = self.g; xg = x.view(B, g, C // g, L); mb = m.to(x.dtype).view(B, 1, 1, L); cnt = m.float().sum((1, 2)).clamp_min(1.0).view(B, 1) * (C // g)
        mean = (xg * mb).sum((2, 3), dtype=torch.float32) / cnt; xc = (xg - mean.to(x.dtype)[:, :, None, None]) * mb; var = (xc * xc).sum((2, 3), dtype=torch.float32) / cnt
        rstd = torch.rsqrt(var + self.eps); w = self.weight.view(1, g, C // g); b = self.bias.view(1, g, C // g)
        scale = (rstd[:, :, None] * w).reshape(B, C, 1); shift = (b - (mean * rstd)[:, :, None] * w).reshape(B, C, 1)
        return (x * scale.to(x.dtype) + shift.to(x.dtype)) * m.to(x.dtype)


class Block(nn.Module):
    def __init__(self, ch, dil, groups=8):
        super().__init__()
        self.n1 = MaskedGroupNorm(groups, ch); self.c1 = nn.Conv1d(ch, ch, 3, padding=dil, dilation=dil)
        self.n2 = MaskedGroupNorm(groups, ch); self.c2 = nn.Conv1d(ch, ch, 3, padding=1)

    def forward(self, x, m):
        h = self.c1(F.silu(self.n1(x, m))); h = h * m.to(h.dtype)
        h = self.c2(F.silu(self.n2(h, m))); h = h * m.to(h.dtype)
        return x + h


class CounterNet(nn.Module):
    def __init__(self, in_ch=N_CH, s_cols=(0, 2, 3), s_abs=(3,), c1=64, c2=128, dils=(1, 2, 4, 8, 16), K=K_CLASSES, derive=True, p_drop=0.1, s_emb=32):
        super().__init__()
        self.derive = derive; self.s_cols = tuple(s_cols); self.s_abs = tuple(s_abs); self.K = K
        self.register_buffer("sel", torch.tensor(self.s_cols, dtype=torch.long), persistent=False)
        self.register_buffer("absmask", torch.tensor([1.0 if c in self.s_abs else 0.0 for c in self.s_cols]), persistent=False)
        cin = in_ch + (8 if derive else 0)
        self.stem = nn.Conv1d(cin, c1, 5, padding=2); self.stem_n = MaskedGroupNorm(8, c1)
        self.stage1 = nn.ModuleList([Block(c1, d) for d in dils])
        self.up = nn.Conv1d(c1, c2, 1); self.up_n = MaskedGroupNorm(8, c2)
        self.stage2 = nn.ModuleList([Block(c2, d) for d in dils])
        self.out_n = MaskedGroupNorm(8, c2)
        self.att = nn.Conv1d(c2, 1, 1)
        self.s_lin = nn.Linear(len(self.s_cols), s_emb)
        hd = 3 * c2 + s_emb + (4 if derive else 0)
        self.head = nn.Sequential(nn.Linear(hd, 256), nn.SiLU(), nn.Dropout(p_drop), nn.Linear(256, K))

    def config(self):
        return dict(s_cols=list(self.s_cols), s_abs=list(self.s_abs), c1=self.stem.out_channels, c2=self.up.out_channels, derive=self.derive, K=self.K,
                    dils=[b.c1.dilation[0] for b in self.stage1])

    def forward(self, P, S):
        P = torch.nan_to_num(P.float(), nan=0.0, posinf=0.0, neginf=0.0); m = (P[:, 13:14] > 0.5).float(); x = P * m                                                                      # mask aware input: invalid points are zero
        if self.derive:
            ex, scal = derive_features(x, m[:, 0]); x = torch.cat([x, ex], 1)
        h = F.silu(self.stem_n(self.stem(x), m)); h = h * m.to(h.dtype)
        for b in self.stage1: h = b(h, m)
        h = F.silu(self.up_n(self.up(h), m)); h = h * m.to(h.dtype)
        for b in self.stage2: h = b(h, m)
        h = F.silu(self.out_n(h, m)); h = h * m.to(h.dtype)
        a = self.att(h).float().squeeze(1).masked_fill(m[:, 0] < 0.5, -1e4); a = torch.softmax(a, -1).to(h.dtype)
        att = (h * a[:, None]).sum(-1); mean = h.sum(-1) / m.sum(-1).clamp_min(1.0).to(h.dtype); mx = h.masked_fill(m < 0.5, -1e4).amax(-1)
        Ss = torch.nan_to_num(S.float()[:, self.sel], nan=0.0); Ss = torch.where(self.absmask[None] > 0, Ss.abs(), Ss)
        z = [att, mean, mx, F.silu(self.s_lin(Ss)).to(h.dtype)]
        if self.derive: z.append(scal.to(h.dtype))
        return self.head(torch.cat(z, 1)).float()


def count_params(model): return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------------------------------- npz loading
def mmap_npz(path, key):
    """Zero-copy np.memmap of an array stored (not compressed) in an npz file; None if that is not possible."""
    try:
        with zipfile.ZipFile(path) as z:
            info = z.getinfo(key + ".npy")
            if info.compress_type != zipfile.ZIP_STORED: return None
            off = info.header_offset
        with open(path, "rb") as f:
            f.seek(off); hdr = f.read(30)
            if hdr[:4] != b"PK\x03\x04": return None
            n_len, e_len = struct.unpack("<HH", hdr[26:30]); f.seek(off + 30 + n_len + e_len)
            ver = np.lib.format.read_magic(f)
            if ver == (1, 0): shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
            elif ver == (2, 0): shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
            else: return None
            start = f.tell()
        if fortran or dtype.hasobject: return None
        return np.memmap(path, dtype=dtype, mode="r", offset=start, shape=tuple(shape))
    except Exception:
        return None


class PairSet:
    """Several PROFILE npz files viewed as one set. P stays on disk (memmap) when the npz is stored uncompressed; S, k and the extra keys are loaded."""
    EXTRA = ("votes", "A", "B", "zrel", "xcol", "ri", "seq", "dw", "zmid")

    def __init__(self, paths, name=""):
        self.paths = list(paths); self.name = name; self.P = []; S = []; k = []; ex = {}; self.n_dropped = 0; self.mapped = []
        for p in self.paths:
            z = np.load(p); mm = mmap_npz(p, "P"); self.mapped.append(mm is not None); self.P.append(mm if mm is not None else z["P"])
            S.append(np.asarray(z["S"], np.float32)); k.append(np.asarray(z["k"]).astype(np.int64))
            if "n_dropped" in z.files: self.n_dropped += int(z["n_dropped"])
            for key in self.EXTRA:
                if key in z.files: ex.setdefault(key, []).append(np.asarray(z[key]))
            z.close(); assert self.P[-1].shape[0] == len(S[-1]) == len(k[-1]), f"{p}: P/S/k length mismatch"
            assert self.P[-1].shape[1:] == (N_CH, MAXLEN), f"{p}: P shape {self.P[-1].shape}"
        self.S = np.concatenate(S); self.k = np.concatenate(k); self.N = len(self.k)
        self.extra = {key: np.concatenate(v) for key, v in ex.items() if len(v) == len(self.paths)}
        self.off = np.cumsum([0] + [len(s) for s in S])

    def __len__(self): return self.N

    def get(self, idx):
        """idx: sorted int array -> float16 [B, 14, 256] (copy)."""
        idx = np.asarray(idx); f = np.searchsorted(self.off, idx, side="right") - 1; out = np.empty((len(idx), N_CH, MAXLEN), np.float16)
        for i in np.unique(f):
            sel = np.nonzero(f == i)[0]; out[sel] = self.P[i][idx[sel] - self.off[i]]
        return out

    @property
    def jitter(self): return self.S[:, 5]

    def subset_info(self): return f"{self.name}: {self.N} pairs in {len(self.paths)} file(s), P {'mmap' if all(self.mapped) else 'RAM'}"


# ---------------------------------------------------------------------------------------------------- synthetic data
def make_synthetic(n, seed=0, lmax=200, ladder=False, with_votes=True, lasagna=True):
    """Random data with the contract shapes. k is the winding count of channels 2/3 (v6 sin/cos): total phase change = k turns (+ noise) along the line.
    Channels 5,6 (frozen) copy it with noise and a 15% chance of an off-by-one winding; channels 7..10 are shifted noisy copies; the rest is noise."""
    rng = np.random.default_rng(seed); P = np.zeros((n, N_CH, MAXLEN), np.float16); probs = np.array([0.02, 0.30, 0.25, 0.20, 0.15, 0.04, 0.02, 0.01, 0.01]); probs /= probs.sum()
    k = rng.choice(K_CLASSES, n, p=probs).astype(np.int16); L = rng.integers(30, lmax + 1, n).astype(np.float64); npts = np.minimum(np.ceil(L).astype(int) + 1, MAXLEN)
    S = np.zeros((n, 6), np.float32); t = np.arange(MAXLEN)
    for i in range(n):
        m = npts[i]; u = np.linspace(0, 1, m); u = u + 0.04 * np.sin(2 * np.pi * rng.uniform(0.5, 3) * u + rng.uniform(0, 6.3)) * np.sin(np.pi * u); sg = rng.choice([-1, 1])
        ph = sg * 2 * np.pi * (k[i] * u + rng.uniform(-0.1, 0.1))
        amp = 0.7 + 0.3 * rng.random(m); noise = lambda s_: np.convolve(rng.normal(0, s_, m + 6), np.ones(7) / 7, "valid")[:m] * 2.6
        v = ph + noise(0.15); fr = ph + noise(0.35) + (sg * 2 * np.pi * rng.choice([-1, 1]) * u if rng.random() < 0.15 else 0)
        p3 = ph + noise(0.2); m3 = ph + noise(0.2)
        P[i, 2, :m] = amp * np.sin(v); P[i, 3, :m] = amp * np.cos(v); P[i, 5, :m] = np.sin(fr); P[i, 6, :m] = np.cos(fr)
        P[i, 7, :m] = np.sin(p3); P[i, 8, :m] = np.cos(p3); P[i, 9, :m] = np.sin(m3); P[i, 10, :m] = np.cos(m3)
        P[i, 4, :m] = np.clip(0.9 + 0.1 * rng.normal(size=m), 0, 1.2); P[i, 0, :m] = np.clip(0.5 + 0.2 * rng.normal(size=m), 0, 1); P[i, 1, :m] = np.clip(0.3 + 0.3 * rng.normal(size=m), 0, 1)
        if lasagna: P[i, 11, :m] = np.clip(0.5 + 0.3 * rng.normal(size=m), 0, 1); P[i, 12, :m] = np.clip(0.2 + 0.1 * rng.normal(size=m), 0, 1)
        P[i, 13, :m] = 1
        S[i] = [L[i] / 100, rng.uniform(0.3, 2.0), rng.uniform(0.5, 1.0), rng.uniform(-0.3, 0.3), 1.05 + 0.05 * rng.random(), np.nan if ladder else rng.uniform(0, 0.2)]
    out = dict(P=P, S=S, k=k)
    if with_votes:
        kf = k.astype(np.float64); nz = lambda s_: rng.normal(0, s_, n); wrong = lambda: np.where(rng.random(n) < 0.12, rng.choice([-1, 1], n), 0)
        votes = np.c_[kf + nz(0.12) + wrong(), kf + nz(0.2) + wrong(), kf + nz(0.2) + wrong(), kf + nz(0.3) + wrong(), kf + nz(0.1) + wrong(), kf + nz(0.1) + wrong(), kf * 0.9 + nz(0.15),
                      kf * 0.9 + nz(0.2), np.maximum(kf + wrong(), 0), np.maximum(kf + wrong(), 0), np.maximum(kf + wrong(), 0), np.maximum(kf + wrong(), 0),
                      rng.uniform(0.1, 0.9, n), rng.uniform(0.5, 1.0, n), rng.uniform(0.1, 0.9, n), L * 2.0]
        out["votes"] = votes.astype(np.float32)
    if ladder:
        ri = np.sort(rng.integers(0, 40, n)).astype(np.int32); out.update(ri=ri, seq=np.arange(n, dtype=np.int32), dw=k.astype(np.float32), zmid=rng.uniform(2000, 20000, n).astype(np.float32), n_dropped=np.int64(3))
    else:
        out.update(A=rng.uniform(0, 1000, (n, 3)).astype(np.float32), B=rng.uniform(0, 1000, (n, 3)).astype(np.float32), zrel=rng.uniform(0, 500, n).astype(np.float32), xcol=rng.uniform(0, 1536, n).astype(np.float32))
    return out
