"""Candidate 3D architectures for the phase-field network (same interface as train_v6.UNet5: forward(x, full=False)).
  plain   : two 3x3x3 conv + GroupNorm + SiLU per level (the v3-v5 design, 5 levels)
  resenc  : pre-activation residual blocks (2 per encoder level, 1 per decoder level), nnU-Net ResEnc style
  mednext : depthwise 5x5x5 conv -> GroupNorm -> pointwise expand x2 -> GELU -> pointwise project, with a residual connection (MedNeXt / ConvNeXt style)"""
import torch, torch.nn as nn, torch.utils.checkpoint as cp
import train_v3 as t3
NB = t3.NB


class ResBlock(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.n1 = nn.GroupNorm(8, ci); self.c1 = nn.Conv3d(ci, co, 3, padding=1, bias=False); self.n2 = nn.GroupNorm(8, co); self.c2 = nn.Conv3d(co, co, 3, padding=1, bias=False)
        self.skip = nn.Conv3d(ci, co, 1, bias=False) if ci != co else nn.Identity(); self.act = nn.SiLU(inplace=False)
    def forward(self, x):
        h = self.c1(self.act(self.n1(x))); h = self.c2(self.act(self.n2(h))); return h + self.skip(x)


class NextBlock(nn.Module):
    def __init__(self, ci, co, k=5, exp=2):
        super().__init__()
        self.proj = nn.Conv3d(ci, co, 1, bias=False) if ci != co else nn.Identity()
        self.dw = nn.Conv3d(co, co, k, padding=k // 2, groups=co, bias=False); self.n = nn.GroupNorm(8, co)
        self.pw1 = nn.Conv3d(co, co * exp, 1); self.act = nn.GELU(); self.pw2 = nn.Conv3d(co * exp, co, 1)
    def forward(self, x):
        x = self.proj(x); return x + self.pw2(self.act(self.pw1(self.n(self.dw(x)))))


def make_block(kind, ci, co, n):
    if kind == "plain":
        return nn.Sequential(nn.Conv3d(ci, co, 3, padding=1, bias=False), nn.GroupNorm(8, co), nn.SiLU(inplace=True), nn.Conv3d(co, co, 3, padding=1, bias=False), nn.GroupNorm(8, co), nn.SiLU(inplace=True))
    if kind == "resenc": return nn.Sequential(*[ResBlock(ci if i == 0 else co, co) for i in range(n)])
    return nn.Sequential(*[NextBlock(ci if i == 0 else co, co) for i in range(n)])


class Net(nn.Module):
    def __init__(self, kind="plain", cin=8, ch=(24, 48, 96, 192, 288), ckpt=True):
        super().__init__(); self.ckpt = ckpt; self.kind = kind
        self.enc = nn.ModuleList([make_block(kind, cin if i == 0 else ch[i - 1], c, 2) for i, c in enumerate(ch)])
        self.down = nn.ModuleList([nn.Conv3d(c, c, 2, stride=2) for c in ch[:-1]])
        self.up = nn.ModuleList([nn.ConvTranspose3d(ch[i + 1], ch[i], 2, stride=2) for i in range(len(ch) - 1)])
        self.dec = nn.ModuleList([make_block(kind, 2 * ch[i], ch[i], 1) for i in range(len(ch) - 1)])
        self.norm = nn.GroupNorm(8, ch[0]) if kind != "plain" else nn.Identity(); self.head = nn.Conv3d(ch[0], 2 + NB, 1)
    def _run(self, f, x): return cp.checkpoint(f, x, use_reentrant=False) if (self.ckpt and self.training) else f(x)
    def forward(self, x, full=False):
        skips = []
        for i, e in enumerate(self.enc):
            x = self._run(e, x)
            if i < len(self.down): skips.append(x); x = self.down[i](x)
        for i in reversed(range(len(self.up))): x = self._run(self.dec[i], torch.cat([self.up[i](x), skips[i]], 1))
        o = self.head(self.norm(x))
        if full: return o
        return self.readout(o)

    @staticmethod
    def readout(o, mode=None):
        """(sin, cos) vector from the raw output: 'reg' = regression head (default; its length is the confidence), 'cls' = circular mean of the 24-bin softmax, 'avg' = mean of both"""
        import os
        mode = mode or os.environ.get("READOUT", "reg"); reg = o[:, :2].float()
        p = torch.softmax(o[:, 2:].float(), 1); c = t3.CENT.to(o.device).view(1, NB, 1, 1, 1); cls = torch.stack([(p * torch.sin(c)).sum(1), (p * torch.cos(c)).sum(1)], 1)
        return reg if mode == "reg" else cls if mode == "cls" else 0.5 * (reg + cls)
