"""Read-only view of several equally shaped (in y, x) arrays stacked along z; supports basic 3-slice indexing and .shape (enough for the tile readers in predict_*.py)."""
import numpy as np


class Cat:
    def __init__(self, arrs):
        self.arrs = list(arrs); self.offs = np.r_[0, np.cumsum([a.shape[0] for a in self.arrs])].astype(int); self.shape = (int(self.offs[-1]),) + tuple(self.arrs[0].shape[1:])

    def __getitem__(self, key):
        zs = key[0]; rest = tuple(key[1:]); z0 = 0 if zs.start is None else int(zs.start); z1 = self.shape[0] if zs.stop is None else int(zs.stop); out = []
        for a, o in zip(self.arrs, self.offs[:-1]):
            lo, hi = max(z0, o), min(z1, o + a.shape[0])
            if hi > lo: out.append(np.asarray(a[(slice(lo - o, hi - o),) + rest]))
        return np.concatenate(out, 0) if len(out) > 1 else out[0]
