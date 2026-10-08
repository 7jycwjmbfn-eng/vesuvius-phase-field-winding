"""Split the ladder profile files into 5 train / test pairs by collection fold (ri % 5), after the certlib exclusions (regions touching the truth box, training z range).
  python make_ladder_folds.py LAD0.npz LAD1.npz OUTDIR"""
import sys, json, os
import numpy as np
f0, f1, out = sys.argv[1], sys.argv[2], sys.argv[3]; os.makedirs(out, exist_ok=True)
d = [np.load(f) for f in (f0, f1)]; keys = ["P", "S", "k", "ri", "seq", "dw", "zmid"]; D = {k: np.concatenate([x[k] for x in d]) for k in keys}
regs = json.load(open("E:/vesuvius_ladder_regions.json")); tb_lo = np.array([9984, 2304, 3584]); tb_hi = tb_lo + np.array([1024, 1536, 1536]); bad = [i for i, r in enumerate(regs) if np.all(np.array(r["lo"]) < tb_hi) and np.all(np.array(r["hi"]) > tb_lo)]
keep = ~np.isin(D["ri"], bad) & ~((D["zmid"] >= 9984) & (D["zmid"] < 10500)); print(f"pairs {len(keep)}; kept {int(keep.sum())} after exclusions ({len(bad)} regions touching the truth box)")
for f in range(5):
    for name, m in (("train", keep & (D["ri"] % 5 != f)), ("test", keep & (D["ri"] % 5 == f))):
        np.savez(os.path.join(out, f"LAD{name}_{f}.npz"), **{k: D[k][m] for k in keys}, n_dropped=0); print(f, name, int(m.sum()))
