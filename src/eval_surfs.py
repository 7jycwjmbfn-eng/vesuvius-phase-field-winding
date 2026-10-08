"""surfkit readouts of a SURF npz in a custom L3 z window (surfkit_8c.py itself is not modified; area_rows is used instead of cmd_area, which counts every surface twice).
  [CROP=y0,y1,x0,x1 (L3)] python eval_surfs.py SURF.npz ZLO ZHI [piece|array]"""
import os, sys
import numpy as np
import surfkit_8c as sk

src = sys.argv[1]; zlo, zhi = float(sys.argv[2]), float(sys.argv[3]); mode = sys.argv[4] if len(sys.argv) > 4 else "piece"
sk.ZW = (zlo, zhi)
surfs = sk.load(src)
if os.environ.get("CROP"):                                  # keep only the surface points inside this L3 xy rectangle
    y0, y1, x0, x1 = (float(v) for v in os.environ["CROP"].split(","))
    surfs = [(n, np.where(((P[..., 1] >= y0) & (P[..., 1] < y1) & (P[..., 2] >= x0) & (P[..., 2] < x1))[..., None], P, np.nan)) for n, P in surfs]
print(f"{src}: {len(surfs)} surfaces, window L3 z {zlo:.0f}-{zhi:.0f}, crop {os.environ.get('CROP', 'none')}")
rows = sk.area_rows(surfs); raw = np.array([r[1] for r in rows]); pc = np.array([r[2] for r in rows]); cb = np.array([r[3] for r in rows]); ct = np.array([r[4] for r in rows])
print(f"area: summed array area {raw.sum():.2f} cm2, largest array {raw.max():.2f}, largest single 4-connected piece {pc.max():.2f} cm2, largest clean contiguous {np.nanmax(cb):.2f} cm2, clean total {np.nansum(ct):.2f} cm2")
sk.jump(surfs, mode=mode)
