"""Assemble downloaded lasagna chunks (zarr v2, zstd, 64^3) into npy arrays for one box.  The arrays keep the lasagna grid: nx, ny, grad_mag are L2/4 (index = L2 / 4),
cos is L2/2 (index = L2 / 2); array origin = box origin (multiples of 4).
  python assemble_las.py LAS_DIR OUT_DIR Z0 Z1 Y0 Y1 X0 X1"""
import os, sys
import numpy as np
import zarr
src, out = sys.argv[1], sys.argv[2]; z0, z1, y0, y1, x0, x1 = (int(v) for v in sys.argv[3:9]); os.makedirs(out, exist_ok=True)
for name, lv, f in (("nx", 4, 4), ("ny", 4, 4), ("grad_mag", 4, 4), ("cos", 3, 2)):
    a = zarr.open(os.path.join(src, name, str(lv)), mode="r")
    sl = (slice(z0 // f, z1 // f), slice(y0 // f, y1 // f), slice(x0 // f, x1 // f)); arr = np.asarray(a[sl])
    np.save(os.path.join(out, ("gm" if name == "grad_mag" else name) + ".npy"), arr); print(name, arr.shape, "nonzero", float((arr > 0).mean()), flush=True)
