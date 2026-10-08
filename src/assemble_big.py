"""Assemble the downloaded zarr chunks (G:/vesuvius_big) into ct_big.npy and sp_big.npy, shape (512, 3072, 3072), origin L2 (10496, 1920, 3328).
  python assemble_big.py"""
import os, sys
import numpy as np
import zarr
SRC = os.environ.get("BOX_DIR", "G:/vesuvius_big/"); OUT = os.environ.get("BOX_HOT", "D:/vesuvius_big_hot/"); Z0, Z1, Y0, Y1, X0, X1 = (int(v) for v in os.environ.get("BOX_RANGE", "10496,11008,1920,4992,3328,6400").split(","))
shape = (Z1 - Z0, Y1 - Y0, X1 - X0)
os.makedirs(OUT, exist_ok=True); ct = zarr.open(SRC + "ct/2", mode="r"); sp = zarr.open(SRC + "sp/0", mode="r")
for name, a in (("ct_big", ct), ("sp_big", sp)):
    o = np.lib.format.open_memmap(OUT + name + ".npy", mode="w+", dtype=np.uint8, shape=shape)
    for z in range(Z0, Z1, 64):
        o[z - Z0:z - Z0 + 64] = np.asarray(a[z:z + 64, Y0:Y1, X0:X1]); print(name, "z", z, flush=True)
    o.flush(); print(name, shape, "nonzero fraction (subsampled)", float((o[::8, ::8, ::8] > 0).mean()), flush=True)
