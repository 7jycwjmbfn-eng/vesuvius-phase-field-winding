"""PHerc0139 test box: downsample the full-resolution frozen_8c phase by 2 (vector average of sin and cos over each 2x2x2 block, then arctan2).
  python p0139_prep.py [SRC] [OUT]
SRC default G:/vesuvius_wfield_8c/run2/frozen_8c_pred_p0139_4352_5376.npy (float16 radians, (1024,1536,1536))
OUT default E:/vesuvius_p0139_hot/pred/frozen8c_d2.npy (float16 radians, (512,768,768)); output voxel i covers input voxels 2i and 2i+1.
Reads the source in 32-layer chunks and writes the output with plain file writes (a .part file renamed at the end), memory stays below 0.5 GB."""
import os, sys, time
import numpy as np

SRC = sys.argv[1] if len(sys.argv) > 1 else "G:/vesuvius_wfield_8c/run2/frozen_8c_pred_p0139_4352_5376.npy"
OUT = sys.argv[2] if len(sys.argv) > 2 else "E:/vesuvius_p0139_hot/pred/frozen8c_d2.npy"
CH = 32; SUB = 4                                                                                                          # source layers per chunk, source layers per sin/cos step
t0 = time.time(); src = np.load(SRC, mmap_mode="r"); Z, Y, X = src.shape
assert src.dtype == np.float16 and Z % CH == 0 and Y % 2 == 0 and X % 2 == 0 and CH % SUB == 0 and SUB % 2 == 0, (src.dtype, src.shape)
oshape = (Z // 2, Y // 2, X // 2); print(f"source {src.shape} {src.dtype} -> {oshape} float16", flush=True)
part = OUT + ".part"
with open(part, "wb") as f:
    np.lib.format.write_array_header_1_0(f, {"descr": np.lib.format.dtype_to_descr(np.dtype(np.float16)), "fortran_order": False, "shape": oshape})
    for z0 in range(0, Z, CH):
        blk = np.asarray(src[z0:z0 + CH]); outc = np.empty((CH // 2, oshape[1], oshape[2]), np.float16)
        for s0 in range(0, CH, SUB):
            p = blk[s0:s0 + SUB].astype(np.float32); sh = (SUB // 2, 2, oshape[1], 2, oshape[2], 2)
            ms = np.sin(p).reshape(sh).mean(axis=(1, 3, 5)); mc = np.cos(p).reshape(sh).mean(axis=(1, 3, 5))
            outc[s0 // 2:(s0 + SUB) // 2] = np.arctan2(ms, mc).astype(np.float16)
        f.write(outc.tobytes()); del blk
        print(f"  layers {z0}-{z0 + CH} done, {time.time() - t0:.0f} s", flush=True)
os.replace(part, OUT); chk = np.load(OUT, mmap_mode="r"); print("saved", OUT, chk.shape, chk.dtype, f"{time.time() - t0:.0f} s", flush=True)
