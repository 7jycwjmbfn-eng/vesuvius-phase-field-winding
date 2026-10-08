"""Download the PHercParis4 lasagna channels (cos level 3, nx / ny / grad_mag level 4; zstd zarr v2, 64^3 chunks) for the big box, into G:/vesuvius_big/lasagna.
Big box (L2 voxels): z 10496-11008, y 1920-4992, x 3328-6400; L0 = 4 x L2, level l = L0 / 2^l."""
import os, sys, subprocess, time
from concurrent.futures import ThreadPoolExecutor
B = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHercParis4/representations/predictions/lasagna/20260411134726-lasagna-20260419180421-L2/PHercParis4-20260411134726-las-sd2-5b17ff6c_{}.ome.zarr/{}"
OUT = os.environ.get("LAS_DIR", "G:/vesuvius_big/lasagna/"); _r = [int(v) for v in os.environ.get("BOX_RANGE", "10496,11008,1920,4992,3328,6400").split(",")]; BOX = ((_r[0], _r[1]), (_r[2], _r[3]), (_r[4], _r[5])); C = 64
jobs = []
for name, lv in (("cos", 3), ("nx", 4), ("ny", 4), ("grad_mag", 4)):
    f = 4 / 2 ** lv; r = [(int(a * f) // C, (int(b * f) - 1) // C) for a, b in BOX]
    os.makedirs(OUT + f"{name}/{lv}", exist_ok=True)
    subprocess.run(["curl", "-s", "-m", "30", "-o", OUT + f"{name}/{lv}/.zarray", B.format(name, f"{lv}/.zarray")])
    jobs += [(B.format(name, f"{lv}/{z}/{y}/{x}"), OUT + f"{name}/{lv}/{z}/{y}/{x}") for z in range(r[0][0], r[0][1] + 1) for y in range(r[1][0], r[1][1] + 1) for x in range(r[2][0], r[2][1] + 1)]


def get(j):
    url, path = j
    if os.path.exists(path): return 0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for a in range(4):
        r = subprocess.run(["curl", "-s", "-f", "--max-time", "300", "-o", path + ".part", url])
        if r.returncode == 0: os.replace(path + ".part", path); return 0
        if r.returncode == 22: return 0                                      # chunk does not exist: zeros
        time.sleep(2 * (a + 1))
    return 1


t = time.time(); print(len(jobs), "chunk jobs", flush=True)
with ThreadPoolExecutor(16) as ex: fails = sum(ex.map(get, jobs))
print(f"finished, failures {fails}, {time.time() - t:.0f} s", flush=True)
