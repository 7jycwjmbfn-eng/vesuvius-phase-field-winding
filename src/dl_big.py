"""Threaded download of the PHercParis4 CT (level 2, 128^3 raw chunks) and official surface prediction (m7-L2-th0.2, 192^3 blosc chunks) for the big box
centred on the umbilicus: L2 z 10496-11008, y 1920-4992, x 3328-6400.  Files are stored in zarr v2 layout under G:/vesuvius_big; existing files are skipped.
  python dl_big.py [ct|sp|both] [WORKERS]"""
import os, sys, subprocess, time
from concurrent.futures import ThreadPoolExecutor
B = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHercParis4"
CT_URL = B + "/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/2"
SP_URL = B + "/representations/predictions/surfaces/20260411134726-surface-20260413222639-surface-m7-L2-th0.2.zarr/0"
OUT = os.environ.get("BOX_DIR", "G:/vesuvius_big/")
Z0, Z1, Y0, Y1, X0, X1 = (int(v) for v in os.environ.get("BOX_RANGE", "10496,11008,1920,4992,3328,6400").split(","))


def rng(a, b, c): return range(a // c, (b - 1) // c + 1)


def jobs(what):
    j = []
    if what in ("ct", "both"):
        j += [(f"{CT_URL}/{z}/{y}/{x}", f"{OUT}ct/2/{z}/{y}/{x}") for z in rng(Z0, Z1, 128) for y in rng(Y0, Y1, 128) for x in rng(X0, X1, 128)]
    if what in ("sp", "both"):
        j += [(f"{SP_URL}/{z}/{y}/{x}", f"{OUT}sp/0/{z}/{y}/{x}") for z in rng(Z0, Z1, 192) for y in rng(Y0, Y1, 192) for x in rng(X0, X1, 192)]
    return j


def get(job):
    url, path = job
    if os.path.exists(path) and os.path.getsize(path) > 0: return 0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for attempt in range(4):
        r = subprocess.run(["curl", "-s", "-f", "--max-time", "600", "-o", path + ".part", url])
        if r.returncode == 0: os.replace(path + ".part", path); return 0
        if r.returncode == 22:                                              # 404: the chunk does not exist (all-zero chunk in zarr)
            open(path, "wb").close(); return 0
        time.sleep(2 * (attempt + 1))
    return 1


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "both"; nw = int(sys.argv[2]) if len(sys.argv) > 2 else 16; J = jobs(what); t = time.time(); fails = 0
    print(f"{len(J)} chunk jobs ({what})", flush=True)
    with ThreadPoolExecutor(nw) as ex:
        for i, r in enumerate(ex.map(get, J)):
            fails += r
            if i % 200 == 0: print(f"  {i}/{len(J)} done, {time.time() - t:.0f} s, failures {fails}", flush=True)
    print(f"finished {len(J)} jobs, failures {fails}, {time.time() - t:.0f} s", flush=True)
