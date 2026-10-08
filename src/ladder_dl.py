"""Download the CT (L2), surface prediction and lasagna chunks around every human winding-ladder collection (union of chunks, no duplicates), into G:/vesuvius_ladder (zarr v2 layout).
Regions: ladder bounding box plus MARGIN voxels on every side, at least MINSIZE, aligned to 4.  Writes regions.json (box origin z,y,x and size per collection).
  python ladder_dl.py plan            -> counts chunks and estimates bytes
  python ladder_dl.py get [WORKERS]   -> downloads"""
import os, sys, json, subprocess, time
import numpy as np
from concurrent.futures import ThreadPoolExecutor

B = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHercParis4"
CT_URL = B + "/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/2"
SP_URL = B + "/representations/predictions/surfaces/20260411134726-surface-20260413222639-surface-m7-L2-th0.2.zarr/0"
LAS = B + "/representations/predictions/lasagna/20260411134726-lasagna-20260419180421-L2/PHercParis4-20260411134726-las-sd2-5b17ff6c_{}.ome.zarr/{}"
OUT = "G:/vesuvius_ladder/"; MARGIN = 80; MINSIZE = 192
SHAPE = (18946, 8174, 8174)


def regions():
    out = []
    for f in ("relative_windings.json", "abs_winding.json"):
        for name, c in json.load(open(os.environ.get("LADDER_DIR", "layerjudge/") + f))["collections"].items():
            pts = np.array([p["p"] for p in c["points"].values()], float).reshape(-1, 3)[:, [2, 1, 0]]
            if len(pts) < 2: continue
            lo = pts.min(0) - MARGIN; hi = pts.max(0) + MARGIN; mid = (lo + hi) / 2; half = np.maximum((hi - lo) / 2, MINSIZE / 2)
            lo = np.floor((mid - half) / 4).astype(int) * 4; hi = np.ceil((mid + half) / 4).astype(int) * 4
            lo = np.maximum(lo, 0); hi = np.minimum(hi, SHAPE)
            out.append({"id": f"{f[:3]}_{name}", "lo": lo.tolist(), "hi": hi.tolist()})
    return out


def rng(a, b, c): return range(a // c, (b - 1) // c + 1)


def jobs(regs):
    J = {}
    for r in regs:
        (z0, y0, x0), (z1, y1, x1) = r["lo"], r["hi"]
        for z in rng(z0, z1, 128):
            for y in rng(y0, y1, 128):
                for x in rng(x0, x1, 128): J[f"{CT_URL}/{z}/{y}/{x}"] = f"{OUT}ct/2/{z}/{y}/{x}"
        for z in rng(z0, z1, 192):
            for y in rng(y0, y1, 192):
                for x in rng(x0, x1, 192): J[f"{SP_URL}/{z}/{y}/{x}"] = f"{OUT}sp/0/{z}/{y}/{x}"
        for name, lv in (("cos", 3), ("nx", 4), ("ny", 4), ("grad_mag", 4)):
            f = 4 / 2 ** lv
            for z in rng(int(z0 * f), int(z1 * f), 64):
                for y in rng(int(y0 * f), int(y1 * f), 64):
                    for x in rng(int(x0 * f), int(x1 * f), 64): J[LAS.format(name, f"{lv}/{z}/{y}/{x}")] = f"{OUT}lasagna/{name}/{lv}/{z}/{y}/{x}"
    return J


def get(job):
    url, path = job
    if os.path.exists(path) and os.path.getsize(path) > 0: return 0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for attempt in range(4):
        r = subprocess.run(["curl", "-s", "-f", "--max-time", "300", "-o", path + ".part", url])
        if r.returncode == 0: os.replace(path + ".part", path); return 0
        if r.returncode == 22: open(path, "wb").close(); return 0                                  # 404: empty chunk
        time.sleep(1 + attempt)
    return 1


if __name__ == "__main__":
    regs = regions(); json.dump(regs, open(OUT.rstrip("/") + "_regions.json" if False else "E:/vesuvius_ladder_regions.json", "w")); J = jobs(regs)
    n_ct = sum("volumes" in u for u in J); n_sp = sum("surfaces" in u for u in J); n_las = len(J) - n_ct - n_sp
    print(f"{len(regs)} regions; chunks: CT {n_ct} (about {n_ct * 2.1 / 1024:.1f} GB raw), surface prediction {n_sp} (about {n_sp * 0.3 / 1024:.1f} GB), lasagna {n_las} (about {n_las * 0.2 / 1024:.1f} GB)")
    if sys.argv[1] == "get":
        w = int(sys.argv[2]) if len(sys.argv) > 2 else 24; t0 = time.time(); fails = 0
        with ThreadPoolExecutor(w) as ex:
            for i, r in enumerate(ex.map(get, J.items())):
                fails += r
                if i % 500 == 0: print(f"  {i}/{len(J)} done, {time.time() - t0:.0f} s, failures {fails}", flush=True)
        print(f"finished {len(J)} jobs, failures {fails}, {time.time() - t0:.0f} s")
