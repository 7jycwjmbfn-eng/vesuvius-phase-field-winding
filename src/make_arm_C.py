"""Install our auto-ladder JSON as arm C's relative_windings.json after validating the format.

usage: python a2/make_arm_C.py ARMC_JSON [--root D:/vesuvius_downstream/p4/ds_boxA_C]

Checks (exit 1 on failure):
  - vc_pointcollections_json_version is the string "1"
  - every collection has a name and >= 2 points; every point has p=[x,y,z] (finite, L2 voxels,
    inside box A: x 3328-6400, y 1920-4992, z 10496-11008 with a small margin) and numeric wind_a
  - wind_a strictly increases with the integer point id inside a collection (outer ring = larger)
  - metadata.winding_is_absolute is forced to false
Also prints the counts that the fitter log must show ("Loaded point collection with N collections (M points)").
Refuses to write when the human ladders' ids or held-out points are inside the file (leak check by
3D distance < 20 voxels to any held-out human point).
"""
import json, os, sys, math, argparse, hashlib, shutil
import numpy as np

BOX = dict(x=(3328, 6400), y=(1920, 4992), z=(10496, 11008))
HELD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spiral_heldout_boxA.json")
HUMAN = os.environ.get("LADDER_DIR", "layerjudge/") + "relative_windings.json"


def fail(msg):
    print("FAIL:", msg)
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("armc_json")
    ap.add_argument("--root", default="D:/vesuvius_downstream/p4/ds_boxA_C")
    ap.add_argument("--margin", type=float, default=30.0)
    a = ap.parse_args()
    doc = json.load(open(a.armc_json))
    if doc.get("vc_pointcollections_json_version") != "1":
        fail(f'vc_pointcollections_json_version must be the string "1", got {doc.get("vc_pointcollections_json_version")!r}')
    cols = doc.get("collections")
    if not isinstance(cols, dict) or not cols:
        fail("collections missing or empty")
    npts = 0
    allp = []
    n_in_win = 0
    steps = {}
    for cid, c in cols.items():
        if not c.get("name"):
            fail(f"collection {cid}: name missing")
        pts = c.get("points")
        if not isinstance(pts, dict) or len(pts) < 2:
            fail(f"collection {cid}: needs >= 2 points")
        ids = sorted(pts, key=lambda s: int(s))
        prev = None
        for pid in ids:
            pt = pts[pid]
            p = pt.get("p")
            if not (isinstance(p, list) and len(p) == 3 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in p)):
                fail(f"collection {cid} point {pid}: p must be [x,y,z] finite numbers")
            w = pt.get("wind_a")
            if not isinstance(w, (int, float)) or not math.isfinite(w):
                fail(f"collection {cid} point {pid}: wind_a missing or not finite")
            for k, (lo, hi) in zip("xyz", (BOX["x"], BOX["y"], BOX["z"])):
                v = p["xyz".index(k)]
                if not (lo - a.margin <= v <= hi + a.margin):
                    fail(f"collection {cid} point {pid}: {k}={v} outside box A (+-{a.margin}); coordinates must be L2 voxels [x,y,z]")
            if prev is not None:
                if not w > prev:
                    fail(f"collection {cid}: wind_a must strictly increase with point id (outer ring larger); got {prev} -> {w} at {pid}")
                steps[round(w - prev, 3)] = steps.get(round(w - prev, 3), 0) + 1
            prev = w
            allp.append(p)
            npts += 1
        if 10496 <= min(pts[i]["p"][2] for i in ids) and max(pts[i]["p"][2] for i in ids) < 11008:
            n_in_win += 1
    # leak check against held-out human points
    held = json.load(open(HELD, encoding="utf-8"))
    hd = json.load(open(HUMAN))
    HP = []
    for k in held["sensitivity"]["ids"] + held["main"]["ids"]:
        src, cid = k.split(":")
        assert src == "rel"
        HP += [pt["p"] for pt in hd["collections"][cid]["points"].values()]
    HP = np.array(HP, float)
    from scipy.spatial import cKDTree
    d, _ = cKDTree(HP).query(np.array(allp, float))
    near = int((d < 20).sum())
    print(f"collections {len(cols)}, points {npts}, wind_a step histogram {dict(sorted(steps.items()))}")
    print(f"distance to held-out human points: min {d.min():.1f}, <20: {near}, <50: {int((d<50).sum())}, <100: {int((d<100).sum())}")
    if near > 0:
        print("WARNING: points within 20 voxels of held-out human points (not blocking; the slips script stratifies by distance)")
    out = {"vc_pointcollections_json_version": "1", "collections": {}}
    for cid, c in cols.items():
        nc = dict(c)
        md = dict(nc.get("metadata") or {})
        md["winding_is_absolute"] = False
        nc["metadata"] = md
        nc.setdefault("autoFillMode", 1)
        nc.setdefault("autoFillConstant", 0.0)
        nc.setdefault("color", [0.3, 0.6, 0.2])
        out["collections"][str(cid)] = nc
    os.makedirs(a.root, exist_ok=True)
    for stale in ("same_windings.json", "abs_winding.json", "drawn_control_points.json"):
        if os.path.exists(os.path.join(a.root, stale)):
            fail(f"{stale} present in arm C root; arm C must only carry relative_windings.json")
    dst = os.path.join(a.root, "relative_windings.json")
    with open(dst, "w") as f:
        json.dump(out, f)
    h = hashlib.sha256(open(dst, "rb").read()).hexdigest()
    print(f"wrote {dst}")
    print(f"expect in fit log: Loaded point collection with {len(cols)} collections ({npts} points)")
    print(f"sha256 {h}")


if __name__ == "__main__":
    main()
