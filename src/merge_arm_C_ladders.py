import json, numpy as np, sys
sys.path.insert(0, "."); import train_wfield_8c as tw
uz, uy, ux = tw.umbilicus()
out = {"collections": {}, "vc_pointcollections_json_version": "1"}; n = 0; stats = []
for s in (201, 202, 203):
    d = json.load(open(f"E:/vesuvius_ap3_boxA_s{s}.json"))
    for cid, c in d["collections"].items():
        n += 1; c = dict(c); c["name"] = f"auto_s{s}_{cid}"; out["collections"][str(n)] = c
        ids = sorted(c["points"], key=int); P = np.array([c["points"][i]["p"] for i in ids]); W = np.array([c["points"][i]["wind_a"] for i in ids])
        r = lambda p: np.hypot(p[1] - np.interp(p[2], uz, uy), p[0] - np.interp(p[2], uz, ux))   # p = x,y,z
        rr = np.array([r(p) for p in P]); stats.append((len(ids), W[-1] - W[0], rr[-1] - rr[0], np.all(np.diff(W) > 0), c["metadata"]["phase_sign_along_ray"]))
print("collections", n, "points", sum(len(c["points"]) for c in out["collections"].values()))
S = np.array(stats, float); print("monotone wind_a", S[:, 3].mean(), " radius grows with wind_a:", np.mean(S[:, 2] > 0), "shrinks:", np.mean(S[:, 2] < 0), " mean wind span", S[:, 1].mean())
json.dump(out, open("E:/vesuvius_ap3_boxA_merged.json", "w")); print("written")
