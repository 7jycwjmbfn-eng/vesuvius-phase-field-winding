import json, numpy as np
d = json.load(open("E:/vesuvius_ap3_boxA_merged.json"))["collections"]; out = {"collections": {}, "vc_pointcollections_json_version": "1"}; n = 0; kept = 0; tot = 0; npts = 0
for cid, c in d.items():
    ids = sorted(c["points"], key=int); pts = [c["points"][i] for i in ids]; cur = [pts[0]]
    def flush(cur):
        global n, npts
        if len(cur) >= 2:
            n += 1; w0 = cur[0]["wind_a"]; nc = {k: v for k, v in c.items() if k != "points"}; nc["name"] = f"{c['name']}_s{n}"; nc["points"] = {}
            for j, p in enumerate(cur): q = dict(p); q["wind_a"] = float(p["wind_a"] - w0); nc["points"][str(j + 1)] = q
            out["collections"][str(n)] = nc; npts += len(cur)
    for a, b in zip(pts[:-1], pts[1:]):
        tot += 1; step = b["wind_a"] - a["wind_a"]; ok = b["link_prob"] >= 0.999 and min(a["v6_conf"], b["v6_conf"]) >= 0.85 and step <= 2
        if ok: cur.append(b); kept += 1
        else: flush(cur); cur = [b]
    flush(cur)
print(f"links kept {kept} of {tot}; collections {n}, points {npts}")
json.dump(out, open("E:/vesuvius_ap3_boxA_strict.json", "w"))
