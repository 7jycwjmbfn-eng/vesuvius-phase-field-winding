"""Profile file of "truth pairs" for the winding-count network (box A, W2 dense truth, slab z 10500-11000).
Pairs are built exactly like truth_pairs.py: a random ray on the truth slab (radial direction rotated by N(0, 25 deg), small z slope), the truth phase along the ray must run monotonically,
integer phase crossings are sheet centres, k = difference of crossing numbers.  Differences to truth_pairs:
  - k in 1..KMAX (default 6); a fraction P_ZERO (4%) of the pairs use two jittered endpoints of the SAME crossing (k = 0);
  - each endpoint is moved along the line by a random phase offset U(-0.2, 0.2) turns (piecewise-linear inverse of the monotone truth phase), then checked on the real truth phase:
    the nearest sheet centre must still be the one of the crossing (else the pair is dropped);
  - the ray start (z index, x column of the slab) and the pair start a are limited to [ZLO, ZHI) x [XLO, XHI) (spatial hold-out); b may leave the range.
  python profiles_truth.py OUT.npz N SEED ZLO ZHI [XLO XHI]
ZLO, ZHI: slab layer index 0..500 (absolute z = 10500 + index).  XLO, XHI: slab x column 0..1536 (absolute x = 3584 + column), default 0 1536.
Run from the wfield directory.  Env: NOVOTES=1 (votes all NaN, profiles only), RAYLEN (default 140 voxels), KMAX (default 6), KUNIF=1 (draw k uniformly in 1..min(KMAX, crossings-1)
first, then the start crossing; default is the truth_pairs rule: start crossing first, then k), PERP_L2 (default 3, shift of the parallel lines in L2 voxels).
Memory: P takes N * 7168 bytes (N is limited to 120000 per file; run several shards with different seeds / ranges and concatenate).

OUT.npz keys (contract shared with the ladder profile script):
  P      float16 [N, 14, 256]   0 CT/255; 1 surface prediction/255; 2 sin(psi_v6); 3 cos(psi_v6); 4 v6 confidence clipped to [0, 1.2]; 5 sin(psi_frozen); 6 cos(psi_frozen);
                                7, 8 sin, cos of psi_v6 on the parallel line shifted by +PERP_L2 voxels along perp; 9, 10 same for -PERP_L2; 11 lasagna cos/255; 12 lasagna grad_mag/255; 13 valid mask
         Samples lie on the a->b segment at 1 L2 voxel step starting at a: position_i = min(i, L), i = 0..n-1, n = min(ceil(L) + 1, 256), so the last sample is exactly b; zero padded after n.
         psi vectors are trilinear interpolations of sin and cos (not renormalised, magnitude <= 1).  All volumes are trilinear with nearest-edge clamping.
         perp = normalised cross(dir, ref), ref = [1,0,0] or [0,1,0] if |dir_z| >= 0.9 (as in certlib.votes), shifts are PERP_L2 L2 voxels (certlib.votes shifts 3 D=2 voxels = 6 L2 voxels).
  S      float32 [N, 6]         L/100, r_a/1000, |cos(angle between line and radial direction)|, (b_z - a_z)/L, z_mid/10000, jitter (max |phase offset| of the endpoints in turns)
  k      int16 [N]              truth winding count (>= 0)
  A, B   float32 [N, 3]         endpoints, absolute L2 coordinates (z, y, x)
  votes  float32 [N, 16]        certlib.BoxA.votes(A, B) (NaN with NOVOTES=1)
  zrel, xcol float32 [N]        slab layer index (0..500) and slab x column (0..1536) of endpoint a
  n_dropped int                 pairs dropped because L > 255
  meta   str                    command line and constants"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import sys, time, json, math
import numpy as np
import certlib as cl

if os.environ.get("BOX", "A") != "A": sys.exit("profiles_truth: box A only (env BOX must be unset)")
OUT = sys.argv[1]; N = int(sys.argv[2]); SEED = int(sys.argv[3]); ZLO = int(sys.argv[4]); ZHI = int(sys.argv[5])
XLO = int(sys.argv[6]) if len(sys.argv) > 6 else 0; XHI = int(sys.argv[7]) if len(sys.argv) > 7 else 1536
if not (0 <= ZLO < ZHI <= 500 and 0 <= XLO < XHI <= 1536): sys.exit("need 0 <= ZLO < ZHI <= 500 and 0 <= XLO < XHI <= 1536")
if N > 120000: sys.exit("N > 120000: split into shards (memory)")
NOVOTES = os.environ.get("NOVOTES") == "1"; RAYLEN = float(os.environ.get("RAYLEN", 140)); KMAX = int(os.environ.get("KMAX", 6)); KUNIF = os.environ.get("KUNIF") == "1"
PERP_L2 = float(os.environ.get("PERP_L2", 3)); P_ZERO = float(os.environ.get("P_ZERO", 0.04)); JIT = float(os.environ.get("JIT", 0.2)); ANG = float(os.environ.get("ANG", 25)); DZ = float(os.environ.get("DZ", 0.35)); NMAX = 256
LAB = "G:/vesuvius_77f5/psi_w2_8cbox_L2_77f5.npy"; ORG_T = np.array([9984, 2304, 3584.]); ZABS = 10500; ORG_B = cl.ORG_B           # truth label origin (as train_wfield_8c.ORG), slab start, box A origin
cp = sorted(json.load(open(r"D:\vesuvius_downstream\p4\ds\umbilicus.json"))["control_points"], key=lambda c: c["z"]); uz, uy, ux = (np.array([c[t] for c in cp], float) for t in ("z", "y", "x"))
rng = np.random.default_rng(SEED); t0 = time.time()

lab = np.load(LAB, mmap_mode="r"); slab = lab[ZABS - int(ORG_T[0]):ZABS - int(ORG_T[0]) + 500]                                          # memmap view, only the voxels along the rays are read
psi6 = np.load(cl.H + "pred_v6_psi.npy", mmap_mode="r"); conf6 = np.load(cl.H + "pred_v6_conf.npy", mmap_mode="r"); psif = np.load(cl.FROZ, mmap_mode="r")
cosv = np.load(cl.H + "las/cos.npy", mmap_mode="r"); gmv8 = np.load(cl.H + "las/gm.npy", mmap_mode="r"); spv = np.load(cl.SPCT + "sp_big.npy", mmap_mode="r"); ctv = np.load(cl.SPCT + "ct_big.npy", mmap_mode="r")
box = None if NOVOTES else cl.BoxA()
print(f"setup {time.time() - t0:.1f} s, slab {slab.shape}, novotes={NOVOTES}", flush=True)


def interp(vol, c, fns):
    """trilinear interpolation of fn(vol) for every fn in fns at float index coordinates c [M, 3] (clamped to the volume = mode nearest); gathers the 8 corners directly from the memmap"""
    hi = np.array(vol.shape) - 1.0; c = np.clip(c, 0.0, hi); i0 = np.minimum(np.floor(c), hi - 1.0).astype(np.intp); f = c - i0; outs = [np.zeros(len(c)) for _ in fns]
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                w = (f[:, 0] if dz else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dx else 1 - f[:, 2]); v = np.asarray(vol[i0[:, 0] + dz, i0[:, 1] + dy, i0[:, 2] + dx]).astype(np.float32)
                for j, fn in enumerate(fns): outs[j] += w * fn(v)
    return [o.astype(np.float32) for o in outs]


ident = lambda v: v


def profile(a, b):
    """14 x 256 float16 profile of the segment a -> b (absolute L2 coordinates); None if the segment is longer than 255 voxels"""
    d = b - a; Lv = float(np.linalg.norm(d)); n = int(math.ceil(Lv)) + 1
    if n > NMAX: return None
    dirv = d / Lv; ref = np.array([1.0, 0, 0]) if abs(dirv[0]) < 0.9 else np.array([0, 1.0, 0]); perp = np.cross(dirv, ref); perp /= np.linalg.norm(perp)
    pts = a + dirv * np.minimum(np.arange(n), Lv)[:, None]; g = (pts - ORG_B - 0.5) / 2.0; sh = PERP_L2 / 2.0 * perp; Pm = np.zeros((14, NMAX), np.float32)
    s_, c_ = interp(psi6, np.concatenate([g, g + sh, g - sh]), [np.sin, np.cos]); Pm[2, :n], Pm[3, :n] = s_[:n], c_[:n]; Pm[7, :n], Pm[8, :n] = s_[n:2 * n], c_[n:2 * n]; Pm[9, :n], Pm[10, :n] = s_[2 * n:], c_[2 * n:]
    Pm[4, :n] = np.clip(interp(conf6, g, [ident])[0], 0, 1.2); s_, c_ = interp(psif, g, [np.sin, np.cos]); Pm[5, :n], Pm[6, :n] = s_, c_
    Pm[0, :n] = interp(ctv, pts - ORG_B, [ident])[0] / 255.0; Pm[1, :n] = interp(spv, pts - ORG_B, [ident])[0] / 255.0
    Pm[11, :n] = interp(cosv, g, [ident])[0] / 255.0; Pm[12, :n] = interp(gmv8, g / 2.0, [ident])[0] / 255.0; Pm[13, :n] = 1.0
    return Pm.astype(np.float16), Lv


P = np.zeros((N, 14, NMAX), np.float16); S = np.zeros((N, 6), np.float32); K = np.zeros(N, np.int16); A = np.zeros((N, 3), np.float64); B = np.zeros((N, 3), np.float64)
V = np.full((N, 16), np.nan, np.float32); ZR = np.zeros(N, np.float32); XC = np.zeros(N, np.float32)
meta = "profiles_truth " + " ".join(sys.argv[1:]) + f" | RAYLEN={RAYLEN} KMAX={KMAX} KUNIF={KUNIF} PERP_L2={PERP_L2} P_ZERO={P_ZERO} JIT={JIT} ANG={ANG} DZ={DZ} NOVOTES={NOVOTES}"


def save(m):
    tmp = OUT[:-4] + ".tmp.npz"                                                                                                        # write next to OUT, then replace, so a killed run never leaves a half-written OUT
    np.savez(tmp, P=P[:m], S=S[:m], k=K[:m], A=A[:m], B=B[:m], votes=V[:m], zrel=ZR[:m], xcol=XC[:m], n_dropped=n_dropped, meta=meta); os.replace(tmp, OUT)


m = 0; tries = 0; n_dropped = 0; n_votes_err = 0; t_prof = 0.0; t_votes = 0.0; t_ray = time.time()
while m < N and tries < 80 * N:
    tries += 1
    z = int(rng.integers(ZLO, ZHI)); y = int(rng.integers(2 + 40, slab.shape[1] - 2 - 40)); x = int(rng.integers(max(XLO, 2 + 40), min(XHI, slab.shape[2] - 2 - 40)))
    if slab[z, y, x] == 255: continue
    cy, cx = np.interp(ZABS + z, uz, uy), np.interp(ZABS + z, uz, ux); gy = ORG_T[1] + y - cy; gx = ORG_T[2] + x - cx; r = np.hypot(gy, gx)
    if r < 300: continue
    th = rng.normal(0, np.radians(ANG)); d = np.array([gy / r * np.cos(th) - gx / r * np.sin(th), gy / r * np.sin(th) + gx / r * np.cos(th)]); steps = np.arange(0, RAYLEN, 0.5); ys = y + d[0] * steps; xs = x + d[1] * steps
    dz = rng.uniform(-DZ, DZ); zs = z + dz * steps
    inside = (ys >= 0) & (ys <= slab.shape[1] - 1) & (xs >= 0) & (xs <= slab.shape[2] - 1) & (zs >= 0) & (zs <= slab.shape[0] - 1); n_in = int(inside.argmin()) if not inside.all() else len(steps)
    if n_in < 100: continue
    ys = ys[:n_in]; xs = xs[:n_in]; zs = zs[:n_in]; steps = steps[:n_in]; lv = slab[np.rint(zs).astype(int), np.rint(ys).astype(int), np.rint(xs).astype(int)]
    bad = np.nonzero(lv == 255)[0]; n_ok = int(bad[0]) if len(bad) else len(lv)
    if n_ok < 60: continue
    lv = lv[:n_ok]; ys = ys[:n_ok]; xs = xs[:n_ok]; zs = zs[:n_ok]; steps = steps[:n_ok]
    u = np.unwrap(lv.astype(np.float64) * 2 * np.pi / 255) / (2 * np.pi); sgn = 1.0 if u[-1] >= u[0] else -1.0; u = u * sgn                  # absolute truth phase in turns, sheet centres at integers
    if np.any(np.diff(u) < -0.12): continue
    ms = np.arange(np.ceil(u[0]), np.floor(u[-1]) + 1)
    if len(ms) < 2: continue
    um = np.maximum.accumulate(u); ix = np.arange(len(u), dtype=float); pos = np.array([np.interp(mm, um, ix) for mm in ms])
    if rng.random() < P_ZERO: k = 0; i0 = int(rng.integers(0, len(ms))); i1 = i0
    elif KUNIF: k = int(rng.integers(1, min(KMAX, len(ms) - 1) + 1)); i0 = int(rng.integers(0, len(ms) - k)); i1 = i0 + k
    else: i0 = int(rng.integers(0, len(ms) - 1)); k = int(rng.integers(1, min(KMAX, len(ms) - 1 - i0) + 1)); i1 = i0 + k
    for _ in range(20 if k == 0 else 1):                                                                                               # k = 0: redraw the jitters until the two endpoints are >= 1 voxel apart
        ends = []; ok = True
        for ii in (i0, i1):                                                                                                            # jitter: target phase m + j, position by linear interpolation of the monotone phase
            q = float(np.interp(ms[ii] + rng.uniform(-JIT, JIT), um, ix)); ph = float(np.interp(q, ix, u)) - ms[ii]
            if abs(ph) >= 0.5: ok = False                                                                                              # the nearest sheet centre would change
            ends.append((q, abs(ph)))
        if k > 0 or abs(ends[1][0] - ends[0][0]) >= 2: break
    if not ok or (k == 0 and abs(ends[1][0] - ends[0][0]) < 2): continue
    pt = lambda q: (np.interp(q, ix, zs), np.interp(q, ix, ys), np.interp(q, ix, xs)); za, ya, xa = pt(ends[0][0]); zb, yb, xb = pt(ends[1][0])
    if not (ZLO <= za < ZHI and XLO <= xa < XHI): continue                                                                             # pair start inside the hold-out window
    a = np.array([ZABS + za, ya + ORG_T[1], xa + ORG_T[2]], float); b = np.array([ZABS + zb, yb + ORG_T[1], xb + ORG_T[2]], float)
    if float(np.linalg.norm(b - a)) < 1.0: continue
    t1 = time.time(); res = profile(a, b); t_prof += time.time() - t1
    if res is None: n_dropped += 1; continue
    t1 = time.time()
    if not NOVOTES:
        try: V[m] = box.votes(a, b)
        except Exception as e: n_votes_err += 1
    t_votes += time.time() - t1
    Pm, Lv = res; P[m] = Pm; rv = np.array([0.0, a[1] - np.interp(a[0], uz, uy), a[2] - np.interp(a[0], uz, ux)]); ra = float(np.linalg.norm(rv)); rv /= max(ra, 1e-6)            # radial direction at a (horizontal)
    S[m] = (Lv / 100, ra / 1000, abs(float(np.dot((b - a) / Lv, rv))), (b[0] - a[0]) / Lv, (a[0] + b[0]) / 2 / 10000, max(ends[0][1], ends[1][1]))
    K[m] = k; A[m] = a; B[m] = b; ZR[m] = za; XC[m] = xa; m += 1
    if m % 500 == 0: print(f"  {m} pairs ({tries} tries, {n_dropped} dropped, {n_votes_err} votes errors), {time.time() - t0:.0f} s, {m / (time.time() - t_ray):.1f} pairs/s (profile {t_prof:.0f} s, votes {t_votes:.0f} s)", flush=True)
    if m % 2000 == 0: save(m)
save(m); print(f"saved {OUT} {m} pairs, {tries} tries, {n_dropped} dropped (L>255), {n_votes_err} votes errors, total {time.time() - t0:.0f} s, {m / max(time.time() - t_ray, 1e-9):.1f} pairs/s (profile {t_prof:.0f} s, votes {t_votes:.0f} s)", flush=True)
