"""Test whether long automatic fibres can join surface pieces: a fibre that touches two pieces says they are the same sheet (a fibre never leaves its sheet).
Piece vertices (L2 zyx, bigextract2 npz) go into a kd-tree; every fibre point (L3 zyx x 2 -> L2) within TOL voxels of a vertex is assigned to that piece; a piece is "touched" by a
fibre when it gets >= MINPTS points.  For every pair of pieces touched by one fibre the truth turn numbers at the contact points decide whether the link is a true continuation
(seam-aware, as kjoin2).  Precision is reported by the fibre arc length between the two contacts (the gap that the fibre bridges) and by the number of supporting fibres.
  python fibre_links.py PIECES.npz FIBRES.npz OZ OY OX SZ SY SX [TOL] [MINPTS]      (box origin and size in L2 voxels)"""
import sys, time
import numpy as np
from scipy.spatial import cKDTree
import isosurf as iso

pf, ff = sys.argv[1], sys.argv[2]; org = np.array([float(v) for v in sys.argv[3:6]]); size = np.array([float(v) for v in sys.argv[6:9]])
TOL = float(sys.argv[9]) if len(sys.argv) > 9 else 3.0; MINPTS = int(sys.argv[10]) if len(sys.argv) > 10 else 3
t0 = time.time(); d = np.load(pf); V = d["V"]; voff = d["voff"]; n = len(voff) - 1 if len(voff) > len(d["k"]) else len(d["k"]); piece = np.repeat(np.arange(len(d["k"])), np.diff(np.r_[voff, len(V)]) if len(voff) == len(d["k"]) else np.diff(voff))
sel = np.arange(0, len(V), 3); tree = cKDTree(V[sel].astype(np.float64)); ps = piece[sel]; print(f"{len(d['k'])} pieces, tree on {len(sel)} vertices, {time.time() - t0:.0f} s", flush=True)
f = np.load(ff); P = f["pts"].astype(np.float64) * 2.0; off = f["off"]; fid = np.repeat(np.arange(len(off) - 1), np.diff(off))
inb = np.all((P >= org) & (P < org + size), 1); keep = inb & (np.arange(len(P)) % 2 == 0); Pk = P[keep]; fk = fid[keep]; pos = np.nonzero(keep)[0]
print(f"fibres {len(off) - 1}, fibre points inside the box (every 2nd) {len(Pk)}, {time.time() - t0:.0f} s", flush=True)
dist, idx = tree.query(Pk, distance_upper_bound=TOL, workers=-1); ok = np.isfinite(dist); pid = np.full(len(Pk), -1); pid[ok] = ps[idx[ok]]; vtx = np.full((len(Pk), 3), np.nan); vtx[ok] = V[sel][idx[ok]]
# per (fibre, piece): number of points, mean contact position, mean position index along the fibre
m = pid >= 0; key = fk[m].astype(np.int64) * (len(d["k"]) + 1) + pid[m]; uk, inv = np.unique(key, return_inverse=True); cnt = np.bincount(inv)
fb = uk // (len(d["k"]) + 1); pc = uk % (len(d["k"]) + 1); cpos = np.zeros((len(uk), 3)); np.add.at(cpos, inv, vtx[m]); cpos /= cnt[:, None]
arc = np.zeros(len(uk)); np.add.at(arc, inv, pos[m].astype(np.float64)); arc /= cnt                                                    # mean point index along the fibre
good = cnt >= MINPTS; fb, pc, cpos, arc, cnt = fb[good], pc[good], cpos[good], arc[good], cnt[good]
print(f"(fibre, piece) contacts with >= {MINPTS} points: {len(fb)}, fibres touching >= 2 pieces: {int((np.bincount(fb)[np.bincount(fb) > 1]).size)}, {time.time() - t0:.0f} s", flush=True)
# all pairs of contacts of one fibre
order = np.argsort(fb, kind="stable"); fb, pc, cpos, arc = fb[order], pc[order], cpos[order], arc[order]; bnd = np.r_[0, np.nonzero(np.diff(fb))[0] + 1, len(fb)]
A = []; B = []
for a0, a1 in zip(bnd[:-1], bnd[1:]):
    if a1 - a0 < 2: continue
    ii, jj = np.triu_indices(a1 - a0, 1); ii += a0; jj += a0; sel2 = pc[ii] != pc[jj]; A.append(ii[sel2]); B.append(jj[sel2])
A = np.concatenate(A); B = np.concatenate(B); print(f"contact pairs {len(A)}, {time.time() - t0:.0f} s", flush=True)
nA = iso.turn_numbers(cpos[A]); nB = iso.turn_numbers(cpos[B]); tA = iso.theta_wrapped(cpos[A]); tB = iso.theta_wrapped(cpos[B])
have = np.isfinite(nA) & np.isfinite(nB); true = np.zeros(len(A), bool); true[have] = (nB[have] - nA[have]) == -np.rint((tB[have] - tA[have]) / (2 * np.pi))
gap = np.abs(arc[A] - arc[B]) * 2.0 * 2.0                                                                                   # point index (every 2nd point, ~1 L3 step each) -> L2 voxels, rough
lo = np.minimum(pc[A], pc[B]); hi = np.maximum(pc[A], pc[B]); pairkey = lo.astype(np.int64) * (len(d["k"]) + 1) + hi
print(f"\ncontact pairs with truth at both ends: {int(have.sum())} of {len(A)}; precision (true continuation) {true[have].mean() * 100:.1f}%")
for lo_, hi_ in ((0, 10), (10, 30), (30, 100), (100, 300), (300, 1e9)):
    s = have & (gap >= lo_) & (gap < hi_)
    if s.any(): print(f"  fibre gap {lo_:>4.0f}-{hi_:<7.0f} voxels: pairs {int(s.sum()):8d}, unique piece pairs {len(np.unique(pairkey[s])):7d}, precision {true[s].mean() * 100:5.1f}%")
# piece pairs: supported by >= S fibres, precision of the unique pairs
u, inv2 = np.unique(pairkey[have], return_inverse=True); sup = np.bincount(inv2); tr = np.bincount(inv2, true[have]) / sup; gmin = np.full(len(u), 1e9); np.minimum.at(gmin, inv2, gap[have])
for S in (1, 2, 3, 5, 10):
    s = sup >= S; far = s & (gmin >= 10)
    print(f"  piece pairs with >= {S:2d} supporting fibres: {int(s.sum()):6d}, pair-level true share {np.mean(tr[s] >= 0.5) * 100:5.1f}%; of them with a gap >= 10 voxels: {int(far.sum()):6d}, true share {np.mean(tr[far] >= 0.5) * 100 if far.any() else float('nan'):5.1f}%")
kk = d["k"]; dk = np.abs(kk[pc[A]] - kk[pc[B]])
for name, sel_ in (("same label (dk = 0)", dk == 0), ("dk = 1", dk == 1), ("dk >= 2", dk >= 2)):
    s_ = have & sel_
    if s_.any(): print(f"  {name}: pairs {int(s_.sum()):6d}, unique piece pairs {len(np.unique(pairkey[s_])):6d}, precision {true[s_].mean() * 100:5.1f}%")
    for lo_, hi_ in ((10, 100), (100, 1e9)):
        s2 = s_ & (gap >= lo_) & (gap < hi_)
        if s2.any(): print(f"      gap {lo_:.0f}-{hi_:.0f}: pairs {int(s2.sum()):6d}, precision {true[s2].mean() * 100:5.1f}%")
np.savez_compressed(sys.argv[1].replace(".npz", "_fibrelinks.npz"), u=u, sup=sup, tr=tr, gmin=gmin); print("saved", time.time() - t0)
