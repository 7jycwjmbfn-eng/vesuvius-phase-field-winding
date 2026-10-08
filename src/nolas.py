"""Feature matrix of the winding-count certifier WITHOUT any lasagna vote (no grad_mag, no cos), for volumes that have no registered lasagna (PHerc0139).
Input D: dict of certlib.rows_to_dict (keys truth, V, gm1, gm2, cos, sp1, sp2, ct, minc, meanc, pmin, L); gm1, gm2 and cos are never read.
  V columns: [v6, frozen, +3 line, -3 line, path (1/conf^2), path (1/conf^4)]
  cand = mode of the six rounded |phase votes| (NaN vote = -1 = invalid); a tie is broken toward the v6 vote, and if v6 is not in the tie, toward the tied value closest to v6 (then the smaller)
  features_nolas(D) -> (X, cand), X columns:
    0 votes equal to cand, 1 valid votes, 2-7 vote j == cand (6 booleans), 8 mean |V - cand| over valid votes (unrounded V),
    9 sp1 == cand, 10 sp2 == cand, 11 ct == cand, 12 |ct - cand|, 13 minc, 14 meanc, 15 pmin (NaN -> -1), 16 log1p(L), 17 cand"""
import numpy as np

NAMES = ["n_eq_cand", "n_valid", "v6_eq", "frozen_eq", "p3_eq", "m3_eq", "path2_eq", "path4_eq", "mean_absdiff", "sp1_eq", "sp2_eq", "ct_eq", "ct_absdiff", "minc", "meanc", "pmin", "log1p_L", "cand"]


def phase_mode(Vabs):
    """Vabs (n, 6) float with NaN for missing votes. Returns (cand int (n,), Vi int (n, 6) with -1 for invalid, counts (n, 6))."""
    ok = np.isfinite(Vabs)
    Vi = np.where(ok, np.rint(np.where(ok, Vabs, 0.0)), -1).astype(np.int64)
    eq = (Vi[:, :, None] == Vi[:, None, :]) & ok[:, :, None] & ok[:, None, :]                       # eq[i, a, b]: vote a and vote b valid and equal
    counts = eq.sum(2)                                                                              # votes sharing the value of vote a (0 if vote a invalid)
    m = counts.max(1)
    tied = (counts == m[:, None]) & ok                                                              # votes whose value is a mode
    dist = np.abs(Vi - Vi[:, :1])                                                                   # distance to the v6 value (v6 first column)
    big = 10 ** 6
    key = np.where(tied, dist * big + np.clip(Vi, 0, big - 1), 10 ** 12)                            # smallest distance to v6, then smaller value
    pick = key.argmin(1)
    cand = np.where(ok.any(1), Vi[np.arange(len(Vi)), pick], 0)
    return cand, Vi, counts


def features_nolas(D):
    Vabs = np.abs(np.asarray(D["V"], dtype=float))
    cand, Vi, _ = phase_mode(Vabs)
    ok = Vi >= 0
    eqc = (Vi == cand[:, None]) & ok
    diff = np.abs(np.where(ok, Vabs, 0.0) - cand[:, None]); nvalid = ok.sum(1)
    mean_diff = np.where(nvalid > 0, (diff * ok).sum(1) / np.maximum(nvalid, 1), -1.0)
    ct = np.asarray(D["ct"], dtype=float)
    X = np.c_[eqc.sum(1), nvalid, eqc.astype(int), mean_diff,
              np.asarray(D["sp1"]) == cand, np.asarray(D["sp2"]) == cand, ct == cand, np.nan_to_num(np.abs(ct - cand), nan=-1),
              np.nan_to_num(np.asarray(D["minc"], dtype=float), nan=-1), np.nan_to_num(np.asarray(D["meanc"], dtype=float), nan=-1), np.nan_to_num(np.asarray(D["pmin"], dtype=float), nan=-1),
              np.log1p(np.asarray(D["L"], dtype=float)), cand].astype(np.float64)
    return X, cand
