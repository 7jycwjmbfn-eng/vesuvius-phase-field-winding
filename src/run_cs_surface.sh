#!/bin/bash
# surface-level test of a certified-link sync: extraction (gate 0.27, erosion 1) -> no-join SURF + purity -> join with the new pair file -> ruler readings.  usage: run_cs_surface.sh W
W=$1; H=/e/vesuvius_big_hot; P=/e/vesuvius_cs_W${W}; export OMP_NUM_THREADS=4
CONF_FILE=E:/vesuvius_big_hot/pred_v6_conf.npy CONF_MIN=0.27 python bigextract2.py E:/vesuvius_cs_W${W}_u.npy 2 10496 1920 3328 E:/vesuvius_cs_W${W}.pieces.npz 0.02 0.45 ${EXW:-2} ${EXSLAB:-96} > ${P}.ext.log 2>&1
python bigeval.py E:/vesuvius_cs_W${W}.pieces.npz E:/vesuvius_cs_W${W}.nojoin.surf.npz > ${P}.eval_nojoin.log 2>&1
CROP=960,2496,1664,3200 python eval_surfs.py E:/vesuvius_cs_W${W}.nojoin.surf.npz 5248 5504 piece > ${P}.ruler_nojoin.log 2>&1
BOX_ORG=10496,1920,3328 CONFLICT=0.1 python kjoin2.py E:/vesuvius_cs_W${W}.pieces.npz E:/vesuvius_cs_W${W}_u_pairs.npz 5 2 E:/vesuvius_cs_W${W}.join.surf.npz > ${P}.kjoin.log 2>&1
CROP=960,2496,1664,3200 python eval_surfs.py E:/vesuvius_cs_W${W}.join.surf.npz 5248 5504 piece > ${P}.ruler_join.log 2>&1
echo DONE_W${W} > ${P}.done
