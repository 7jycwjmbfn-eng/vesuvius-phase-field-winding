#!/bin/bash
# fine-tune the counter on the training folds of the human ladders (init from run2), one run per held-out fold
export OMP_NUM_THREADS=2
for f in 0 1 2 3 4; do
  python counter_train.py --train E:/vesuvius_counter/T3.npz E:/vesuvius_counter/folds/LADtrain_$f.npz E:/vesuvius_counter/folds/LADtrain_$f.npz E:/vesuvius_counter/folds/LADtrain_$f.npz E:/vesuvius_counter/folds/LADtrain_$f.npz E:/vesuvius_counter/folds/LADtrain_$f.npz E:/vesuvius_counter/folds/LADtrain_$f.npz \
     --val E:/vesuvius_counter/TE2.npz --test E:/vesuvius_counter/TE1.npz --ladder E:/vesuvius_counter/folds/LADtest_$f.npz --out E:/vesuvius_counter/ft_$f \
     --init E:/vesuvius_counter/run2/model.pt --epochs 6 --lr 3e-4 --bs 512 --keep-last --no-gbm --patience 99 > E:/vesuvius_counter/ft_$f.log 2>&1
done
echo FT_DONE > E:/vesuvius_counter/ft_done.txt
