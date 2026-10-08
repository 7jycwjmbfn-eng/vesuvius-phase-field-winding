# Phase-field sheet extraction for PHerc. Paris 4

Status: research code and results as of 2026-10-08, published before the end of the Vesuvius Challenge progress prize period (2026-10-31). The readings are development readings; the limits are listed at the end of this file and in `docs/RESULTS.md`. The model weights are on Hugging Face: https://huggingface.co/qizhiran/vesuvius-phase-field-winding (CC BY-NC-SA 4.0).

This code predicts, for every voxel of a scroll CT, the phase of the papyrus sheet it belongs to (0 at the sheet centre, 2 pi at the next sheet outward). Unwrapping that phase gives an integer sheet index, and the level sets of the unwrapped field are surface pieces that follow one sheet each. The output is a set of surfaces in the community tifxyz format, plus flattened CT renderings in which the papyrus fibres are visible.

Scroll and region: PHerc. Paris 4, level 2 (9.6 um) data, region W2 and the boxes around the umbilicus described below.

## What this repository shows, and what it does not

Shown, each with a script and a log in this repository:

- Surfaces: in two held-out boxes the sheet-switch rate is 3.1% and 2.4%, against 6.7% for the Pipeline 9 surfaces in the same crops, and the clean area is larger. The largest single piece is not larger than Pipeline 9's (the preregistered criterion fails). See the first table.
- Winding counts against the community certificate (PCU): on the 3,625 pairs that match between its published per-pair rows and the human ladders, the stack of the counter network and the gradient-boosting certifier has 16.1 (one wrap), 14.8 (two wraps) and 18.8 (three wraps) points more coverage at the 99% target, with similar precision for one and two wraps. In the band z2 11000-12000, where the PCU report finds its certificate weakest, one-wrap coverage is 80.5% at 99.4% against 51.0% at 98.0%; below 96 um wrap spacing it is 84.4% against 35.6% (45 pairs). `docs/RESULTS.md`, sections "Same-pair comparison" and "Where the PCU report lists failures".
- Generated ladders in the official spiral fit (box A window, two seeds, exploratory): a filtered set of 14,364 generated ladders lowers the slips per wrap on 8 held-out human ladders (41 pairs) by 13.6 points (95% interval -23.8 to -5.8) against no constraints; the unfiltered set of 18,914 ladders changes nothing (+2.7, -5.9 to +11.0).

Not shown:

- No improvement of the official fit is established. The preregistered control (a few human ladders lowering the slips) did not pass in this window, the filter thresholds were chosen after looking at agreement with the label field inside the truth slab, there are 41 held-out pairs and one window, and nothing was run in the PCU bands.
- The stack does not help inside the automatic ladder generator: at equal link precision the gradient-boosting certifier generates as many ladders.
- The stack certifies a two-wrap pair with fewer wraps than the truth more often than PCU does (9 of 765 certified pairs against 1 of 585), spans of more than four wraps are not certified at 99%, and PHerc. 0139 is undecided because its label field may be missing sheets.
- Retraining the counter network with wider rays (120,000 more pairs) and a missed-sheet augmentation lowered the coverage on the human ladders (60.4% and 53.8% against 65.3% at the 99% target) and left the undercounts unchanged; the current model was kept (`docs/RESULTS.md`).
- Speed and scale: PCU runs on whole slices at about 6 s per slice. This code needs the phase network and the extraction of CT, surface prediction and lasagna arrays, which exists only for box A, the PHerc. 0139 box and the neighbourhoods of the human ladders.

The dated lab notes with the rules written before each reading are in `docs/lab_notes/PREREG_bigbox_cn.md` (Chinese). Hard-coded data paths are listed in `docs/PATHS.md`.

## Result in one table

Same crops and same window as the Pipeline 9 surfaces of Will Stevens, measured with the human-verified patch ruler in `src/surfkit_8c.py` (window L3 z 5248-5504, i.e. 4.9 mm). Sheet switch = share of on-patch samples that have a stretch where the surface runs on the neighbouring sheet. Areas are in cm2.

| | switch rate | total area | largest single piece | largest clean contiguous | clean total |
|---|---|---|---|---|---|
| Box A, this method (model v6 + confidence gate) | 2.2% (67/3085) | 51.20 | 1.4443 | 0.89 | 44.90 |
| Box A, Pipeline 9 | 6.7% | 46.13 | 1.4442 | 0.71 | 31.19 |
| Box B, this method | 1.3% (48/3733) | 32.01 | 0.7936 | 0.40 | 26.27 |
| Box B, Pipeline 9 | 6.7% | 19.71 | 0.8158 | 0.34 | 13.68 |

Reading the table:

- The switch rate is 3 times (box A) and 5 times (box B) lower than for Pipeline 9. The clean area is 1.4 times and 1.9 times larger.
- The largest single piece is the same as Pipeline 9 in box A (1.4443 and 1.4442) and 3% smaller in box B. In box A both methods reach the geometric limit of the crop and window, so this number does not separate them. Pipeline 9 has the larger connected arrays (2.02 cm2 in box A, 1.20 cm2 in box B, against 1.44 and 0.79 here).
- Pipeline 9 surfaces are used only as the comparison. They are not an input of this method. The Pipeline 9 switch rate is the same 265/3961 in both boxes because all its on-patch samples lie in the overlap of the two crops (recomputed with the ruler on 2026-10-08, both crops).
- The two boxes are not independent replications, and part of the ruler region lies in the columns the network was trained on. The box A ruler region (L2 y 1920-3584, x 3328-4864) lies inside the box B ruler region, so both boxes share it. About 64% of the area of the box A ruler region and about 22% of the box B ruler region lie inside the W2 columns (y 2304-3840, x 3584-5120); there the evaluation is held out in z only (training z 10000-10380, evaluation from z 10496). The confidence gate 0.27 was chosen on a development slab (L2 z 10500-10750, W2 columns), which is the lower half of the box A window and holds about half of the box A on-patch samples (1526 of 3085; the first count is without joins). The percentages are computed from the box coordinates and are approximate.

Figures for these boxes are in `results/figures`: paired cross-sections (ours left, Pipeline 9 right) and flattened CT of the largest groups, where horizontal and vertical fibres can be seen.

More detail, including the readings of the two other model variants and the experiments that did not work, is in `docs/RESULTS.md`.

## Winding-count certificates and automatic ladders

The same phase field also gives the number of sheets between two points: unwrap the predicted phase along the line between them. Counting along one line is right for about 84% of the one-wrap pairs of the community winding ladders (256 collections, 17,425 pairs, all z bands). Three certifiers were built on top of that (`docs/RESULTS.md` has the tables and the caveats). They are scored on the 5,943 pairs of one to four windings (227 collections) that remain after excluding the network's training z range and the ladder regions that touch the W2 truth box. The counter network reads profiles of up to 255 voxels, and all 5,943 pairs are within that length. The 10,787 pairs of five or more windings are not scored in the numbers below; `docs/RESULTS.md` gives a precision that includes them.

1. A gradient-boosting certifier on several separately computed counts (the phase network with flipped inputs, the first model, parallel lines, minimum-cost paths; most come from the output of one network, so they are partly correlated) and the official lasagna `grad_mag`, cos and the surface prediction, trained on the human ladders. At an overall precision target of 99% it accepts 38.1% of the pairs at 99.03% precision (5-fold by collection; the threshold of each fold comes from the other four folds, and one threshold on the pooled predictions gives 38.8%).
2. A counter network (`src/counter_net.py`): a 1D convolutional network that reads the whole 14-channel profile of the segment between the two points (CT, surface prediction, v6 and frozen_8c phase, side lines, lasagna) and outputs the count. The surface prediction and the CT are also inputs of v6. It is trained on 250,000 pairs from the dense truth of the W2 slab and on no human ladder. On test pairs from the second half of the W2 slab, at the 99% target (thresholds from a validation set), it accepts 93.4% of the pairs at 99.28% precision; the gradient-boosting certifier, trained on about 43,000 pairs and seeing only summary votes, accepts 81.6% at 99.14%. The slab is held out in z only, and the z ranges of training and test rays can overlap because the rays have a z slope.
3. A stack of the two (`src/certnet.py`). On the human ladders, with the same 5,943 pairs and the same fold protocol as 1, at the 99% target it accepts 65.3% of the pairs (bootstrap interval over collections 62.0 to 68.8) at 98.87% precision; one wrap 85.6% at 99.7%, two wraps 61.2% at 97.9%, three wraps 52.2% at 97.8%, four wraps 55.0% at 99.5%. The target is 99% and the realized precision is about 98.8%. These precisions count pairs whose true count is 1 to 4. When the 4,200 pairs of five or more windings that have a profile are counted as well and a certified count must be 1 to 4, the precision is 97.88% (40 of those 4,200 pairs are certified as 1 to 4). Over six random assignments of collections to folds the coverage at the 99% target averages 61.9% (range 56.0 to 66.3, standard deviation 4.2; mean precision 98.76%), and with folds made of spatially separated clusters of ladder regions it is 60.9% at 98.51%. The bootstrap interval covers the sampling of collections only. In the style of the community winding-count certificate report (PCU, `docs/PCU_REPORT.md` of the Jashann/vesuvius-scrolling repository): one-wrap pairs 85.6% coverage at 99.68% precision (5 errors in 1549), 85.7% at 100% in the band L2 z 8400-9400 and 79.8% at 99.49% in the band 11000-12000. These are development readings. Two variants of the counter network were read on the ladders (the one used and its fine-tuned version), and the stack and its thresholds come from the ladder folds. The first variant, trained on the easier mix, was not read on the ladders during development; read afterwards it gives 53.2% at 98.86%, so the larger and harder training mix is worth about 12 points of coverage.

The PCU report states 99.62% at 70% coverage on 1,502 one-wrap pairs of Paris 4, 98.58% at 48% on two-wrap pairs and 94.89% at 36% on three-wrap pairs; these figures are the report's own. A pair-by-pair comparison on the 3,625 pairs that match between its published per-pair rows and our ladder pairs (91.4% of its pairs) is in `docs/RESULTS.md`; there the PCU rule reproduces its reported figures and the stack at the 99% target has 16.1 (one wrap), 14.8 (two wraps) and 18.8 (three wraps) points more coverage, at similar precision for one and two wraps and 3.0 points higher precision for three wraps. On one-wrap pairs the gradient-boosting certifier is close to them and the stack has 85.6% coverage at 99.68% (overall target 99%, realized 98.87%, one threshold for one to four windings). On two-wrap pairs PCU has 48% coverage at 98.58% and the stack 61.2% at 97.9%; the coverages differ, so the precisions cannot be compared directly. On three-wrap pairs the gradient-boosting certifier reaches 36.4% coverage at 96.2% (98% target), above PCU at the same coverage (36% at 94.89%), and the stack 52.2% at 97.8%. The ladders may contain annotation errors, which were not checked by hand, so the precisions are measured against the human ladders as they are.

- Truth pairs and the label field: on 4000 pairs drawn from the label field of the held-out slab (certifier and threshold fixed on the ladders), the gradient-boosting certifier accepts 45.5% at 99.67%. The slab is held out in z only, and the label field and the turn-number field built from the same official segments disagree on about 5% of the links, so these numbers measure agreement with the label field.
- Automatic ladders from predictions only (`src/autoladder2.py`, `src/autoladder3.py`; output in the community point-collection format), scored on 1000 lines through the held-out slab against the label field. The primary tolerance is 0.15 turn around the label-field sheet centre. With the stack as the certifier, 73.6%, 81.4% and 88.6% of the accepted links (3161, 2707 and 2104) have both end points within 0.15 turn and the right count at PCONF 0.6, 0.7 and 0.8. At the loose tolerance of 0.30 turn the figures are 98.5%, 98.6% and 99.0%, and 99.15% to 99.29% of the links have a count equal to the turn-number difference. At the default PCONF 0.8, 43.3% of the truth sheets crossed have a ladder point within 0.30 turn in a chain of at least three points; at PCONF 0.6 the share is 65.6% (chains with a wrong link are included, and overlapping lines count a sheet more than once). The gradient-boosting generator on the same lines gives close results at the same PCONF: at its 98% threshold 44.0% of the sheets at 98.74% link precision (PCONF 0.8) and 67.3% at 98.17% (PCONF 0.6), against 43.3% at 98.95% and 65.6% at 98.51% for the stack. The stack therefore gives no gain over the gradient-boosting generator on the automatic ladders; its gain is on held-out truth pairs and on human ladder pairs (longer pairs, counts of two to four windings). The PCONF levels of the stack were set after the first readings of the gradient-boosting generator (0.7, 0.8, 0.9), which was run at 0.6 afterwards. The scored lines select positions where the label field is valid and monotone (radius at least 300 voxels); on an unfiltered ray grid the certified count equals the turn-number difference for 95.1% of the steps. Counting every accepted link, whatever the distance of its end points from the sheet centres, 0.8% to 0.95% of the stack's links have a wrong count. Only one 500-layer slab of one scroll and radial directions were tested.
- Production ladders: `src/autoladder_prod3.py` (counter network, gradient-boosting certifier and stack; model trained without any human ladder region that touches box A; PCONF 0.7) wrote `results/auto_ladders_boxA_all.json` (18,914 ladders, 92,862 points, three seeds of 15,000 lines) and, after filtering by link probability >= 0.999, v6 confidence >= 0.85 at both ends and count <= 2 (`src/strict_arm_C_ladders.py`), `results/auto_ladders_boxA_strict.json` (14,364 ladders, 43,499 points). Inside the truth slab (z 10500-11000) the count agrees with the label field for 93.2% of 21,058 scorable links in the first file and 97.3% of 9,468 links in the second (`src/score_ladders_vs_label.py`). No human point is an input when a ladder is built.
- PHerc. 0139 (crushed scroll, another scan; no registered lasagna): against the label field of that box, the certifiers without lasagna votes reach 92.3% (gradient boosting, preregistered line 97%) and 86.3% (counter network). The 263 certification errors of the gradient-boosting certifier are all overcounts; of the 1379 errors of the counter network, one is an undercount. The label field comes from 37 official segments and may be missing sheets, so this reading does not decide whether the certificates hold on this scroll. On the disputed pairs the surface-prediction crossings and the CT peak counts agree with the network count more often than with the label count, but both are inputs of v6 and of the counter network, so they are not independent evidence. An overcount by networks trained on Paris 4 on a scroll with a different sheet spacing would give the same pattern. A count of a random sample of the disputed pairs by someone who does not see the labels is needed.
- Not working: feeding the certified counts into the tile sync (labels along ladders improved, surfaces worse on the ruler), a ray-grid synchronisation of the certified ladders (local alignment 97% right, global labels 59% against 63% for the tile sync), fine-tuning the counter network on the ladders (for the stack, coverage at the 98% target rose from 81.3% to 84.3% and at the 99% target fell from 65.3% to 56.3%; the counter alone is lower at every target; the version trained on truth pairs only is used), a detector of impure pieces from certified links (recall 6% to 11%).

```
python src/ladder_dl.py ; python src/ladder_run2.py OUT2.npz 0 128 ; python src/ladder_run3.py OUT3.npz 0 128      # votes for every ladder pair (two halves of the 256 collections each)
python src/ladder_tables.py                                                                                          # gradient-boosting certifier tables (in-fold slopes, bootstrap over collections)
python src/truth_pairs.py PAIRS.npz 4000 100 ; python src/truth_certify.py PAIRS.npz -- ROWS2... -- ROWS3...       # truth-pair check of the gradient-boosting certifier
KMAX=10 RAYLEN=260 KUNIF=1 python src/profiles_truth.py T1.npz 25000 11 0 250 0 1100                               # profile pairs for the counter (more shards, JIT=0.35 ANG=45 for the harder ones)
python src/counter_train.py --train T1.npz ... --val TE2.npz --test TE1.npz --out run2 --epochs 14 --bs 512 --ls 0.02  # counter network
python src/profiles_ladder.py LAD0.npz 0 128 ; python src/profiles_ladder.py LAD1.npz 128 256                       # ladder profiles (GPU)
python src/counter_train.py --eval-only --out run2 --val TE2.npz --test TE1.npz --ladder LAD0.npz LAD1.npz           # counter predictions on the ladders
python src/stack_ladder.py run2/pred_LAD0.npz run2/pred_LAD1.npz ; python src/train_stack.py run2/pred_LAD0.npz run2/pred_LAD1.npz
PCONF=0.6 python src/autoladder3.py OUT.pkl 1000 99 250 500 ; python src/autoladder2_eval.py OUT.pkl                  # automatic ladders with the stack (scored against the truth)
python src/autoladder_prod.py LADDERS.json 5000 1                                                                    # ladders in box A with the gradient-boosting certifier (source of the sample in results/, not scored)
```

Intermediate files that these commands read and are not in `results/`: the ladder rows (`ladder_run2` and `ladder_run3` outputs, one npz per half), `ladder_regions.json` (written by `ladder_dl.py`), the 77f5 label field and the box A arrays listed in `docs/PATHS.md`. The ladder-profile run takes about 55 minutes per half on the GPU.

## Method

1. Inputs: CT (level 2), the official surface prediction (`surface-m7-L2-th0.2`), the official lasagna channels (nx, ny, grad_mag at L2/4, cos at L2/2) and the umbilicus table.
2. Phase network (`src/train_v6.py`, `src/archs.py`): a 3D U-Net with residual encoder blocks (channels 24, 48, 96, 192, 288, GroupNorm, SiLU), 8 input channels (CT, surface prediction, radial unit vector from the umbilicus, lasagna nx, ny, grad_mag, cos), 2 output channels (sin and cos of the phase). Trained for 8000 steps with EMA weights on L2 z 10000-10380 of one scroll, with phase labels derived from the official segments. Augmentation includes obscuration blobs (blur, blank, no surface prediction) and dropping the lasagna group.
3. Inference (`src/predict_v6.py`): 96^3 tiles with 16 voxels of overlap, 2x down-sampling by vector averaging. The length of the predicted (sin, cos) vector is used as a confidence value.
4. Unwrapping (`src/bigsync2.py`): the volume is cut into tiles (48 voxels, step 24); each tile is unwrapped with scikit-image; integer offsets between tiles come from an L1 network problem (HiGHS with a time limit, then IRLS and integer coordinate descent).
5. Surface extraction (`src/bigextract2.py`): for every integer layer k, voxels with |u - k| < 0.45 and confidence >= 0.27 are kept, eroded once, and marching cubes is run on u - k. Connected components of at least 0.02 cm2 are pieces. The volume is processed in z slabs and the slab meshes are welded, so memory does not grow with the height.
6. Joining (`src/kjoin2.py`): two pieces are joined when their ends meet in neighbouring tile cores within 5 voxels, their layer difference equals the integer offset the network assigned between those tiles, and they do not overlap as stacked sheets (more than 10% of overlapping cells with a radius difference above 4 voxels rejects the join). No ground truth is used by this step.
7. Output (`src/piece2surf.py`, `src/surfkit_8c.py tifxyz`): a regular grid per group, written as tifxyz. The official lasagna `fit.py` with `configs/flatten_fast.json` flattens a group, and `src/render_77f5.py` samples the CT along the normal to make the fibre images.

The confidence threshold 0.27 was chosen on a development slab (L2 z 10500-10750) that the network was not trained on, with the human-patch ruler. The slab lies in the W2 columns, which are held out in z only, and it is the lower half of the box A evaluation window (see the note under the first table). It removes about 6% of the voxels and lowered the switch rate there from 2.0% to 0.5% while keeping 93% of the clean area.

## What was not good

- Taller boxes. Extending box A downward by 512 layers (L2 z 9984-11008) raises the largest level-set piece from 0.84 to 1.36 cm2 and the largest joined group from 1.33 to 1.62 cm2. In the upper 256-layer window the largest piece is then smaller (0.80 against 1.44) and the sheet-switch rate is 6.2% (box A alone: 2.2%). Over the lower 256 layers it is 11.8%. Pipeline 9 has 6.7% and 10.6% in the same windows. The lower half is not better than Pipeline 9. Only the 512-layer boxes are reported as results.
- Model choice. The phase accuracy of the models is almost the same (88.8% to 89.0% on the development slab). The confidence gate moves the switch rate; the architecture changes it little. Averaging the old 4-channel model with v6 gave 4.6% (box A) and 3.0% (box B) and was not adopted.
- Removing the erosion step and relying on the confidence gate instead gives 5.5% to 22% switch rate on the development slab, so the erosion stays.
- Cutting thin bridges on the mesh (morphological opening) lost 14% of the clean area for a small gain in purity.
- The comparison readings were taken three times on boxes A and B (first model, ensemble, final model). The threshold was fixed on the development slab before the last reading, but the boxes are no longer untouched.

## Data

The input files are the official open data of the Vesuvius Challenge (CT, surface prediction, lasagna, umbilicus). The truth used for training and for the purity numbers in `docs/RESULTS.md` is a dense winding field derived from the official segments; the human-verified patch samples used by the ruler are listed in `src/surfkit_8c.py` (`patch_samples_ua.npz`). Provenance of both files still has to be written down before release.

## How to run

All data paths are set at the top of the scripts. `docs/PATHS.md` lists them. Order of the steps:

```
python src/dl_big.py both 24                      # CT and surface prediction chunks for a box (env BOX_DIR, BOX_RANGE)
python src/dl_lasagna.py                          # lasagna chunks
python src/assemble_big.py                        # chunks -> ct_big.npy, sp_big.npy
python src/assemble_las.py LAS_DIR OUT_DIR Z0 Z1 Y0 Y1 X0 X1
BOX=big DOWN=2 BOX_HOT=... BOX_ORG=... LAS_DIR=... python src/predict_v6.py MODEL.pt resenc Z0 Z1 OUT_PREFIX
python src/bigsync2.py PSI.npy 1 U.npy 48 24 8 240
CONF_FILE=CONF.npy CONF_MIN=0.27 python src/bigextract2.py U.npy 2 OZ OY OX PIECES.npz 0.02 0.45 2 96
BOX_ORG=OZ,OY,OX CONFLICT=0.1 python src/kjoin2.py PIECES.npz U_pairs.npz 5 2 SURF.npz
python src/surfkit_8c.py tifxyz SURF.npz OUT_DIR
CROP=y0,y1,x0,x1 python src/eval_surfs.py SURF.npz 5248 5504 piece
```

The scripts are research code as they were run. They need Python 3.12, PyTorch with CUDA, NumPy, SciPy, scikit-image, zarr, numba, tifffile and matplotlib. The full chain for one 512-layer box of 3072 x 3072 voxels takes about 35 minutes of GPU for inference and about 30 to 40 minutes of CPU for the rest, on one RTX 4080 laptop GPU and 32 threads. Memory commit is about 12 GB for inference and about 10 GB for the tile sync with 6 worker processes.

## Limitations

- One scroll, one training region (380 layers). On PHerc. 0139, an earlier 4-channel model reached 83.6% phase accuracy and 92.6% area-weighted purity on the first 512 layers; v6 without lasagna reached 83.9% phase accuracy there, and the winding-count certifier reaches 92.3% against the label field of that box, below the preregistered 97% line; the label field may be missing sheets, so this reading does not decide whether the certificates hold on that scroll (see above).
- Phase defects (vortex lines) limit how large a piece can become. About 4% of the voxels of the development slab touch a defect, and 84% of the mesh boundaries are free ends in regions where the gradient of the unwrapped phase is twice the average.
- Lower z (L2 9984-10496) is harder: the local switch rate is 3% to 8% per 64-layer slab, against 0.7% to 1.6% above L2 10496.
- The ruler covers L3 y 576-1792 and x 896-2432 only, which is a part of each box. The ruler regions of the two boxes overlap, and part of them lies in the W2 columns (see the note under the first table).
- Winding-count certificates: the readings are development readings against the label field, on one scroll. The readings of the automatic ladders were made with a version of `src/certnet.py` whose grad_mag slopes were fitted on all ladder pairs and whose cos peak count was not made absolute; both are fixed in the code now (at the 99% threshold the stack's own training slopes give 66.0% coverage at 99.01% precision and the old slopes 67.8% at 98.91%). The baseline gradient-boosting model of the counter comparison was trained on 42,989 pairs and stops early (189 of 250 iterations), which the certifier on the ladders (5.9 thousand rows) does not. See `docs/RESULTS.md`.

## License

Code and documentation: MIT (`LICENSE`). The model weights (`frozen_8c.pt`, `v6_resenc_ema_final.pt`, the counter network and the stack) are not in this repository; they are at https://huggingface.co/qizhiran/vesuvius-phase-field-winding under CC BY-NC-SA 4.0 because they are trained on the Vesuvius Challenge open data, which is distributed by default under CC BY-NC 4.0. No file of the PCU repository is redistributed here: `results/pcu_*.log` contain only numbers computed from its published per-pair rows.
