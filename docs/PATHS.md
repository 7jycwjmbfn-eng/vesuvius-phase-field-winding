# Hard-coded paths

The scripts were run on one Windows machine and keep their data paths as string literals at the top of the file or in environment variables. Scripts import each other from the same directory (`src/`). Edit the paths before running elsewhere. The human ladders file (`relative_windings.json` of the public spiral dataset) is read from the directory in the environment variable `LADDER_DIR` (default `layerjudge/`).

| path literal | content | used in |
|---|---|---|
| `D:/vesuvius_big_hot/` | box A arrays: CT, surface prediction, phase predictions at L2 | `assemble_big.py`, `certlib.py`, `predict.py`, `predict_v6.py` |
| `D:/vesuvius_big_hot/ct_big.npy` | box A arrays: CT, surface prediction, phase predictions at L2 | `bridge_clf.py`, `plot_autoladder.py`, `truth_pairs.py` |
| `D:/vesuvius_big_hot/pred/frozen8c_psi.npy` | box A arrays: CT, surface prediction, phase predictions at L2 | `bridge_clf.py`, `certlib.py`, `truth_pairs.py` |
| `D:/vesuvius_big_hot/sp_big.npy` | box A arrays: CT, surface prediction, phase predictions at L2 | `bridge_clf.py`, `truth_pairs.py` |
| `E:/vesuvius_big_hot/` | box A phase predictions and sync outputs | `certlib.py`, `ladder_line.py`, `truth_pairs.py` |
| `E:/vesuvius_big_hot/las/cos.npy` | box A phase predictions and sync outputs | `bridge_clf.py` |
| `E:/vesuvius_big_hot/las/gm.npy` | box A phase predictions and sync outputs | `bridge_clf.py` |
| `E:/vesuvius_big_hot/pieces_v6g.npz` | box A phase predictions and sync outputs | `bridge_clf.py` |
| `E:/vesuvius_big_hot/pred_v6_conf.npy` | box A phase predictions and sync outputs | `bridge_clf.py` |
| `E:/vesuvius_big_hot/pred_v6_psi.npy` | box A phase predictions and sync outputs | `bridge_clf.py` |
| `E:/vesuvius_big_hot/u_frozen8c.npy` | box A phase predictions and sync outputs | `bridge_clf.py` |
| `E:/vesuvius_big_hot/u_v6.npy` | box A phase predictions and sync outputs | `rbu_sync.py` |
| `E:/vesuvius_boxB_hot/` | box B arrays | `certlib.py` |
| `E:/vesuvius_counter/run2/model.pt` | counter network files: truth-pair profiles T*.npz, V1, TE*, ladder profiles LAD*.npz, run directories, stack model | `certnet.py` |
| `E:/vesuvius_counter/stack_model.pkl` | counter network files: truth-pair profiles T*.npz, V1, TE*, ladder profiles LAD*.npz, run directories, stack model | `autoladder3.py`, `certnet.py`, `train_stack.py` |
| `E:/vesuvius_counter/stack_pairs_oof.npz` | counter network files: truth-pair profiles T*.npz, V1, TE*, ladder profiles LAD*.npz, run directories, stack model | `stack_ladder_dump.py` |
| `E:/vesuvius_counter_tmp/` | counter network files: truth-pair profiles T*.npz, V1, TE*, ladder profiles LAD*.npz, run directories, stack model | `counter_train.py` |
| `E:/vesuvius_downstream_tmp/fiberA/` | W2 CT at level 3 and the human-verified patch samples used by the ruler | `surfkit_8c.py` |
| `E:/vesuvius_downstream_tmp/fiberA/ctW2_L3.npy` | W2 CT at level 3 and the human-verified patch samples used by the ruler | `render_77f5.py` |
| `E:/vesuvius_downstream_tmp/fiberA/p9comp_w2u.npz` | W2 CT at level 3 and the human-verified patch samples used by the ruler | `eval_wfield_8c.py` |
| `E:/vesuvius_downstream_tmp/fiberA/patch_samples_ua.npz` | W2 CT at level 3 and the human-verified patch samples used by the ruler | `bridge_clf.py`, `patchlabel.py` |
| `E:/vesuvius_downstream_tmp/fiberA/willtruth.npz` | W2 CT at level 3 and the human-verified patch samples used by the ruler | `eval_wfield_8c.py` |
| `E:/vesuvius_impure_detect.npz` | output of impure_detect.py | `impure_detect.py` |
| `E:/vesuvius_inv_tmp/adapt0139/` | scratch | `adapt0139.py` |
| `E:/vesuvius_inv_tmp/bridge_clf/` | scratch | `bridge_clf.py` |
| `E:/vesuvius_ladder2_rows_*.npz` | per-pair vote rows of the human ladders (ladder_run2.py, ladder_run3.py) | `counter_train.py` |
| `E:/vesuvius_ladder2_rows_0.npz` | per-pair vote rows of the human ladders (ladder_run2.py, ladder_run3.py) | `adapt0139.py`, `autoladder2.py`, `autoladder_prod.py`, `certnet.py`, `compare_ladder.py` ... |
| `E:/vesuvius_ladder2_rows_1.npz` | per-pair vote rows of the human ladders (ladder_run2.py, ladder_run3.py) | `adapt0139.py`, `autoladder2.py`, `autoladder_prod.py`, `certnet.py`, `compare_ladder.py` ... |
| `E:/vesuvius_ladder3_rows_*.npz` | per-pair vote rows of the human ladders, second part | `counter_train.py` |
| `E:/vesuvius_ladder3_rows_0.npz` | per-pair vote rows of the human ladders, second part | `adapt0139.py`, `autoladder2.py`, `autoladder_prod.py`, `certnet.py`, `compare_ladder.py` ... |
| `E:/vesuvius_ladder3_rows_1.npz` | per-pair vote rows of the human ladders, second part | `adapt0139.py`, `autoladder2.py`, `autoladder_prod.py`, `certnet.py`, `compare_ladder.py` ... |
| `E:/vesuvius_ladder_regions.json` | boxes around the human ladders (ladder_dl.py) | `certlib.py`, `ladder_dl.py`, `ladder_run.py`, `ladder_run2.py`, `ladder_run3.py` ... |
| `E:/vesuvius_p0139_hot/` | PHerc. 0139 box arrays and truth pairs | `certlib.py`, `truth_pairs.py` |
| `E:/vesuvius_p0139_hot/pred/frozen8c_d2.npy` | PHerc. 0139 box arrays and truth pairs | `p0139_prep.py` |
| `E:/vesuvius_p0139_truthpairs_A.npz` | PHerc. 0139 box arrays and truth pairs | `adapt0139.py` |
| `E:/vesuvius_rbu_sync_out.pkl` | output of rbu_sync.py | `dispute_links.py`, `rbu_sync.py` |
| `E:/vesuvius_truthpairs_A.npz` | truth-pair rows (truth_pairs.py) | `adapt0139.py` |
| `E:/vesuvius_truthpairs_B.npz` | truth-pair rows (truth_pairs.py) | `adapt0139.py` |
| `G:/vesuvius_77f5/p0139/` | dense winding field derived from the official segments (truth) | `eval_wfield_8c.py` |
| `G:/vesuvius_77f5/p0139/centre_w023.json` | dense winding field derived from the official segments (truth) | `adapt0139.py`, `isosurf.py`, `profiles_truth139.py`, `truth_pairs.py` |
| `G:/vesuvius_77f5/p0139/psi_box.npy` | dense winding field derived from the official segments (truth) | `adapt0139.py`, `profiles_truth139.py`, `truth_pairs.py` |
| `G:/vesuvius_77f5/p0139/theta_box.npy` | dense winding field derived from the official segments (truth) | `isosurf.py` |
| `G:/vesuvius_77f5/psi_w2_8cbox_L2_77f5.npy` | dense winding field derived from the official segments (truth) | `profiles_truth.py`, `train_wfield_8c.py` |
| `G:/vesuvius_77f5/theta_w2_L3_raw_77f5.npy` | dense winding field derived from the official segments (truth) | `isosurf.py` |
| `G:/vesuvius_big/` | box A raw chunks and lasagna channels | `assemble_big.py`, `dl_big.py` |
| `G:/vesuvius_big/ct/2/.zarray` | box A raw chunks and lasagna channels | `ladder_run.py` |
| `G:/vesuvius_big/lasagna/` | box A raw chunks and lasagna channels | `dl_lasagna.py` |
| `G:/vesuvius_big/lasagna_oldbox_npy/` | box A raw chunks and lasagna channels | `train_v6.py` |
| `G:/vesuvius_big/sp/0/.zarray` | box A raw chunks and lasagna channels | `ladder_run.py` |
| `G:/vesuvius_boxB_cold/` | box B raw chunks | `certlib.py` |
| `G:/vesuvius_ladder/` | CT, surface prediction and lasagna crops around the human ladders | `ladder_dl.py`, `ladder_run.py`, `ladder_run2.py`, `ladder_run3.py`, `profiles_ladder.py` |
| `G:/vesuvius_p0139_cold/` |  | `certlib.py` |
| `G:/vesuvius_wfield_8c/` | trained models and training data of the phase networks (frozen_8c, v6) | `train_wfield_8c.py` |
| `G:/vesuvius_wfield_8c/iso_pred_{os.path.basename(src)}_{z0}_{y0}_{x0}.npy` | trained models and training data of the phase networks (frozen_8c, v6) | `isosurf.py` |
| `G:/vesuvius_wfield_8c/run2/frozen_8c.pt` | trained models and training data of the phase networks (frozen_8c, v6) | `eval_compare.py`, `ladder_run2.py`, `profiles_ladder.py`, `train_v5.py` |
| `G:/vesuvius_wfield_8c/run2/frozen_8c_pred_p0139_4352_5376.npy` | trained models and training data of the phase networks (frozen_8c, v6) | `p0139_prep.py` |
| `G:/vesuvius_wfield_8c/v4/model_final.pt` | trained models and training data of the phase networks (frozen_8c, v6) | `eval_compare.py`, `ladder_run2.py` |
| `G:/vesuvius_wfield_8c/v6_resenc_8kb/ema_final.pt` | trained models and training data of the phase networks (frozen_8c, v6) | `eval_compare.py`, `ladder_run.py`, `ladder_run2.py`, `ladder_run3.py`, `profiles_ladder.py` |
