# Hierarchical D-FINE-L

This branch replaces the proposal-crop stack with one end-to-end detector.

## Architecture

Each decoder query has three factorized outputs:

- `pred_boxes`: class-agnostic localization;
- `pred_objectness_logits`: defect versus background/localization quality;
- `pred_class_logits`: nine conditional defect classes.

Hungarian matching uses only defectness, L1 box distance, and GIoU. Category
logits cannot prevent a weak class from receiving a localization match. The
nine-class loss is computed only on matched positive queries. At inference,
queries are ranked once by defectness and receive the argmax conditional class;
there is no class-wise duplication of the 300 queries.

The legacy one-class `dec_score_head.N.weight/bias` parameter names and shapes
are preserved. Consequently, `best_stg2.pth` initializes the objectness heads,
box heads, backbone, encoder, and decoder; only `class_weight/class_bias` and
the nine-class denoising embedding start new.

## PEFT ranking refinement

`dfine_hgnetv2_l_hierarchical_peft_aic.yml` starts from the trained
`best_stg1.pth` detector and preserves its localization representation:

- HGNetv2 and HybridEncoder are frozen;
- the original decoder attention/FFN weights are frozen;
- each decoder FFN receives a parallel zero-initialized 256→64→256
  AdaptFormer residual;
- decoder score heads and LayerNorm tensors remain trainable;
- bounding-box heads use a separate very small learning rate (`1e-5`).

Matched positive objectness losses are scaled by a square-root-tempered and
clamped class-frequency weight. Matching itself remains class agnostic. The
conditional nine-class loss adds `0.2 * log(class_prior)` to training logits;
raw inference logits and the objectness-only score definition are unchanged.

Run a migration/gradient check before training:

```bash
cd /root/autodl-tmp/AIC_steel_defect
/root/miniconda3/envs/aic-steel/bin/python scripts/install_hierarchical_dfine.py
/root/miniconda3/envs/aic-steel/bin/python scripts/smoke_test_hierarchical_dfine_train_step.py \
  --config configs/dfine/custom/objects365/dfine_hgnetv2_l_hierarchical_peft_aic.yml \
  --checkpoint /root/autodl-tmp/D-FINE/output/dfine_hgnetv2_l_hierarchical_aic/best_stg1.pth \
  --batch 1 --expect-peft
```

Train with `bash scripts/run_hierarchical_dfine_peft_train.sh`.  The validated
48 GiB configuration uses a total batch size of 96 for 15 epochs and disables
the strong augmentation policy after epoch 12.  Four workers are retained
after 8/16-worker pressure tests showed contention while preparing full
96-image batches.  The run saves every epoch for later PR-curve selection and
writes to the clean `_v2` output directory.

## Install and validate

```bash
cd /root/autodl-tmp/AIC_steel_defect
/root/miniconda3/envs/aic-steel/bin/python scripts/install_hierarchical_dfine.py
/root/miniconda3/envs/aic-steel/bin/python scripts/convert_mixed_yolo_to_coco_hierarchical.py --overwrite
/root/miniconda3/envs/aic-steel/bin/python scripts/smoke_test_hierarchical_dfine.py \
  --checkpoint /root/autodl-tmp/D-FINE/output/dfine_hgnetv2_l_obj2aic_defect/best_stg2.pth
```

Run one real forward/backward batch before full training:

```bash
/root/miniconda3/envs/aic-steel/bin/python scripts/smoke_test_hierarchical_dfine_train_step.py \
  --checkpoint /root/autodl-tmp/D-FINE/output/dfine_hgnetv2_l_obj2aic_defect/best_stg2.pth --batch 1
```

## Train

```bash
tmux new -s hierarchical_dfine
cd /root/autodl-tmp/AIC_steel_defect
bash scripts/run_hierarchical_dfine_train.sh
```

Detach with `Ctrl-b d`. The output is written to
`/root/autodl-tmp/D-FINE/output/dfine_hgnetv2_l_hierarchical_aic`.
