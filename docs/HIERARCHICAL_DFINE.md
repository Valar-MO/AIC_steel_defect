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
