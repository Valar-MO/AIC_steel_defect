# GL-Cascade Geometry Audit

The CPU-only audit is implemented in `scripts/audit_gl_cascade_geometry.py`.
It reads VOC XML annotations; it does not read the test set, create crop images,
or consume GPU memory.  Recommendations are derived from `train` only.  `val`
is reported separately as a holdout diagnostic.

Run on the server:

```bash
cd /root/autodl-tmp/AIC_steel_defect
/root/miniconda3/envs/aic-steel/bin/python scripts/audit_gl_cascade_geometry.py \
  --output-dir outputs/gl_cascade_geometry_audit \
  --tile-size 2048 --overlap 512 \
  --global-input 1024 --local-input 1536 \
  --anchor-clusters 5 --report-splits train val
```

The audit used to freeze `configs/gl_cascade/gl_cascade_full.yaml` found:

| Train-only measure | Result | Design consequence |
| --- | ---: | --- |
| GT with a completely visible official local crop | 4,786 / 4,855 (98.58%) | Local crops can be the primary detector input. |
| Local crop views with no >=50% visible GT | 10,593 / 15,360 (68.97%) | Empty local crops stay in training; positive-only tiles would create a false defect prior. |
| Local short side, median / p05 | 33.75 / 12.75 px | A real P2 at stride 4 is required in the local pathway. |
| Global short side, median / p05 | 12.63 / 4.75 px | The 1024 global pathway supplies context, not primary small-defect recall. |
| Non-completely-covered GT | 69 | Preserve a global fallback and visibility-aware crop supervision. |
| `zonglie` complete-crop rate | 325 / 383 (84.86%) | Long defects drive most crop-boundary failures. |

The class-balanced log-space anchor clustering gives width/height ratios
`[0.064, 0.188, 0.488, 1.224, 3.564]`.  The local 1536-pixel square-root-area
distribution has p05/p25/p50/p90/p95 of `17.4/32.1/45.8/109.4/160.4`, which
motivates P2-P6 base sizes `[16, 32, 64, 128, 256]`.
